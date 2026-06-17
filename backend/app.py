from flask import Flask, request, jsonify, send_from_directory, session
import os
import sqlite3
import hashlib
import io
import zipfile
import logging
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash

from database import db, init_db
from models import Image
from ml_model import predict_category

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / 'frontend'
DB_PATH = BASE_DIR / 'data.db'
UPLOAD_FOLDER = BASE_DIR / 'uploads'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='')
app.secret_key = os.environ.get('FLASK_SECRET') or 'dev-secret-change-me'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

init_db(app)

EMOJI_MAP = {
    'Individual Person': '👤',
    'Vehicle': '🚗',
    'Food': '🍕',
    'Pet': '🐾',
    'Fish': '🐟',
    'Flowers': '🌸',
    'Place/Landmark': '🏛️',
    'Environment': '🌿',
    'Uncategorized': '❓',
}


# ── Raw SQLite helper (for auth only) ──────────────────────────────
def get_raw_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_users_table():
    with get_raw_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                first_name TEXT,
                last_name TEXT,
                username TEXT UNIQUE,
                email TEXT UNIQUE,
                password_hash TEXT
            )
        ''')


init_users_table()


# ── Auth ────────────────────────────────────────────────────────────
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    first = data.get('firstName', '').strip()
    last = data.get('lastName', '').strip()
    username = data.get('username', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    if not first or not username or not email or len(password) < 8:
        return jsonify({'success': False, 'error': 'Invalid input'}), 400
    pw_hash = generate_password_hash(password)
    try:
        with get_raw_db() as conn:
            cur = conn.execute(
                'INSERT INTO users (first_name,last_name,username,email,password_hash) VALUES (?,?,?,?,?)',
                (first, last, username, email, pw_hash)
            )
            session['user_id'] = cur.lastrowid
        return jsonify({'success': True, 'user_id': session['user_id']})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'error': 'User or email already exists'}), 400


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    if not email or not password:
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 400
    with get_raw_db() as conn:
        row = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    if not row or not check_password_hash(row['password_hash'], password):
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
    session['user_id'] = row['id']
    return jsonify({'success': True, 'user_id': row['id']})


@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


@app.route('/api/me', methods=['GET'])
def me():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    with get_raw_db() as conn:
        row = conn.execute(
            'SELECT first_name, last_name, username, email FROM users WHERE id = ?', (user_id,)
        ).fetchone()
    if not row:
        return jsonify({'success': False, 'error': 'User not found'}), 404
    return jsonify({
        'success': True,
        'first_name': row['first_name'],
        'last_name': row['last_name'],
        'username': row['username'],
        'email': row['email'],
    })


# ── Upload (ML-powered) ─────────────────────────────────────────────
def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _classify_and_save(path, user_id):
    file_hash = _sha256(path)
    existing = Image.query.filter_by(user_id=user_id, file_hash=file_hash).first()
    if existing:
        return {
            'filename': path.name,
            'category': existing.category,
            'confidence': existing.confidence,
            'duplicate': True,
        }
    result = predict_category(str(path))
    img = Image(
        user_id=user_id,
        filename=path.name,
        filepath=f"{user_id}/{path.name}",
        category=result['category'],
        confidence=result['confidence'],
        file_hash=file_hash,
        file_size=path.stat().st_size,
    )
    db.session.add(img)
    db.session.commit()
    return {
        'filename': path.name,
        'category': result['category'],
        'confidence': result['confidence'],
        'duplicate': False,
    }


@app.route('/api/upload', methods=['POST'])
def upload_files():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    files = request.files.getlist('files')
    if not files:
        return jsonify({'success': False, 'error': 'No files uploaded'}), 400

    user_dir = UPLOAD_FOLDER / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for f in files:
        if f.filename.lower().endswith('.zip'):
            try:
                zdata = io.BytesIO(f.read())
                with zipfile.ZipFile(zdata) as z:
                    for member in z.namelist():
                        if member.endswith('/'):
                            continue
                        out = user_dir / Path(member).name
                        with z.open(member) as src, open(out, 'wb') as dst:
                            dst.write(src.read())
                        saved.append(_classify_and_save(out, user_id))
            except Exception as e:
                logger.error(f'ZIP processing error: {e}')
        else:
            out = user_dir / Path(f.filename).name
            f.save(out)
            saved.append(_classify_and_save(out, user_id))

    return jsonify({'success': True, 'uploaded': len(saved), 'results': saved})


# ── ML result endpoints ─────────────────────────────────────────────
@app.route('/api/categories', methods=['GET'])
def list_categories():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    from sqlalchemy import func
    rows = (
        db.session.query(Image.category, func.count(Image.id).label('count'))
        .filter(Image.user_id == user_id)
        .group_by(Image.category)
        .order_by(func.count(Image.id).desc())
        .all()
    )
    categories = [
        {'category': r.category, 'count': r.count, 'emoji': EMOJI_MAP.get(r.category, '🖼️')}
        for r in rows
    ]
    return jsonify({'success': True, 'categories': categories})


@app.route('/api/images', methods=['GET'])
def list_images():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    category = request.args.get('category')
    query = Image.query.filter_by(user_id=user_id)
    if category:
        query = query.filter_by(category=category)
    images = query.order_by(Image.upload_date.desc()).all()
    return jsonify({'success': True, 'images': [img.to_dict() for img in images]})


@app.route('/api/stats', methods=['GET'])
def stats():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    from sqlalchemy import func
    total_photos = Image.query.filter_by(user_id=user_id).count()
    total_categories = (
        db.session.query(func.count(func.distinct(Image.category)))
        .filter(Image.user_id == user_id)
        .scalar() or 0
    )
    person_photos = Image.query.filter_by(user_id=user_id, category='Individual Person').count()
    return jsonify({
        'success': True,
        'total_photos': total_photos,
        'total_categories': total_categories,
        'person_photos': person_photos,
    })


# Kept for backward compat (dashboard Top People section)
@app.route('/api/persons', methods=['GET'])
def list_persons():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    from sqlalchemy import func
    rows = (
        db.session.query(Image.category, func.count(Image.id).label('count'))
        .filter(Image.user_id == user_id)
        .group_by(Image.category)
        .order_by(func.count(Image.id).desc())
        .all()
    )
    persons = [
        {
            'id': i + 1,
            'name': r.category,
            'count': r.count,
            'emoji': EMOJI_MAP.get(r.category, '🖼️'),
            'unknown': r.category == 'Uncategorized',
        }
        for i, r in enumerate(rows)
    ]
    return jsonify({'success': True, 'persons': persons})


# ── Static file serving ─────────────────────────────────────────────
@app.route('/uploads/<path:filepath>')
def uploaded_file(filepath):
    return send_from_directory(str(UPLOAD_FOLDER), filepath)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def static_proxy(path):
    if path == '':
        path = 'index.html'
    target = FRONTEND_DIR / path
    if target.exists() and target.is_file():
        return send_from_directory(str(FRONTEND_DIR), path)
    return send_from_directory(str(FRONTEND_DIR), 'index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
