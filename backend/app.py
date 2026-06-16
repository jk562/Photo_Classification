from flask import Flask, request, jsonify, send_from_directory, session
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from pathlib import Path
import random
import zipfile
import io
import cv2

def detect_faces(image_path):
    # Load the cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(str(image_path))
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / 'frontend'
DB_PATH = BASE_DIR / 'data.db'
UPLOAD_FOLDER = BASE_DIR / 'uploads'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='')
app.secret_key = os.environ.get('FLASK_SECRET') or 'dev-secret-change-me'


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as db:
        db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                first_name TEXT,
                last_name TEXT,
                username TEXT UNIQUE,
                email TEXT UNIQUE,
                password_hash TEXT
            )
        ''')
        db.execute('''
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                filename TEXT,
                faces_count INTEGER,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')


init_db()


@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    first = data.get('firstName','').strip()
    last = data.get('lastName','').strip()
    username = data.get('username','').strip()
    email = data.get('email','').strip().lower()
    password = data.get('password','')
    if not first or not username or not email or len(password) < 8:
        return jsonify({'success': False, 'error': 'Invalid input'}), 400
    pw_hash = generate_password_hash(password)
    try:
        with get_db() as db:
            cur = db.execute('INSERT INTO users (first_name,last_name,username,email,password_hash) VALUES (?,?,?,?,?)',
                             (first,last,username,email,pw_hash))
            user_id = cur.lastrowid
            session['user_id'] = user_id
        return jsonify({'success': True, 'user_id': user_id})
    except sqlite3.IntegrityError as e:
        return jsonify({'success': False, 'error': 'User or email already exists'}), 400


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    email = data.get('email','').strip().lower()
    password = data.get('password','')
    if not email or not password:
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 400
    with get_db() as db:
        cur = db.execute('SELECT * FROM users WHERE email = ?', (email,))
        row = cur.fetchone()
        if not row or not check_password_hash(row['password_hash'], password):
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
        session['user_id'] = row['id']
        return jsonify({'success': True, 'user_id': row['id']})


@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


@app.route('/api/upload', methods=['POST'])
def upload_files():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    files = request.files.getlist('files')
    if not files:
        return jsonify({'success': False, 'error': 'No files uploaded'}), 400
    saved = []
    user_dir = UPLOAD_FOLDER / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    with get_db() as db:
        for f in files:
            filename = f.filename
            if filename.lower().endswith('.zip'):
                # extract zip
                try:
                    zdata = io.BytesIO(f.read())
                    with zipfile.ZipFile(zdata) as z:
                        for member in z.namelist():
                            if member.endswith('/'):
                                continue
                            out = user_dir / Path(member).name
                            with z.open(member) as src, open(out, 'wb') as dst:
                                dst.write(src.read())
                            faces = detect_faces(out)
                            db.execute('INSERT INTO photos (user_id,filename,faces_count) VALUES (?,?,?)', (user_id, out.name, faces))
                            saved.append({'filename': out.name, 'faces': faces})
                except Exception as e:
                    continue
            else:
                out = user_dir / Path(filename).name
                f.save(out)
                faces = detect_faces(out)
                db.execute('INSERT INTO photos (user_id,filename,faces_count) VALUES (?,?,?)', (user_id, out.name, faces))
                saved.append({'filename': out.name, 'faces': faces})
    return jsonify({'success': True, 'uploaded': len(saved), 'results': saved})


@app.route('/api/persons', methods=['GET'])
def list_persons():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    # Simple grouping by faces_count as a placeholder
    with get_db() as db:
        cur = db.execute('SELECT faces_count, COUNT(*) as count FROM photos WHERE user_id = ? GROUP BY faces_count ORDER BY count DESC', (user_id,))
        rows = cur.fetchall()

    persons = []
    i = 1
    for r in rows:
        faces_count = r['faces_count']
        if faces_count == 1:
            name = "Single Face"
        elif faces_count == 2:
            name = "Two Faces"
        elif faces_count == 0:
            name = "No Faces"
        else:
            name = f"Group of {faces_count}"

        persons.append({
            'id': i,
            'name': name,
            'count': r['count'],
            'emoji': '👤' if faces_count > 0 else '❓',
            'unknown': faces_count == 0
        })
        i += 1
    return jsonify({'success': True, 'persons': persons})


# Serve frontend files
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
