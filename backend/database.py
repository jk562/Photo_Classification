"""
====================================================================
 database.py – Database Initialization (SQLAlchemy)
====================================================================
"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def init_db(app):
    db.init_app(app)
    with app.app_context():
        from models import Image  # noqa: F401
        db.create_all()
