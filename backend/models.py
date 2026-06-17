"""
====================================================================
 models.py – SQLAlchemy Database Models
====================================================================
"""

from datetime import datetime
from database import db


class Image(db.Model):
    __tablename__ = "images"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, nullable=True, index=True)

    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)

    category = db.Column(db.String(50), nullable=False, default="Uncategorized", index=True)
    confidence = db.Column(db.Float, nullable=False, default=0.0)

    file_hash = db.Column(db.String(64), nullable=False, index=True)
    file_size = db.Column(db.Integer, nullable=False, default=0)

    upload_date = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "filepath": self.filepath,
            "url": f"/uploads/{self.filepath}",
            "category": self.category,
            "confidence": self.confidence,
            "file_size": self.file_size,
            "upload_date": self.upload_date.strftime("%Y-%m-%d %H:%M:%S") if self.upload_date else None,
        }

    def __repr__(self):
        return f"<Image id={self.id} filename={self.filename!r} category={self.category!r}>"
