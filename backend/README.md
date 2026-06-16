Flask backend for FaceVault demo

Quick start:

1. Create a virtual environment and install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the app:

```powershell
python app.py
```

3. Open http://127.0.0.1:5000 in your browser. The frontend is served from the `frontend/` folder.

Notes:
- This backend uses a lightweight SQLite DB at `backend/data.db` and stores uploads under `backend/uploads/`.
- The face detection is a placeholder (random counts). Replace with a real model for production.
