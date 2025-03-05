cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

<br>
<br>

pip install -r requirements.txt
pip install fastapi uvicorn

uvicorn main:app --reload --host 127.0.0.1 --port 8000
