cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

<br>
<br>

pip install -r requirements.txt<br>
pip install fastapi uvicorn

uvicorn main:app --reload --host 127.0.0.1 --port 8000

```
Back
├─ alembic
│  ├─ env.py
│  ├─ README
│  ├─ script.py.mako
│  └─ versions
│     └─ 8e2f95869469_create_image_log_tables.py
├─ alembic.ini
├─ app
│  ├─ auth.py
│  ├─ config.py
│  ├─ db.py
│  ├─ main.py
│  ├─ models
│  │  ├─ db_models.py
│  │  └─ request_models.py
│  ├─ routers
│  │  ├─ img2img.py
│  │  ├─ inpaint.py
│  │  ├─ logs.py
│  │  └─ txt2img.py
│  ├─ services
│  │  ├─ image_generator.py
│  │  ├─ img2img
│  │  │  ├─ edge_copy.py
│  │  │  └─ pose_copy.py
│  │  └─ inpaint
│  │     ├─ erase_object.py
│  │     ├─ object_detect.py
│  │     └─ redraw_object.py
│  └─ utils
│     └─ s3.py
├─ db.sqlite
├─ README.md
└─ requirements.txt

```