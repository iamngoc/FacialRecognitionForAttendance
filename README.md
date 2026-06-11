# 🍹 Sleepy Durian World – Facial Recognition Time Recording

**Project** | Windows + CPU | ArcFace + Docker

---

## 🗺️ Overview

```
Camera → ArcFace AI → Person recognized? → Log time in DB → Dashboard
```

**Technology:** Python 3.11 · InsightFace/ArcFace (buffalo_l) · FastAPI · PostgreSQL + pgvector · Docker · Nginx

---

## ⚙️ Step 1: Install Prerequisites

### Docker Desktop
1. https://www.docker.com/products/docker-desktop/ → Download
2. Install, enable WSL 2, restart
3. Check: `docker --version` (in PowerShell)

### Python 3.11
1. https://www.python.org/downloads/ → Python 3.11
2. **☑️ Check "Add Python to PATH"!**
3. Check: `python --version`

---

## 🚀 Step 2: Start the System

```powershell
# Open project folder
cd C:\Users\%USERNAME%\Desktop\TimeRecordingwithfacialRecognition

# First start: delete old volume to ensure init.sql runs cleanly
docker volume rm sd_db_data

# Build and start (first time ~10-15 minutes)
docker-compose up --build
```

Wait until you see:
```
sd_backend  | System ready.
```

**Check in browser:**

| URL | What you see |
|-----|-------------|
| http://localhost:8000 | API status JSON |
| http://localhost:8000/docs | Interactive API docs (Swagger) |
| http://localhost:3000 | Dashboard |

---

## 🧑‍🤝‍🧑 Step 3: Register Employees (Enrollment)

Open a **new** PowerShell window:

```powershell
cd C:\Users\%USERNAME%\Desktop\TimeRecordingwithfacialRecognition\scripts

# Install Python packages (only once)
pip install insightface onnxruntime opencv-python requests numpy

# Register via webcam (9 photos)
python enrollment_script.py --id 1 --name "Sleepy Durian"

# Register from a photo folder (alternative)
python enrollment_script.py --id 1 --name "Sleepy Durian" --folder "C:/photos"
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `--id` | ✅ Yes | Employee ID (must exist in database) |
| `--name` | ✅ Yes | Employee name (for logging) |
| `--folder` | ❌ No | Path to folder with photos — skips webcam |

**Webcam instructions:**
- Camera window opens automatically
- **SPACE** = take photo
- **q** = cancel enrollment
- 9 photos total: front, left, right, up, down, smile, neutral, smile again, front again

---

## 📷 Step 4: Start Live Camera

```powershell
python camera_loop.py
```

Stand in front of the camera → system recognizes you → time is logged automatically!

Press **`q`** in the camera window to quit.

---

## 🗄️ Check Database Directly

```powershell
# Show all employees (and whether they have an embedding registered)
docker exec -it sd_database psql -U sd_admin -d sd_timerecording -c \
  "SELECT id, name, email, (embedding IS NOT NULL) AS registered FROM employees;"

# Today's time recording
docker exec -it sd_database psql -U sd_admin -d sd_timerecording -c \
  "SELECT e.name, t.scan_type, t.scan_time, t.confidence_score \
   FROM timerecording t JOIN employees e ON t.employee_id = e.id \
   ORDER BY t.scan_time;"

# Working hours summary (via view)
docker exec -it sd_database psql -U sd_admin -d sd_timerecording -c \
  "SELECT * FROM office_hours;"
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | System status |
| GET | `/health` | Docker health check |
| GET | `/employees` | List all active employees |
| POST | `/scan` | Recognize face + log time |
| POST | `/enrollment/{employee_id}` | Register employee with photos |
| GET | `/report/today` | All scans for today |
| GET | `/report/week/{employee_id}` | Scans for last 7 days |

Full interactive docs: **http://localhost:8000/docs**

---

## 🔧 Common Issues

| Problem                       | Solution                                                                    |
|-------------------------------|-----------------------------------------------------------------------------|
| `docker: command not found`   | Start Docker Desktop, wait until icon is green                              |
| `port 5432 in use`            | Stop local PostgreSQL or change port in `docker-compose.yml`                |
| Camera not found              | Try `KAMERA_INDEX=1 python camera_loop.py`                                  |
| `No module named insightface` | Don't forget to run `pip install ...` first                                 |
| Model download slow           | One-time ~200 MB download — cached afterwards                               |
| Employee not found (404)      | Run `docker volume rm sd_db_data` then `docker-compose up --build`          |
| Tables empty after restart    | Volume was not fully deleted — use `docker volume rm sd_db_data` explicitly |
| Check table Employee          | run `docker exec -it sd_database psql -U sd_admin -d sd_timerecording -c "SELECT id, name, email FROM employees;"`|

---

## 📁 Project Structure

```
TimeRecordingwithfacialRecognization/
├── docker-compose.yml              # Start everything with 1 command
├── database/
│   └── init.sql                    # DB schema + initial employee data
├── backend/
│   ├── Dockerfile                  # Container recipe (python:3.11-slim-bookworm)
│   ├── requirements.txt            # Python packages
│   ├── database_model.py           # SQLAlchemy models (Employee, TimeRecording)
│   ├── database.py                 # DB connection + session factory
│   ├── crud_db_operation.py        # DB read/write operations
│   ├── faceEngine.py               # ArcFace AI engine (buffalo_l, CPU)
│   └── main.py                     # FastAPI web server
├── frontend/
│   ├── index.html                  # Dashboard
│   └── nginx.conf                  # Nginx web server config
├── scripts/
│   ├── enrollment_script.py        # Register employees (webcam or folder)
│   └── camera_loop.py              # Live camera recognition loop
└── data/
    ├── Employee_fotos/           # Photos for enrollment
    └── snapshots/                   # Scan snapshots
```

---

## ⚙️ Configuration (docker-compose.yml)

| Variable | Default             | Description |
|----------|---------------------|-------------|
| `THRESHOLD` | `0.45`              | Minimum similarity score to recognize a person (0.0–1.0) |
| `ANTI_SPAM_SEC` | `15`                | Minimum seconds between two scans of the same person |
| `DATABASE_URL` | (set automatically) | PostgreSQL connection string |

---

## 🗃️ Database Schema

**Table `employees`** — stores staff + AI facial fingerprint
- `id`, `name`, `email`, `department`, `position_`
- `embedding` — 512-dimensional vector (ArcFace fingerprint)
- `active`, `entry_date`, `created_in`, `updated_in`

**Table `timerecording`** — every recognized scan
- `employee_id`, `scan_time`, `scan_date`
- `scan_type` — `COME` or `GO`
- `confidence_score`, `camera_id`

**View `office_hours`** — working hours summary per employee per day
- arrival (first COME), exit (last GO), total working hours

---

## Notes

1. **PostgreSQL + pgvector** stores who work at Sleepy Durian World, when they arrive, and their facial fingerprint as 512 numbers
1. **PostgreSQL + pgvector** stores who work at Sleepy Durian World, when they arrive, and their facial fingerprint as 512 numbers
2. **ArcFace** turns every face into 512 numbers — a unique "fingerprint" — using the `buffalo_l` AI model
3. **FastAPI** is the "waiter" that receives requests from the camera and returns answers
4. **Docker** packs everything into boxes that work the same on every computer
5. **The Dashboard** shows everything nicely in the browser at http://localhost:3000

