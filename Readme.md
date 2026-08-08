# Dashcam Safety Monitor

Real-Time Multi-Model Vision Pipeline for Road Hazard Detection, Priority Bounding Box Overlay, and Audio Emergency Alerts.

This is an academic, advisory driver-assistance demonstration. It does not control steering,
braking, throttle, or vehicle movement.

---

## Architecture
![Architecture Diagram](./docs/architecture.png)

## Run application using Docker

```bash
docker compose up --build -d
```

```bash
docker compose down -v
```

## Run the application locally

### Prerequisites
- Node.js 24 or higher
- Python 3.10 or higher

---

### 1. Frontend Setup (Windows / macOS / Linux)

Open a terminal and run:

```bash
cd dashcam-safety-monitor-frontend
npm install
npm run dev
```
The frontend will start at `http://localhost:3000`.

---

### 2. Backend Setup

#### **Windows (Command Prompt / CMD)**
```cmd
cd dashcam-safety-monitor-backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

#### **Windows (PowerShell)**
```powershell
cd dashcam-safety-monitor-backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

#### **macOS / Linux / Git Bash**
```bash
cd dashcam-safety-monitor-backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
The backend API will start at `http://localhost:8000`.