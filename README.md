---
title: Smart Classroom Attendance
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# Smart Classroom Attendance System


# DataScience_Proj# 🎓 Smart Classroom Attendance System
### Powered by ArcFace CNN · MTCNN Detection · Cosine Similarity

A production-ready, automated attendance system built with Streamlit.
**No `face_recognition` library used** — recognition is handled by ArcFace (state-of-the-art CNN) via DeepFace.

---

## 📐 Architecture

```
Classroom Photo
      │
      ▼
 MTCNN Detector          ← detects all faces (bounding boxes)
      │
      ▼
 Preprocessing           ← resize → CLAHE → bilateral filter
      │
      ▼
 ArcFace CNN (512-d)     ← extract face embeddings
      │
      ▼
 L2 Normalisation
      │
      ▼
 Cosine Similarity       ← compare with student database
      │
      ▼
 Threshold Match         ← tunable (default 0.40)
      │
      ▼
 Excel Attendance        ← no duplicates per day
```

---

## 🚀 Local Setup (Run in 5 minutes)

### 1 · Prerequisites

```bash
Python 3.10 or 3.11  (recommended)
```

### 2 · Clone and install

```bash
git clone https://github.com/<your-username>/smart-attendance.git
cd smart-attendance

# Create a virtual environment (strongly recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> ⚠️ **First run** will download the ArcFace model weights (~500 MB) from DeepFace servers.
> This happens once and is cached automatically.

### 3 · Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖥️ GUI Walkthrough

### Sidebar
| Control | Description |
|---------|-------------|
| Navigation radio | Switch between modules |
| Similarity Threshold slider | Tune match strictness (0.10–0.90) |
| Live metrics | Registered students · Present today · Total records |

### 📸 Register Students
1. Enter the student's **Full Name**
2. Upload **1–5 clear face photos** (front-facing, good lighting)
3. Click **"⚡ Register Student"**
4. The app extracts an ArcFace 512-d embedding per photo and stores it in `data/student_embeddings.pkl`

**Pro tip:** Register with 3+ photos (slight angle variation) for best accuracy.

### 📋 Mark Attendance
1. Upload a **classroom group photo**
2. Select the **attendance date** (defaults to today)
3. Click **"🔍 Detect Faces & Mark Attendance"**
4. The app:
   - Detects all faces using MTCNN
   - Runs ArcFace on each face
   - Matches via cosine similarity
   - Records to `data/attendance.xlsx`
   - Shows annotated image with names + confidence scores

### 📊 View Records
- Filter by **date** or **student**
- View **attendance summary** with percentage
- Download as **CSV** or **formatted Excel**

### ⚙️ Settings
- Threshold guide table
- Preprocessing pipeline explanation
- Data management (clear records / students)

---

## ☁️ Deploy to Streamlit Cloud

### Step 1 — Push to GitHub

```bash
# Initialise git if not already done
git init
git add .
git commit -m "Initial commit: Smart Attendance System"

# Create a new repo on GitHub, then push
git remote add origin https://github.com/<your-username>/smart-attendance.git
git branch -M main
git push -u origin main
```

Make sure these files are present in the repo root:
```
smart-attendance/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── README.md
```

### Step 2 — Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"**
3. Select your repo → branch `main` → main file `app.py`
4. Click **"Deploy!"**

> ⚠️ **Important**: Streamlit Cloud has **ephemeral storage** — `data/` is wiped on each restart.
> For persistent storage, connect to an external database (see below).

### Step 3 — Persistent Storage (Production)

For a real classroom deployment, replace local pickle/Excel with:

| Option | How |
|--------|-----|
| **Google Sheets** | Use `gspread` + service account for attendance |
| **Firebase / Firestore** | Store embeddings + records in JSON documents |
| **AWS S3 / GCS** | Upload `student_embeddings.pkl` and `attendance.xlsx` after each write |
| **Supabase** | PostgreSQL-backed storage with a free tier |

Example — save to S3 after each write:
```python
import boto3
s3 = boto3.client("s3")
s3.upload_file(Config.EMBEDDINGS_FILE, "your-bucket", "student_embeddings.pkl")
```

---

## 🎯 Accuracy Tips

| Situation | Recommendation |
|-----------|----------------|
| Good indoor lighting | Threshold `0.40` |
| Variable / dim lighting | Threshold `0.35` + register more photos |
| High-resolution group photo | Works best with MTCNN |
| Faces < 80 px tall | Use a closer / higher-res camera |
| Multiple looks (glasses, mask) | Register one photo with and one without |

---

## 📂 Project Structure

```
smart-attendance/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme & server config
├── data/                   # Created automatically at runtime
│   ├── student_embeddings.pkl
│   ├── attendance.xlsx
│   └── student_images/
└── README.md
```

---

## 🔬 Technical Details

| Component | Choice | Why |
|-----------|--------|-----|
| Face Recognition CNN | **ArcFace** (512-d) | LFW accuracy 99.40%; best-in-class |
| Face Detector | **MTCNN** | Accurate multi-face; lightweight vs RetinaFace |
| Similarity Metric | **Cosine** | Rotation/scale invariant; standard for embeddings |
| Preprocessing | CLAHE + Bilateral | Handles classroom lighting variation |
| Storage | Pickle + Excel | Simple; portable; no database required |
| Multi-photo matching | Max similarity | Robust against pose/lighting mismatch |

---

## ❓ FAQ

**Q: First startup is slow, why?**
A: DeepFace downloads ArcFace weights (~500 MB) on first run. Subsequent runs use cache.

**Q: Can I use a webcam instead of uploading?**
A: Add `st.camera_input()` and pass the captured frame to `extract_all_faces()`.

**Q: How many students can it handle?**
A: The linear scan is fast up to ~500 students. Beyond that, use FAISS for ANN search.

**Q: What if two students look very similar (twins)?**
A: Lower the threshold to `0.30` and register more diverse photos for each student.