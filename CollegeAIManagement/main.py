import streamlit as st
import sqlite3
import pandas as pd
import os
from datetime import datetime, date, time, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from difflib import get_close_matches

# -------------------------------------------------
# Page config + Accessible, high-contrast theme
# -------------------------------------------------
st.set_page_config(page_title="College AI Portal", page_icon="🎓", layout="wide")

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&family=Source+Sans+3:wght@300;400;600;700&display=swap');

:root{
  --font-body: 'Source Sans 3', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Ubuntu, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif;
  --font-head: 'Montserrat', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Ubuntu, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif;

  /* LIGHT MODE */
  --bg: #ffffff;
  --bg-elev: #f8fafc;        /* slate-50 */
  --text: #0b1220;           /* near-black */
  --muted: #334155;          /* slate-700 */
  --accent: #2563eb;         /* blue-600 */
  --accent-2: #0ea5e9;       /* sky-500 */
  --border: #e5e7eb;         /* gray-200 */
  --chip: #eef2ff;           /* indigo-50 */
  --bubble-user: #eef2ff;    /* tinted */
  --bubble-assist: #f1f5f9;  /* slate-100 */
}

@media (prefers-color-scheme: dark) {
  :root{
    /* DARK MODE */
    --bg: #0b1220;           /* deep navy */
    --bg-elev: #0f172a;      /* slate-900 */
    --text: #f8fafc;         /* almost white */
    --muted: #94a3b8;        /* slate-400 */
    --accent: #60a5fa;       /* blue-400 */
    --accent-2: #34d399;     /* emerald-400 */
    --border: #1f2937;       /* slate-800 */
    --chip: #111827;         /* slate-900 */
    --bubble-user: rgba(99,102,241,0.22);    /* indigo-500 @ 22% */
    --bubble-assist: rgba(148,163,184,0.22); /* slate-400 @ 22% */
  }
}

/* App background + typography */
html, body, [data-testid="stAppViewContainer"]{
  background: var(--bg) !important;
}
html, body, [data-testid="stAppViewContainer"] *{
  color: var(--text);
  font-family: var(--font-body);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3{
  font-family: var(--font-head);
  letter-spacing: .2px;
  color: var(--text);
}

/* Gradient title with shadow fallback for dark bg */
h1 span.gradient-title{
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  text-shadow: 0 0 0.6px rgba(0,0,0,0.5);
}

/* Sidebar contrast */
section[data-testid="stSidebar"] {
  background: var(--bg-elev) !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] *{ font-family: var(--font-body); }

/* Inputs / buttons */
.stButton > button,
.stTextInput input,
.stDateInput input,
.stTimeInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] *{
  font-family: var(--font-body) !important;
  color: var(--text) !important;
}
.stButton > button{
  border: 1px solid var(--border) !important;
}

/* Tables and dataframes */
[data-testid="stTable"], [data-testid="stDataFrame"] {
  background: var(--bg-elev);
  border-radius: 12px;
  border: 1px solid var(--border);
  padding: 6px;
}
[data-testid="stDataFrame"] * { color: var(--text) !important; }

/* Metrics pop a bit */
[data-testid="stMetricValue"], [data-testid="stMetricDelta"]{
  font-weight: 700;
}

/* Chat bubbles */
div[data-testid="stChatMessage"]{
  border-radius: 12px;
  padding: 10px 12px;
  border: 1px solid var(--border);
  background: var(--bubble-assist);
}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"]:has(svg[aria-label="user"])) {
  background: var(--bubble-user);
}

/* Header transparency (leave default) */
[data-testid="stHeader"] { background: transparent; }

/* Links */
a, .stMarkdown a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# =========================
# Bootstrap DB & Utilities
# =========================
DB_DIR = "database"
DB_PATH = os.path.join(DB_DIR, "college_ai.db")
os.makedirs(DB_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# Tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    role TEXT,
    student_id INTEGER,
    first_login INTEGER DEFAULT 1
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    department TEXT,
    semester TEXT,
    age INTEGER
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS subjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS timetable (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    subject_id INTEGER,
    start_time TEXT,
    end_time TEXT
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    date TEXT,
    subject_id INTEGER,
    status TEXT
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS marks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    subject_id INTEGER,
    marks INTEGER
)
""")
conn.commit()

# ---- Add first_login column safely (if an older DB exists) ----
try:
    cursor.execute("ALTER TABLE users ADD COLUMN first_login INTEGER DEFAULT 1")
    conn.commit()
except sqlite3.OperationalError:
    pass  # already exists

# Default admin (admin/admin) with first_login=0
cursor.execute("SELECT 1 FROM users WHERE username='admin' AND role='admin'")
if not cursor.fetchone():
    cursor.execute(
        "INSERT INTO users (username, password, role, student_id, first_login) VALUES (?, ?, 'admin', NULL, 0)",
        ('admin', 'admin')
    )
    conn.commit()

# Normalize first_login flags
cursor.execute("UPDATE users SET first_login=0 WHERE role='admin' AND (first_login IS NULL OR first_login<>0)")
cursor.execute("UPDATE users SET first_login=1 WHERE role='user' AND first_login IS NULL")
conn.commit()

# Session state init
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.session_state.student_id = None

# ---- Helpers ----
def generate_unique_username(name: str) -> str:
    base = (name or "").replace(" ", "") + "123"
    base = base if base else "student123"
    username = base
    counter = 1
    while cursor.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
        username = f"{base}_{counter}"
        counter += 1
    return username

def safe_rerun():
    try:
        st.rerun()
    except Exception:
        pass

def attendance_percentage_for_student(sid: int) -> float:
    total = cursor.execute("SELECT COUNT(*) FROM attendance WHERE student_id=?", (sid,)).fetchone()[0]
    if total == 0:
        return 0.0
    present = cursor.execute(
        "SELECT COUNT(*) FROM attendance WHERE student_id=? AND status='Present'",
        (sid,)
    ).fetchone()[0]
    return (present / total) * 100.0

def predicted_marks_for_student(sid: int) -> float:
    pct = attendance_percentage_for_student(sid)
    rows = cursor.execute("SELECT marks FROM marks WHERE student_id=?", (sid,)).fetchall()
    if not rows:
        return 0.0
    y = np.array([r[0] for r in rows], dtype=float)
    X = np.array([pct] * len(y), dtype=float).reshape(-1, 1)
    try:
        model = LinearRegression().fit(X, y)
        pred = float(model.predict(np.array([[pct]], dtype=float))[0])
        return max(0.0, min(100.0, pred))
    except Exception:
        return float(np.mean(y))

def subject_map() -> dict:
    df = pd.read_sql("SELECT id, name FROM subjects", conn)
    return {row["name"]: int(row["id"]) for _, row in df.iterrows()}

def subject_id_to_name_map() -> dict:
    df = pd.read_sql("SELECT id, name FROM subjects", conn)
    return {int(row["id"]): row["name"] for _, row in df.iterrows()}

def parse_time_str(ts: str) -> time:
    if ts is None or str(ts).strip() == "":
        return time(9, 0)
    s = str(ts).strip()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(s, fmt).time()
        except Exception:
            continue
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return time(9, 0)
        dt_py = dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt
        return getattr(dt_py, "time", lambda: time(9, 0))()
    except Exception:
        return time(9, 0)

def parse_date_str(ds: str) -> date:
    if ds is None or str(ds).strip() == "":
        return date.today()
    dt = pd.to_datetime(ds, errors="coerce")
    if pd.isna(dt):
        return date.today()
    try:
        return dt.date()
    except Exception:
        if isinstance(dt, date):
            return dt
        return date.today()

# ---------- Chatbot: ML Intent Classifier + Helpers ----------
INTENT_SAMPLES = {
    "attendance_overall": [
        "my attendance", "overall attendance", "how much attendance do i have",
        "attendance percentage", "check attendance", "show my attendance"
    ],
    "attendance_subject": [
        "attendance in math", "attendance for DBMS", "subject attendance",
        "attendance for data structures", "how is my attendance in operating systems"
    ],
    "marks_overall": [
        "my marks", "overall marks", "average score", "show my grades", "results please"
    ],
    "marks_subject": [
        "marks in dbms", "score for maths", "grade in physics", "what are my marks in os"
    ],
    "timetable_today": [
        "today's classes", "classes today", "today timetable", "what is my schedule today"
    ],
    "timetable_date": [
        "timetable on 05-10-2025", "classes tomorrow", "schedule for 2025-10-05", "classes on friday"
    ],
    "timetable_next": [
        "upcoming classes", "next classes", "what's next", "next few classes"
    ],
    "profile_info": [
        "my details", "profile", "who am i", "show my info"
    ],
    "change_password": [
        "change my password", "set a new password", "update password", "password change"
    ],
    "help": [
        "help", "what can you do", "how to use this", "commands"
    ]
}

_vectorizer = None
_intent_centroids = None
_intent_labels = None

def build_intent_model():
    """Build a TF-IDF model and dense centroids per intent (no np.matrix)."""
    global _vectorizer, _intent_centroids, _intent_labels
    docs, labs = [], []
    for lab, phrases in INTENT_SAMPLES.items():
        docs += phrases
        labs += [lab] * len(phrases)

    _vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", dtype=np.float64)
    X = _vectorizer.fit_transform(docs)  # csr_matrix

    from collections import defaultdict
    bucket = defaultdict(list)
    for i, lab in enumerate(labs):
        bucket[lab].append(i)

    centroids = []
    labels = []
    for lab, idxs in bucket.items():
        c = X[idxs].mean(axis=0)          # 1 x n (matrix)
        c = np.asarray(c, dtype=np.float64).ravel()  # convert to 1D ndarray
        centroids.append(c)
        labels.append(lab)

    _intent_centroids = np.vstack(centroids)  # (k, n_features)
    _intent_labels = labels

def detect_intent_rule(q: str) -> str:
    ql = (q or "").lower()
    if "change" in ql and "password" in ql: return "change_password"
    if "attendance" in ql:
        if "subject" in ql or " for " in ql or " in " in ql: return "attendance_subject"
        return "attendance_overall"
    if any(k in ql for k in ["mark", "score", "grade", "result"]):
        if "subject" in ql or " for " in ql or " in " in ql: return "marks_subject"
        return "marks_overall"
    if any(k in ql for k in ["timetable", "class", "schedule", "classes"]):
        if "today" in ql: return "timetable_today"
        if "tomorrow" in ql or re.search(r"\d{4}-\d{1,2}-\d{1,2}", ql) or re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{4}", ql):
            return "timetable_date"
        if "next" in ql or "upcoming" in ql: return "timetable_next"
        return "timetable_today"
    if any(k in ql for k in ["my details", "profile", "who am i"]): return "profile_info"
    return "help"

def detect_intent_ml(q: str) -> str:
    """Cosine against dense centroids; fallback to rules if score is weak."""
    if not q or not q.strip():
        return "help"

    global _vectorizer, _intent_centroids, _intent_labels
    if _vectorizer is None or _intent_centroids is None:
        build_intent_model()

    v = _vectorizer.transform([q]).toarray()   # (1, n_features)
    C = _intent_centroids                      # (k, n_features)

    dots = (v @ C.T)[0]                        # (k,)
    v_norm = np.linalg.norm(v)
    C_norms = np.linalg.norm(C, axis=1) + 1e-12
    scores = dots / (max(v_norm, 1e-12) * C_norms)

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_label = _intent_labels[best_idx]

    if best_score < 0.18:
        return detect_intent_rule(q)
    return best_label

def parse_natural_date(q):
    q = (q or "").lower().strip()
    if not q: return None
    if "today" in q: return date.today()
    if "tomorrow" in q: return date.today() + timedelta(days=1)
    if "yesterday" in q: return date.today() - timedelta(days=1)
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", q)
    if m:
        d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try: return date(y, mth, d)
        except: return None
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", q)
    if m:
        y, mth, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try: return date(y, mth, d)
        except: return None
    return None

def fuzzy_find_subject(q, subject_names):
    if not subject_names: return None
    for s in subject_names:
        if s.lower() in (q or "").lower():
            return s
    match = get_close_matches(q or "", subject_names, n=1, cutoff=0.6)
    return match[0] if match else None

def extract_password(q):
    m = re.search(r"change\s+my\s+password\s+(to|as)\s+([^\s].+)$", (q or "").lower())
    if not m: return None
    return m.group(2).strip()

# ==============
# Title
# ==============
st.markdown("""
<h1 style="text-align:center; font-family:'Montserrat',sans-serif; font-weight:700; margin-top:-6px;">
  <span class="gradient-title">
    🎓 College AI Portal
  </span>
</h1>
""", unsafe_allow_html=True)

# ==============
# Login Screen
# ==============
if not st.session_state.logged_in:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    role_choice = st.radio("Login as", ["Admin", "User"], horizontal=True)

    if st.button("Login"):
        if role_choice == "Admin":
            cursor.execute(
                "SELECT * FROM users WHERE username=? AND password=? AND role='admin'",
                (username, password)
            )
        else:
            cursor.execute(
                "SELECT * FROM users WHERE username=? AND password=? AND role='user'",
                (username, password)
            )
        user = cursor.fetchone()
        if user:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = user[3]
            st.session_state.student_id = user[4]
            safe_rerun()
        else:
            st.error("Invalid username or password")

# ========================
# Logged-In Application
# ========================
if st.session_state.logged_in:
    role = st.session_state.role
    student_id = st.session_state.student_id

    # ---- Force password change on first login (users only) ----
    if role == "user":
        row = cursor.execute("SELECT first_login FROM users WHERE username=?", (st.session_state.username,)).fetchone()
        first_login_flag = (row and row[0] in (1, "1"))
        if first_login_flag:
            st.warning("First-time login: please set a new password to continue.")
            with st.form("first_pw_form"):
                new_pw = st.text_input("New Password", type="password")
                new_pw2 = st.text_input("Confirm New Password", type="password")
                submit_pw = st.form_submit_button("Update Password")
            if submit_pw:
                if len(new_pw) < 4:
                    st.error("Password must be at least 4 characters.")
                elif new_pw != new_pw2:
                    st.error("Passwords do not match.")
                else:
                    cursor.execute("UPDATE users SET password=?, first_login=0 WHERE username=?", (new_pw, st.session_state.username))
                    conn.commit()
                    st.success("Password updated. You're good to go!")
                    safe_rerun()
            st.stop()  # Block access until updated

    # Sidebar menu
    menu = ["Home"]
    if role == "admin":
        menu += [
            "Students", "Subjects", "Timetable",
            "Attendance", "Marks", "AI Dashboard",
            "Manage Users", "Logout"
        ]
    else:
        menu += [
            "My Details", "My Attendance", "Timetable",
            "Predict Performance", "Chatbot", "Logout"
        ]
    choice = st.sidebar.selectbox("Menu", menu)

    # Logout
    if choice == "Logout":
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.role = ""
        st.session_state.student_id = None
        safe_rerun()

    # ----------------
    # Home (Admin/User)
    # ----------------
    if choice == "Home":
        if role == "admin":
            st.subheader("🏫 Welcome Admin")
            total_students = cursor.execute("SELECT COUNT(*) FROM students").fetchone()[0]
            total_subjects = cursor.execute("SELECT COUNT(*) FROM subjects").fetchone()[0]
            total_att = cursor.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Students", total_students)
            c2.metric("Subjects", total_subjects)
            c3.metric("Attendance Records", total_att)

            students_df = pd.read_sql("SELECT * FROM students", conn)
            low = 0
            for _, s in students_df.iterrows():
                if attendance_percentage_for_student(int(s["id"])) < 75.0:
                    low += 1
            c4.metric("Low Attendance (<75%)", low)

            st.divider()
            st.subheader("📅 Upcoming Timetable")
            today = str(date.today())
            tt = pd.read_sql(
                """
                SELECT t.id, t.date, t.start_time, t.end_time, s.name AS subject
                FROM timetable t
                JOIN subjects s ON t.subject_id = s.id
                WHERE t.date >= ?
                ORDER BY t.date ASC, t.start_time ASC
                """,
                conn, params=(today,)
            )
            if tt.empty:
                st.info("No upcoming classes.")
            else:
                st.dataframe(tt, use_container_width=True)
        else:
            st.subheader(f"👋 Welcome {st.session_state.username}")
            if student_id is not None:
                pct = attendance_percentage_for_student(student_id)
                pred = predicted_marks_for_student(student_id)
                c1, c2 = st.columns(2)
                c1.metric("Attendance %", f"{pct:.2f}%")
                c2.metric("Predicted Marks", f"{pred:.2f}")

    # ---------------
    # Admin: Students
    # ---------------
    if role == "admin" and choice == "Students":
        st.subheader("Add Student")
        students_df = pd.read_sql("SELECT * FROM students", conn)
        st.dataframe(students_df, use_container_width=True)

        with st.form("add_student_form", clear_on_submit=True):
            s_name = st.text_input("Name")
            s_dept = st.text_input("Department")
            s_sem = st.text_input("Semester")
            s_age = st.number_input("Age", min_value=16, max_value=100, step=1)
            add_submit = st.form_submit_button("Add Student")
        if add_submit:
            if s_name.strip():
                cursor.execute(
                    "INSERT INTO students (name, department, semester, age) VALUES (?,?,?,?)",
                    (s_name, s_dept, s_sem, s_age)
                )
                conn.commit()
                sid = cursor.lastrowid
                uname = generate_unique_username(s_name)
                cursor.execute(
                    "INSERT INTO users (username, password, role, student_id, first_login) VALUES (?,?, 'user', ?, 1)",
                    (uname, uname, sid)
                )
                conn.commit()
                st.success(f"Student added! Username: {uname}  Password: {uname}")
                safe_rerun()
            else:
                st.error("Student name is required.")

        st.markdown("### Manage Students")
        students_df = pd.read_sql("SELECT * FROM students", conn)
        if students_df.empty:
            st.info("No students available.")
        else:
            sel_name = st.selectbox("Select Student to Remove", students_df["name"])
            if st.button("Remove Selected Student"):
                sid = int(students_df[students_df["name"] == sel_name]["id"].iloc[0])
                cursor.execute("DELETE FROM users WHERE student_id=?", (sid,))
                cursor.execute("DELETE FROM attendance WHERE student_id=?", (sid,))
                cursor.execute("DELETE FROM marks WHERE student_id=?", (sid,))
                cursor.execute("DELETE FROM students WHERE id=?", (sid,))
                conn.commit()
                st.success(f"Removed {sel_name}")
                safe_rerun()

    # ----------------
    # Admin: Subjects
    # ----------------
    if role == "admin" and choice == "Subjects":
        st.subheader("Manage Subjects")
        subjects_df = pd.read_sql("SELECT * FROM subjects", conn)
        st.dataframe(subjects_df, use_container_width=True)

        with st.form("add_subject_form", clear_on_submit=True):
            sub_name = st.text_input("Subject Name")
            sub_submit = st.form_submit_button("Add Subject")
        if sub_submit:
            if sub_name.strip():
                try:
                    cursor.execute("INSERT INTO subjects (name) VALUES (?)", (sub_name,))
                    conn.commit()
                    st.success("Subject added.")
                    safe_rerun()
                except sqlite3.IntegrityError:
                    st.error("Subject already exists.")
            else:
                st.error("Subject name is required.")

        st.markdown("### Remove Subject")
        if subjects_df.empty:
            st.info("No subjects available.")
        else:
            s_remove = st.selectbox("Select Subject", subjects_df["name"])
            if st.button("Remove Subject"):
                sub_id = int(subjects_df[subjects_df["name"] == s_remove]["id"].iloc[0])
                cursor.execute("DELETE FROM timetable WHERE subject_id=?", (sub_id,))
                cursor.execute("DELETE FROM marks WHERE subject_id=?", (sub_id,))
                cursor.execute("DELETE FROM attendance WHERE subject_id=?", (sub_id,))
                cursor.execute("DELETE FROM subjects WHERE id=?", (sub_id,))
                conn.commit()
                st.success(f"Removed subject: {s_remove}")
                safe_rerun()

    # -----------------
    # Admin: Timetable
    # -----------------
    if role == "admin" and choice == "Timetable":
        st.subheader("Manage Timetable")
        sub_map = subject_map()
        sub_id_to_name = subject_id_to_name_map()
        timetable_df = pd.read_sql("SELECT * FROM timetable ORDER BY date ASC, start_time ASC", conn)

        st.markdown("#### Add Entry")
        if not sub_map:
            st.warning("Add subjects first.")
        else:
            with st.form("add_tt_form", clear_on_submit=True):
                tt_date = st.date_input("Date", value=date.today())
                tt_subject = st.selectbox("Subject", list(sub_map.keys()))
                tt_start = st.time_input("Start Time")
                tt_end = st.time_input("End Time")
                add_tt = st.form_submit_button("Add to Timetable")
            if add_tt:
                cursor.execute(
                    "INSERT INTO timetable (date, subject_id, start_time, end_time) VALUES (?,?,?,?)",
                    (str(tt_date), sub_map[tt_subject], str(tt_start), str(tt_end))
                )
                conn.commit()
                st.success("Timetable entry added.")
                safe_rerun()

        st.markdown("#### Update / Delete Entry")
        if timetable_df.empty:
            st.info("No timetable entries.")
        else:
            def row_label(row):
                sname = sub_id_to_name.get(int(row["subject_id"]), f"Subject-{row['subject_id']}")
                return f"#{int(row['id'])} | {row['date']} | {sname} | {row['start_time']}-{row['end_time']}"
            timetable_df["label"] = timetable_df.apply(row_label, axis=1)
            sel_label = st.selectbox("Select Timetable Entry", timetable_df["label"])
            row = timetable_df[timetable_df["label"] == sel_label].iloc[0]

            with st.form("update_tt_form"):
                new_date = st.date_input("Date", value=parse_date_str(row["date"]))
                sub_names = list(sub_map.keys())
                current_sub_id = int(row["subject_id"])
                try:
                    current_sub_name = sub_id_to_name.get(current_sub_id, sub_names[0] if sub_names else "")
                    current_index = sub_names.index(current_sub_name) if current_sub_name in sub_names else 0
                except Exception:
                    current_index = 0 if sub_names else 0
                new_subject = st.selectbox("Subject", sub_names, index=current_index)
                new_start = st.time_input("Start Time", value=parse_time_str(row["start_time"]))
                new_end = st.time_input("End Time", value=parse_time_str(row["end_time"]))
                cU, cD = st.columns(2)
                with cU:
                    update_btn = st.form_submit_button("Update Entry")
                with cD:
                    delete_btn = st.form_submit_button("Delete Entry")

            if update_btn:
                cursor.execute(
                    "UPDATE timetable SET date=?, subject_id=?, start_time=?, end_time=? WHERE id=?",
                    (str(new_date), sub_map[new_subject], str(new_start), str(new_end), int(row["id"]))
                )
                conn.commit()
                st.success("Timetable updated.")
                safe_rerun()

            if delete_btn:
                cursor.execute("DELETE FROM timetable WHERE id=?", (int(row["id"]),))
                conn.commit()
                st.success("Timetable entry deleted.")
                safe_rerun()

    # -------------------
    # Admin: Attendance
    # -------------------
    if role == "admin" and choice == "Attendance":
        st.subheader("Mark Attendance (by Timetable)")
        timetable_df = pd.read_sql(
            """
            SELECT t.id, t.date, t.start_time, t.end_time, t.subject_id, s.name AS subject
            FROM timetable t
            JOIN subjects s ON t.subject_id = s.id
            ORDER BY t.date DESC, t.start_time DESC
            """, conn
        )
        if timetable_df.empty:
            st.info("No timetable entries. Add timetable first.")
        else:
            label = timetable_df.apply(
                lambda r: f"#{r['id']} | {r['date']} | {r['subject']} | {r['start_time']}-{r['end_time']}",
                axis=1
            ).tolist()
            sel = st.selectbox("Select Session", label)
            rid = int(sel.split("|")[0].strip("# ").strip())
            chosen = timetable_df[timetable_df["id"] == rid].iloc[0]

            st.write(f"Session: {chosen['date']} • {chosen['subject']} • {chosen['start_time']}-{chosen['end_time']}")

            students_df = pd.read_sql("SELECT * FROM students ORDER BY name ASC", conn)
            if students_df.empty:
                st.info("No students found.")
            else:
                with st.form("att_form"):
                    status_values = {}
                    for _, s in students_df.iterrows():
                        status_values[int(s["id"])] = st.selectbox(
                            f"{s['name']}",
                            ["Present", "Absent"],
                            key=f"att_{rid}_{int(s['id'])}"
                        )
                    submitted = st.form_submit_button("Save Attendance for All")
                if submitted:
                    count = 0
                    for sid, val in status_values.items():
                        cursor.execute(
                            "INSERT INTO attendance (student_id, date, subject_id, status) VALUES (?,?,?,?)",
                            (sid, chosen["date"], int(chosen["subject_id"]), val)
                        )
                        count += 1
                    conn.commit()
                    st.success(f"Attendance saved for {count} students.")

    # --------------
    # Admin: Marks
    # --------------
    if role == "admin" and choice == "Marks":
        st.subheader("Add / Update Marks")
        students_df = pd.read_sql("SELECT * FROM students ORDER BY name ASC", conn)
        subjects_df = pd.read_sql("SELECT * FROM subjects ORDER BY name ASC", conn)

        if students_df.empty or subjects_df.empty:
            st.info("Ensure students and subjects exist.")
        else:
            with st.form("marks_form", clear_on_submit=True):
                s_sel = st.selectbox("Student", students_df["name"])
                sub_sel = st.selectbox("Subject", subjects_df["name"])
                mark_val = st.number_input("Marks (0-100)", min_value=0, max_value=100, step=1)
                save_mark = st.form_submit_button("Save Marks")
            if save_mark:
                sid = int(students_df[students_df["name"] == s_sel]["id"].iloc[0])
                sub_id = int(subjects_df[subjects_df["name"] == sub_sel]["id"].iloc[0])

                cur = cursor.execute(
                    "SELECT id FROM marks WHERE student_id=? AND subject_id=?",
                    (sid, sub_id)
                ).fetchone()
                if cur:
                    cursor.execute("UPDATE marks SET marks=? WHERE id=?", (mark_val, int(cur[0])))
                else:
                    cursor.execute(
                        "INSERT INTO marks (student_id, subject_id, marks) VALUES (?,?,?)",
                        (sid, sub_id, mark_val)
                    )
                conn.commit()
                st.success("Marks saved.")

        st.markdown("### Current Marks")
        all_marks = pd.read_sql(
            """
            SELECT m.id, s.name AS student, subj.name AS subject, m.marks
            FROM marks m
            JOIN students s ON m.student_id = s.id
            JOIN subjects subj ON m.subject_id = subj.id
            ORDER BY student ASC, subject ASC
            """, conn
        )
        st.dataframe(all_marks, use_container_width=True)

    # -------------------
    # Admin: AI Dashboard
    # -------------------
    if role == "admin" and choice == "AI Dashboard":
        st.subheader("AI Notifications & Predictions")
        students_df = pd.read_sql("SELECT * FROM students", conn)
        if students_df.empty:
            st.info("No students.")
        else:
            records = []
            for _, s in students_df.iterrows():
                sid = int(s["id"])
                pct = attendance_percentage_for_student(sid)
                pred = predicted_marks_for_student(sid)
                records.append({
                    "Student": s["name"],
                    "Attendance %": round(pct, 2),
                    "Predicted Marks": round(pred, 2)
                })
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True)

            low_att = df[df["Attendance %"] < 75]
            if not low_att.empty:
                st.warning("Students with low attendance (<75%):")
                st.dataframe(low_att, use_container_width=True)

            at_risk = df[df["Predicted Marks"] < 50]
            if not at_risk.empty:
                st.error("Students predicted to score < 50:")
                st.dataframe(at_risk, use_container_width=True)

    # --------------------
    # Admin: Manage Users
    # --------------------
    if role == "admin" and choice == "Manage Users":
        st.subheader("Manage User Accounts (Usernames & Passwords)")
        users_df = pd.read_sql("""
            SELECT u.id, u.username, u.role, u.student_id, s.name AS student, u.first_login
            FROM users u
            LEFT JOIN students s ON u.student_id = s.id
            ORDER BY u.role DESC, u.username ASC
        """, conn)
        st.dataframe(users_df, use_container_width=True)

        st.markdown("#### Change Username (students)")
        non_admin_users = users_df[users_df["role"] == "user"]
        if non_admin_users.empty:
            st.info("No student users to manage.")
        else:
            sel_user = st.selectbox("Select User", non_admin_users["username"])
            new_username = st.text_input("New Username")
            if st.button("Update Username"):
                if new_username.strip():
                    try:
                        cursor.execute(
                            "UPDATE users SET username=? WHERE username=?",
                            (new_username, sel_user)
                        )
                        conn.commit()
                        st.success(f"Username changed from {sel_user} → {new_username}")
                        safe_rerun()
                    except sqlite3.IntegrityError:
                        st.error("That username is already taken.")
                else:
                    st.error("New username cannot be empty.")

        st.markdown("#### Reset Password")
        all_users = users_df["username"].tolist()
        if all_users:
            sel_user_pw = st.selectbox("Select Account", all_users, key="pw_user_sel")
            new_pw = st.text_input("New Password", type="password")
            if st.button("Set Password"):
                cursor.execute(
                    "UPDATE users SET password=?, first_login=0 WHERE username=?",
                    (new_pw, sel_user_pw)
                )
                conn.commit()
                st.success(f"Password updated for {sel_user_pw}")

        st.markdown("#### Re-link User to Student")
        if not non_admin_users.empty:
            link_user = st.selectbox("User to Link", non_admin_users["username"], key="link_user_sel")
            all_students = pd.read_sql("SELECT id, name FROM students ORDER BY name ASC", conn)
            if all_students.empty:
                st.info("No students available to link.")
            else:
                student_pick = st.selectbox("Student", all_students["name"], key="link_student_sel")
                if st.button("Link User → Student"):
                    sid = int(all_students[all_students["name"] == student_pick]["id"].iloc[0])
                    cursor.execute("UPDATE users SET student_id=? WHERE username=?", (sid, link_user))
                    conn.commit()
                    st.success(f"Linked {link_user} to {student_pick}")
                    safe_rerun()

        st.markdown("#### Delete User (non-admin)")
        del_candidates = users_df[users_df["role"] == "user"]["username"].tolist()
        if del_candidates:
            del_user = st.selectbox("Select User to Delete", del_candidates, key="del_user_sel")
            if st.button("Delete User"):
                cursor.execute("DELETE FROM users WHERE username=?", (del_user,))
                conn.commit()
                st.success(f"Deleted user {del_user}")
                safe_rerun()

    # -------------
    # User: Profile
    # -------------
    if role == "user" and choice == "My Details":
        st.subheader("👤 My Profile")
        if student_id is None:
            st.info("Your account isn't linked to a student profile yet.")
        else:
            student = pd.read_sql("SELECT * FROM students WHERE id=?", conn, params=(student_id,))
            if student.empty:
                st.info("Student profile not found.")
            else:
                s = student.iloc[0]
                c1, c2 = st.columns(2)
                c1.metric("Name", s.get("name", "-"))
                c2.metric("Age", int(s["age"]) if pd.notna(s.get("age")) else 0)
                c3, c4 = st.columns(2)
                c3.metric("Department", s.get("department", "-") if pd.notna(s.get("department")) else "-")
                c4.metric("Semester", s.get("semester", "-") if pd.notna(s.get("semester")) else "-")

                pct = attendance_percentage_for_student(student_id)
                pred = predicted_marks_for_student(student_id)
                st.metric("Attendance %", f"{pct:.2f}%")
                st.metric("Predicted Marks", f"{pred:.2f}")
                st.success("Welcome to your personalized dashboard! 🎓")

    # --------------------
    # User: My Attendance
    # --------------------
    if role == "user" and choice == "My Attendance":
        st.subheader("My Attendance")
        if student_id is None:
            st.info("No linked student.")
        else:
            att_df = pd.read_sql(
                """
                SELECT a.date, a.status, a.subject_id, s.name AS subject
                FROM attendance a
                LEFT JOIN subjects s ON a.subject_id = s.id
                WHERE a.student_id=?
                ORDER BY a.date DESC
                """,
                conn, params=(student_id,)
            )
            if att_df.empty:
                st.info("No attendance records yet.")
            else:
                st.dataframe(att_df, use_container_width=True)
                total = len(att_df)
                present = len(att_df[att_df["status"] == "Present"])
                pct = (present / total) * 100.0 if total else 0.0
                st.metric("Overall Attendance %", f"{pct:.2f}%")

                st.markdown("#### Subject-wise Attendance %")
                if "subject" in att_df.columns:
                    sub_stats = []
                    for sub, g in att_df.groupby("subject"):
                        t = len(g)
                        p = len(g[g["status"] == "Present"])
                        sub_stats.append({"Subject": sub, "Attendance %": round((p / t) * 100.0 if t else 0.0, 2)})
                    st.dataframe(pd.DataFrame(sub_stats), use_container_width=True)

    # -----------------
    # User: Timetable (by date, numbered from 1)
    # -----------------
    if role == "user" and choice == "Timetable":
        st.subheader("My Timetable")
        sel_date = st.date_input("Choose date", value=date.today())
        tt = pd.read_sql(
            """
            SELECT t.date, s.name AS subject, t.start_time, t.end_time
            FROM timetable t
            JOIN subjects s ON t.subject_id = s.id
            WHERE t.date = ?
            ORDER BY t.start_time ASC
            """, conn, params=(str(sel_date),)
        )
        if tt.empty:
            st.info("No classes on this date.")
        else:
            tt.insert(0, "No", range(1, len(tt) + 1))
            st.dataframe(tt.set_index("No"), use_container_width=True)

    # ---------------------------
    # User: Predict Performance
    # ---------------------------
    if role == "user" and choice == "Predict Performance":
        st.subheader("Predicted Performance")
        if student_id is None:
            st.info("No linked student.")
        else:
            pct = attendance_percentage_for_student(student_id)
            pred = predicted_marks_for_student(student_id)
            st.metric("Attendance %", f"{pct:.2f}%")
            st.metric("Predicted Marks", f"{pred:.2f}")

    # -------------
    # User: Chatbot (Upgraded: ML intents + quick actions + clear)
    # -------------
    if role == "user" and choice == "Chatbot":
        st.subheader("Student Chatbot 🤖")
        st.caption("Try: “my attendance”, “marks in DBMS”, “timetable tomorrow”, “classes on 05-10-2025”, “change my password to myNewPass”.")

        # Chat state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "pending_intent" not in st.session_state:
            st.session_state.pending_intent = None
        if "pending_slots" not in st.session_state:
            st.session_state.pending_slots = {}

        # Show chat bubbles
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Top controls
        c1, c2, c3, c4, c5 = st.columns(5)
        user_msg = None
        if c1.button("📊 My attendance"): user_msg = "my attendance"
        if c2.button("🧪 My marks"): user_msg = user_msg or "my marks"
        if c3.button("📅 Today’s timetable"): user_msg = user_msg or "timetable today"
        if c4.button("🔐 Change password"): user_msg = user_msg or "change my password"
        if c5.button("🧹 Clear chat"):
            st.session_state.chat_history = []
            st.stop()

        # If no quick-action pressed, take typed input
        if user_msg is None:
            user_msg = st.chat_input("Type your question...")

        if user_msg:
            # Log the user message
            st.session_state.chat_history.append({"role": "user", "content": user_msg})

            # Load subjects for NLU
            subs_df = pd.read_sql("SELECT name FROM subjects ORDER BY name ASC", conn)
            subject_names = subs_df["name"].tolist() if not subs_df.empty else []

            # Intent selection (ML first, then rule fallback)
            if st.session_state.pending_intent:
                intent = st.session_state.pending_intent
            else:
                intent = detect_intent_ml(user_msg)

            # Handle password change first
            if intent == "change_password":
                new_pw = extract_password(user_msg)
                if not new_pw:
                    bot = "Sure — what do you want to set as your new password? (e.g., “change my password to myNewPass”)"
                else:
                    if len(new_pw) < 4:
                        bot = "Password must be at least 4 characters. Try again."
                    else:
                        cursor.execute("UPDATE users SET password=?, first_login=0 WHERE username=?", (new_pw, st.session_state.username))
                        conn.commit()
                        bot = "Done! Your password has been updated ✅"
                st.session_state.chat_history.append({"role": "assistant", "content": bot})
                st.stop()

            # Slot needs
            need_subject = intent in ("attendance_subject", "marks_subject")
            need_date = intent in ("timetable_date",)

            # SUBJECT slot
            subject_slot = st.session_state.pending_slots.get("subject")
            if need_subject and not subject_slot:
                guess = fuzzy_find_subject(user_msg, subject_names)
                if guess:
                    subject_slot = guess
                else:
                    st.session_state.pending_intent = intent
                    st.session_state.pending_slots["subject"] = None
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "Which subject are you asking about?"
                    })
                    st.stop()

            # DATE slot
            date_slot = st.session_state.pending_slots.get("date")
            if need_date and not date_slot:
                d = parse_natural_date(user_msg)
                if d:
                    date_slot = d
                else:
                    st.session_state.pending_intent = intent
                    st.session_state.pending_slots["date"] = None
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "For which date? (e.g., today, tomorrow, 05-10-2025)"
                    })
                    st.stop()

            # Clear pending
            st.session_state.pending_intent = None
            st.session_state.pending_slots = {}

            # Execute intent
            bot = "Hmm, I didn’t get that. Ask about your **attendance**, **marks**, **timetable**, **profile**, or say **change my password to ...**"

            if intent == "help":
                bot = (
                    "I can help with:\n"
                    "• Attendance (overall or by subject)\n"
                    "• Marks (overall or by subject)\n"
                    "• Timetable (today, a specific date, or upcoming)\n"
                    "• Your profile\n"
                    "• Change password (e.g., “change my password to MyNewPass”)\n"
                    "Tip: you can click the quick buttons above."
                )

            elif intent == "profile_info":
                stu = pd.read_sql("SELECT * FROM students WHERE id=?", conn, params=(student_id,))
                if stu.empty:
                    bot = "Your profile isn’t linked yet. Ask admin to link your account."
                else:
                    s = stu.iloc[0]
                    pct = attendance_percentage_for_student(student_id)
                    pred = predicted_marks_for_student(student_id)
                    bot = (
                        f"**Profile**\n"
                        f"- Name: {s.get('name','-')}\n"
                        f"- Dept: {s.get('department','-')} • Sem: {s.get('semester','-')}\n"
                        f"- Attendance: {pct:.2f}% • Predicted Marks: {pred:.2f}"
                    )

            elif intent == "attendance_overall":
                pct = attendance_percentage_for_student(student_id)
                bot = f"Your overall attendance is **{pct:.2f}%**."

            elif intent == "attendance_subject":
                sub_name = subject_slot or fuzzy_find_subject(user_msg, subject_names)
                if not sub_name:
                    bot = "I couldn’t find that subject. Please try again."
                else:
                    df = pd.read_sql("""
                        SELECT status FROM attendance a
                        JOIN subjects s ON s.id=a.subject_id
                        WHERE a.student_id=? AND s.name=?
                    """, conn, params=(student_id, sub_name))
                    if df.empty:
                        bot = f"No attendance records found for **{sub_name}**."
                    else:
                        total = len(df)
                        present = len(df[df["status"] == "Present"])
                        pc = (present/total*100.0) if total else 0.0
                        bot = f"Your attendance in **{sub_name}** is **{pc:.2f}%**."

            elif intent == "marks_overall":
                rows = cursor.execute("""
                    SELECT m.marks, s.name
                    FROM marks m
                    JOIN subjects s ON s.id = m.subject_id
                    WHERE m.student_id=?
                """, (student_id,)).fetchall()
                if not rows:
                    bot = "No marks available yet."
                else:
                    avg = float(np.mean([r[0] for r in rows]))
                    parts = [f"- {name}: {marks}" for marks, name in rows]
                    bot = f"**Average:** {avg:.2f}\n" + "\n".join(parts)

            elif intent == "marks_subject":
                sub_name = subject_slot or fuzzy_find_subject(user_msg, subject_names)
                if not sub_name:
                    bot = "I couldn’t find that subject. Please try again."
                else:
                    row = cursor.execute("""
                        SELECT m.marks
                        FROM marks m
                        JOIN subjects s ON s.id = m.subject_id
                        WHERE m.student_id=? AND s.name=?
                    """, (student_id, sub_name)).fetchone()
                    bot = f"Your marks in **{sub_name}**: **{row[0]}**." if row else f"No marks yet for **{sub_name}**."

            elif intent in ("timetable_today", "timetable_date", "timetable_next"):
                if intent == "timetable_today":
                    d = date.today()
                    q = """
                        SELECT t.date, s.name, t.start_time, t.end_time
                        FROM timetable t JOIN subjects s ON s.id=t.subject_id
                        WHERE t.date=? ORDER BY t.start_time ASC
                    """
                    tt = pd.read_sql(q, conn, params=(str(d),))
                    if tt.empty:
                        bot = "No classes today."
                    else:
                        lines = [f"{i}. {r['date']} • {r['name']} • {r['start_time']}-{r['end_time']}"
                                 for i, r in enumerate(tt.to_dict('records'), start=1)]
                        bot = "Today’s classes:\n" + "\n".join(lines)

                elif intent == "timetable_date":
                    d = date_slot or parse_natural_date(user_msg) or date.today()
                    q = """
                        SELECT t.date, s.name, t.start_time, t.end_time
                        FROM timetable t JOIN subjects s ON s.id=t.subject_id
                        WHERE t.date=? ORDER BY t.start_time ASC
                    """
                    tt = pd.read_sql(q, conn, params=(str(d),))
                    if tt.empty:
                        bot = f"No classes on {d}."
                    else:
                        lines = [f"{i}. {r['date']} • {r['name']} • {r['start_time']}-{r['end_time']}"
                                 for i, r in enumerate(tt.to_dict('records'), start=1)]
                        bot = f"Classes on {d}:\n" + "\n".join(lines)

                else:  # timetable_next
                    today_str = str(date.today())
                    q = """
                        SELECT t.date, s.name, t.start_time, t.end_time
                        FROM timetable t JOIN subjects s ON s.id=t.subject_id
                        WHERE t.date>=? ORDER BY t.date ASC, t.start_time ASC
                    """
                    tt = pd.read_sql(q, conn, params=(today_str,))
                    if tt.empty:
                        bot = "No upcoming classes."
                    else:
                        tt = tt.head(5)
                        lines = [f"{i}. {r['date']} • {r['name']} • {r['start_time']}-{r['end_time']}"
                                 for i, r in enumerate(tt.to_dict('records'), start=1)]
                        bot = "Upcoming classes:\n" + "\n".join(lines)

            # Smart reply for generic performance questions
            lm = (user_msg or "").lower()
            if ("performance" in lm) or ("predict" in lm) or ("how am i doing" in lm):
                pct = attendance_percentage_for_student(student_id)
                pred = predicted_marks_for_student(student_id)
                bot += f"\n\n📈 Based on your attendance **{pct:.2f}%**, your predicted marks are **{pred:.2f}**."

            st.session_state.chat_history.append({"role": "assistant", "content": bot})