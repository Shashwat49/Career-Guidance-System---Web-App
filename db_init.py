import sqlite3
from pathlib import Path

DB_PATH = Path("career.db")

schema = """
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    english REAL NOT NULL,
    math REAL NOT NULL,
    science REAL NOT NULL,
    history REAL NOT NULL,
    geography REAL NOT NULL,
    interest TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    predicted_career TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(student_id) REFERENCES students(id)
);
"""

def main():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.executescript(schema)
        conn.commit()
    print("✅ SQLite DB initialized at career.db")

if __name__ == "__main__":
    main()
