"""Action utilities for autonomous agent actions and fake database updates."""

import sqlite3
from pathlib import Path
from typing import Dict

DB_PATH = Path(__file__).parents[1] / "database" / "support_logs.db"


def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer TEXT,
            issue TEXT,
            status TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()


def create_ticket(customer: str, issue: str) -> Dict[str, str]:
    ensure_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO tickets (customer, issue, status, created_at) VALUES (?, ?, 'open', datetime('now'))",
        (customer, issue),
    )
    ticket_id = cur.lastrowid
    conn.commit()
    conn.close()
    return {"ticket_id": ticket_id, "status": "open"}


def password_reset(username: str) -> Dict[str, str]:
    ensure_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT username FROM users WHERE username = ?", (username,))
    if cur.fetchone() is None:
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, "Temp#1234"))
    else:
        cur.execute("UPDATE users SET password = ? WHERE username = ?", ("Temp#1234", username))
    conn.commit()
    conn.close()
    return {"username": username, "password": "Temp#1234", "status": "reset"}
