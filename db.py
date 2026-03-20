import sqlite3
import os
import hashlib
from datetime import datetime

import bcrypt

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meals.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ── Password helpers ─────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _check_password(password: str, stored_hash: str) -> bool:
    if stored_hash.startswith("$2"):
        return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
    # Legacy SHA-256 fallback
    return hashlib.sha256(password.encode("utf-8")).hexdigest() == stored_hash


# ── Schema + migrations ──────────────────────────────────────────────────────

def init_db():
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS meal_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            uploaded_at TEXT,
            eaten_at TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'manual',
            image_hash TEXT,
            ai_raw_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS meal_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meal_session_id INTEGER NOT NULL,
            food_name TEXT NOT NULL,
            grams REAL NOT NULL,
            entered_unit TEXT,
            entered_quantity REAL,
            confidence REAL NOT NULL DEFAULT 0,
            nutrition_source TEXT NOT NULL DEFAULT 'fallback',
            matched_product_name TEXT,
            status TEXT NOT NULL DEFAULT 'manually_added',
            calories REAL NOT NULL,
            protein REAL NOT NULL,
            fat REAL NOT NULL,
            carbs REAL NOT NULL,
            FOREIGN KEY (meal_session_id) REFERENCES meal_sessions(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS food_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            original_name TEXT NOT NULL,
            corrected_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_food_overrides (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            food_name_normalized TEXT NOT NULL,
            calories_per_100g REAL NOT NULL,
            protein_per_100g REAL NOT NULL,
            fat_per_100g REAL NOT NULL,
            carbs_per_100g REAL NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, food_name_normalized)
        )
    """)

    _migrate_legacy_meals(conn)

    conn.commit()
    conn.close()


def _migrate_legacy_meals(conn):
    """Migrate old flat 'meals' table into meal_sessions + meal_items."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='meals'"
    )
    if not cursor.fetchone():
        return

    count = conn.execute("SELECT COUNT(*) FROM meals").fetchone()[0]
    if count == 0:
        return

    session_count = conn.execute("SELECT COUNT(*) FROM meal_sessions").fetchone()[0]
    if session_count > 0:
        return  # already migrated

    cols_cursor = conn.execute("PRAGMA table_info(meals)")
    columns = [row["name"] for row in cols_cursor.fetchall()]

    rows = conn.execute("SELECT * FROM meals ORDER BY id").fetchall()

    sessions: dict = {}
    for row in rows:
        r = dict(row)
        uid = r.get("user_id", 0)
        eaten = (
            r.get("eaten_at")
            or r.get("uploaded_at")
            or r.get("timestamp", "")
        )
        uploaded = r.get("uploaded_at") or r.get("timestamp", "")
        key = (uid, eaten)
        if key not in sessions:
            sessions[key] = {
                "user_id": uid,
                "eaten_at": eaten or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "uploaded_at": uploaded,
                "items": [],
            }
        sessions[key]["items"].append(r)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for _key, sess in sessions.items():
        cur = conn.execute(
            """INSERT INTO meal_sessions
                   (user_id, uploaded_at, eaten_at, source, created_at)
               VALUES (?, ?, ?, 'migrated', ?)""",
            (sess["user_id"], sess["uploaded_at"], sess["eaten_at"], now),
        )
        sid = cur.lastrowid
        for it in sess["items"]:
            conn.execute(
                """INSERT INTO meal_items
                       (meal_session_id, food_name, grams, confidence,
                        nutrition_source, status, calories, protein, fat, carbs)
                   VALUES (?, ?, ?, 0, 'unknown', 'migrated', ?, ?, ?, ?)""",
                (
                    sid,
                    it.get("food_name", "Unknown"),
                    it.get("grams", 0),
                    it.get("calories", 0),
                    it.get("protein", 0),
                    it.get("fat", 0),
                    it.get("carbs", 0),
                ),
            )


# ── User management ──────────────────────────────────────────────────────────

def register_user(username: str, password: str) -> bool:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, _hash_password(password)),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> int | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT id, password_hash FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    if not row:
        conn.close()
        return None
    if _check_password(password, row["password_hash"]):
        # Re-hash legacy SHA-256 passwords to bcrypt on successful login
        if not row["password_hash"].startswith("$2"):
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (_hash_password(password), row["id"]),
            )
            conn.commit()
        conn.close()
        return row["id"]
    conn.close()
    return None


# ── Meal sessions ────────────────────────────────────────────────────────────

def save_meal_session(
    user_id: int,
    uploaded_at: str | None,
    eaten_at: str,
    source: str,
    image_hash: str | None,
    ai_raw_json: str | None,
    items: list[dict],
) -> int:
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.execute(
        """INSERT INTO meal_sessions
               (user_id, uploaded_at, eaten_at, source, image_hash, ai_raw_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (user_id, uploaded_at, eaten_at, source, image_hash, ai_raw_json, now),
    )
    session_id = cur.lastrowid
    for item in items:
        conn.execute(
            """INSERT INTO meal_items
                   (meal_session_id, food_name, grams, entered_unit, entered_quantity,
                    confidence, nutrition_source, matched_product_name, status,
                    calories, protein, fat, carbs)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                item["name"],
                item["grams"],
                item.get("entered_unit"),
                item.get("entered_quantity"),
                item.get("confidence", 0),
                item.get("nutrition_source", "fallback"),
                item.get("matched_product_name"),
                item.get("status", "manually_added"),
                item["calories"],
                item["protein"],
                item["fat"],
                item["carbs"],
            ),
        )
    conn.commit()
    conn.close()
    return session_id


def get_meal_sessions(
    user_id: int,
    date_from: str | None = None,
    date_to: str | None = None,
    food_search: str | None = None,
) -> list[dict]:
    conn = get_connection()
    query = "SELECT * FROM meal_sessions WHERE user_id = ?"
    params: list = [user_id]
    if date_from:
        query += " AND eaten_at >= ?"
        params.append(date_from)
    if date_to:
        query += " AND eaten_at <= ?"
        params.append(date_to + " 23:59:59")
    query += " ORDER BY eaten_at DESC"

    sessions = [dict(r) for r in conn.execute(query, params).fetchall()]
    for sess in sessions:
        items = conn.execute(
            "SELECT * FROM meal_items WHERE meal_session_id = ?",
            (sess["id"],),
        ).fetchall()
        sess["items"] = [dict(i) for i in items]

    if food_search:
        q = food_search.lower()
        sessions = [
            s for s in sessions
            if any(q in it["food_name"].lower() for it in s["items"])
        ]
    conn.close()
    return sessions


def get_all_meal_items_flat(user_id: int) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        """SELECT ms.uploaded_at, ms.eaten_at, ms.source,
                  mi.food_name, mi.grams, mi.entered_unit, mi.entered_quantity,
                  mi.confidence, mi.nutrition_source, mi.matched_product_name,
                  mi.status, mi.calories, mi.protein, mi.fat, mi.carbs
           FROM meal_items mi
           JOIN meal_sessions ms ON mi.meal_session_id = ms.id
           WHERE ms.user_id = ?
           ORDER BY ms.eaten_at DESC""",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Food corrections ─────────────────────────────────────────────────────────

def save_food_correction(user_id: int, original_name: str, corrected_name: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO food_corrections (user_id, original_name, corrected_name, timestamp) VALUES (?, ?, ?, ?)",
        (user_id, original_name, corrected_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()


def get_food_corrections(user_id: int) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        """SELECT original_name, corrected_name, COUNT(*) as count
           FROM food_corrections WHERE user_id = ?
           GROUP BY original_name, corrected_name ORDER BY count DESC""",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Analytics helpers ────────────────────────────────────────────────────────

def get_daily_nutrition(user_id: int) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        """SELECT DATE(ms.eaten_at) as date,
                  SUM(mi.calories) as total_calories,
                  SUM(mi.protein) as total_protein,
                  SUM(mi.fat) as total_fat,
                  SUM(mi.carbs) as total_carbs,
                  COUNT(DISTINCT ms.id) as meal_count
           FROM meal_items mi
           JOIN meal_sessions ms ON mi.meal_session_id = ms.id
           WHERE ms.user_id = ?
           GROUP BY DATE(ms.eaten_at)
           ORDER BY date DESC
           LIMIT 30""",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_top_foods(user_id: int, limit: int = 10) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        """SELECT mi.food_name, COUNT(*) as times_eaten,
                  ROUND(AVG(mi.calories), 0) as avg_calories,
                  ROUND(AVG(mi.grams), 0) as avg_grams
           FROM meal_items mi
           JOIN meal_sessions ms ON mi.meal_session_id = ms.id
           WHERE ms.user_id = ?
           GROUP BY LOWER(mi.food_name)
           ORDER BY times_eaten DESC
           LIMIT ?""",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
