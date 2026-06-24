from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "safeguard.db"


def _connect() -> sqlite3.Connection:
    # Open a WAL-mode connection so reads and writes don't block each other
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    # Create all tables on first run, then apply any missing column migrations
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                anon_id       TEXT PRIMARY KEY,
                display_name  TEXT NOT NULL DEFAULT '',
                password_hash TEXT NOT NULL DEFAULT '',
                consent_given INTEGER NOT NULL DEFAULT 0,
                created_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS posts (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                anon_user_id  TEXT NOT NULL,
                display_name  TEXT NOT NULL DEFAULT '',
                text          TEXT NOT NULL,
                tags          TEXT NOT NULL DEFAULT '[]',
                emotion       TEXT NOT NULL,
                confidence    REAL NOT NULL,
                week          TEXT NOT NULL,
                submitted_at  TEXT NOT NULL,
                FOREIGN KEY (anon_user_id) REFERENCES users(anon_id)
            );

            CREATE TABLE IF NOT EXISTS emotion_feedback (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id       INTEGER,
                anon_user_id  TEXT NOT NULL,
                predicted     TEXT NOT NULL,
                is_correct    INTEGER NOT NULL,
                created_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS weekly_reports (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                anon_user_id       TEXT NOT NULL,
                week               TEXT NOT NULL,
                quiz_score         INTEGER NOT NULL,
                emotion_score      REAL NOT NULL,
                combined_score     REAL NOT NULL,
                risk_tier          TEXT NOT NULL,
                needs_human_review INTEGER NOT NULL DEFAULT 0,
                report_json        TEXT NOT NULL,
                reviewer_action    TEXT,
                reviewed_at        TEXT,
                created_at         TEXT NOT NULL,
                FOREIGN KEY (anon_user_id) REFERENCES users(anon_id)
            );

            CREATE TABLE IF NOT EXISTS post_comments (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id       INTEGER NOT NULL,
                anon_user_id  TEXT NOT NULL,
                display_name  TEXT NOT NULL DEFAULT '',
                text          TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                FOREIGN KEY (post_id) REFERENCES posts(id)
            );
        """)
    _migrate()


def _migrate() -> None:
    # Safely add columns that were introduced after the initial schema — skips if already there
    new_cols = [
        "ALTER TABLE users ADD COLUMN display_name  TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE users ADD COLUMN password_hash TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE posts ADD COLUMN display_name  TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE posts ADD COLUMN tags          TEXT NOT NULL DEFAULT '[]'",
    ]
    for sql in new_cols:
        try:
            with _connect() as conn:
                conn.execute(sql)
        except sqlite3.OperationalError:
            pass


def anonymize(raw_identifier: str) -> str:
    # Hash the username with SHA-256 so the real name never touches the database
    return hashlib.sha256(raw_identifier.encode("utf-8")).hexdigest()


def hash_password(password: str) -> str:
    # One-way SHA-256 hash of the password for storage
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_user(anon_id: str) -> dict[str, Any] | None:
    # Look up a user by their anonymous ID, returns None if they don't exist yet
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE anon_id = ?", (anon_id,)).fetchone()
        return dict(row) if row else None


def get_or_create_user(anon_id: str, display_name: str = "", password_hash: str = "") -> dict[str, Any]:
    # Return the existing user row, or create a new account with consent pending
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE anon_id = ?", (anon_id,)).fetchone()
        if row is None:
            now = _now()
            conn.execute(
                "INSERT INTO users (anon_id, display_name, password_hash, consent_given, created_at) "
                "VALUES (?, ?, ?, 0, ?)",
                (anon_id, display_name, password_hash, now),
            )
            return {"anon_id": anon_id, "display_name": display_name,
                    "password_hash": password_hash, "consent_given": 0, "created_at": now}
        return dict(row)


def set_consent(anon_id: str, given: bool) -> None:
    # Record whether the user accepted or declined the consent agreement
    with _connect() as conn:
        conn.execute("UPDATE users SET consent_given = ? WHERE anon_id = ?",
                     (1 if given else 0, anon_id))


def save_post(anon_user_id: str, display_name: str, text: str, tags: list[str],
              emotion: str, confidence: float, week_id: str) -> int:
    # Save a new post with its AI-detected emotion and return the new row ID
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO posts (anon_user_id, display_name, text, tags, emotion, confidence, week, submitted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (anon_user_id, display_name, text, json.dumps(tags), emotion, confidence, week_id, _now()),
        )
        return cur.lastrowid


def get_feed_posts(current_anon_user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    # Fetch the latest posts from all users, marking which ones belong to the current user
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, anon_user_id, display_name, text, tags, emotion, confidence, submitted_at "
            "FROM posts ORDER BY submitted_at DESC LIMIT ?", (limit,),
        ).fetchall()
    result = []
    for r in rows:
        try:
            tags = json.loads(r["tags"] or "[]")
        except (json.JSONDecodeError, TypeError):
            tags = []
        result.append({
            "id":           r["id"],
            "display_name": r["display_name"] or "User",
            "text":         r["text"],
            "tags":         tags,
            "emotion":      r["emotion"],
            "confidence":   r["confidence"],
            "submitted_at": r["submitted_at"],
            "is_own":       r["anon_user_id"] == current_anon_user_id,
        })
    return result


def get_posts_this_week(anon_user_id: str, week_id: str) -> list[dict[str, Any]]:
    # Return just the emotion and confidence for each post this week — used by the scoring module
    with _connect() as conn:
        rows = conn.execute(
            "SELECT emotion, confidence FROM posts WHERE anon_user_id = ? AND week = ?",
            (anon_user_id, week_id),
        ).fetchall()
    return [{"emotion": r["emotion"], "score": r["confidence"]} for r in rows]


def save_comment(post_id: int, anon_user_id: str, display_name: str, text: str) -> int:
    # Save a comment on a post and return the new row ID
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO post_comments (post_id, anon_user_id, display_name, text, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (post_id, anon_user_id, display_name, text, _now()),
        )
        return cur.lastrowid


def get_comments(post_id: int) -> list[dict[str, Any]]:
    # Return all comments for a post, oldest first
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, display_name, text, created_at FROM post_comments "
            "WHERE post_id = ? ORDER BY created_at ASC", (post_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_comment_counts(post_ids: list[int]) -> dict[int, int]:
    # Return a count of comments for each post ID in one query
    if not post_ids:
        return {}
    placeholders = ",".join("?" * len(post_ids))
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT post_id, COUNT(*) cnt FROM post_comments "
            f"WHERE post_id IN ({placeholders}) GROUP BY post_id", post_ids,
        ).fetchall()
    return {r["post_id"]: r["cnt"] for r in rows}


def get_all_posts(anon_user_id: str) -> list[dict[str, Any]]:
    # Return the full post history for a user, newest first
    with _connect() as conn:
        rows = conn.execute(
            "SELECT text, emotion, confidence, week, submitted_at "
            "FROM posts WHERE anon_user_id = ? ORDER BY submitted_at DESC", (anon_user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def save_report(anon_user_id: str, report: dict[str, Any]) -> int:
    # Persist the weekly risk report produced by scoring.build_weekly_report()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO weekly_reports "
            "(anon_user_id, week, quiz_score, emotion_score, combined_score, "
            "risk_tier, needs_human_review, report_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (anon_user_id, report["week"], report["quiz_score"], report["emotion_score"],
             report["combined_score"], report["risk_tier"],
             1 if report["needs_human_review"] else 0, json.dumps(report), _now()),
        )
        return cur.lastrowid


def get_reports(anon_user_id: str) -> list[dict[str, Any]]:
    # Return all weekly reports for a user, newest first
    with _connect() as conn:
        rows = conn.execute(
            "SELECT report_json, reviewer_action, reviewed_at FROM weekly_reports "
            "WHERE anon_user_id = ? ORDER BY created_at DESC", (anon_user_id,),
        ).fetchall()
    results = []
    for r in rows:
        report = json.loads(r["report_json"])
        report["reviewer_action"] = r["reviewer_action"]
        report["reviewed_at"] = r["reviewed_at"]
        results.append(report)
    return results


def get_latest_report(anon_user_id: str) -> dict[str, Any] | None:
    # Return just the most recent report, or None if the user hasn't done a check-in yet
    reports = get_reports(anon_user_id)
    return reports[0] if reports else None


def get_review_queue() -> list[dict[str, Any]]:
    # Return all reports flagged for human review, unactioned ones first
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, anon_user_id, week, combined_score, risk_tier, "
            "report_json, reviewer_action, reviewed_at "
            "FROM weekly_reports WHERE needs_human_review = 1 "
            "ORDER BY (reviewer_action IS NULL) DESC, created_at DESC",
        ).fetchall()
    results = []
    for r in rows:
        item = dict(r)
        item["report"] = json.loads(r["report_json"])
        results.append(item)
    return results


def mark_referral_offered(report_id: int) -> None:
    # Record that a human reviewer offered support for this flagged report
    with _connect() as conn:
        conn.execute(
            "UPDATE weekly_reports SET reviewer_action = 'referral_offered', reviewed_at = ? WHERE id = ?",
            (_now(), report_id),
        )


def save_feedback(post_id: int, anon_user_id: str, predicted: str, is_correct: bool) -> None:
    # Store a thumbs-up or thumbs-down on an AI prediction as a labelled training sample
    with _connect() as conn:
        conn.execute(
            "INSERT INTO emotion_feedback (post_id, anon_user_id, predicted, is_correct, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (post_id, anon_user_id, predicted, 1 if is_correct else 0, _now()),
        )


def get_feedback_stats() -> dict[str, Any]:
    # Aggregate all community votes into total, correct count, accuracy %, and per-emotion breakdown
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) total, SUM(is_correct) correct FROM emotion_feedback").fetchone()
        by_emotion = conn.execute(
            "SELECT predicted, COUNT(*) total, SUM(is_correct) correct "
            "FROM emotion_feedback GROUP BY predicted"
        ).fetchall()
    total   = row["total"] or 0
    correct = int(row["correct"] or 0)
    return {
        "total":      total,
        "correct":    correct,
        "accuracy":   round(correct / total * 100, 1) if total else None,
        "by_emotion": {r["predicted"]: {"total": r["total"], "correct": int(r["correct"] or 0)}
                       for r in by_emotion},
    }


def delete_user_data(anon_id: str) -> dict[str, int]:
    # Permanently erase everything linked to this user — posts, reports, and account (GDPR Art. 17)
    with _connect() as conn:
        posts   = conn.execute("DELETE FROM posts WHERE anon_user_id=?", (anon_id,)).rowcount
        reports = conn.execute("DELETE FROM weekly_reports WHERE anon_user_id=?", (anon_id,)).rowcount
        users   = conn.execute("DELETE FROM users WHERE anon_id=?", (anon_id,)).rowcount
    return {"posts": posts, "reports": reports, "users": users}


def export_user_data(anon_id: str) -> dict[str, Any]:
    # Package everything the app holds about this user into a JSON-ready dict (GDPR Art. 20)
    with _connect() as conn:
        user = conn.execute(
            "SELECT display_name, consent_given, created_at FROM users WHERE anon_id=?", (anon_id,)
        ).fetchone()
        posts = conn.execute(
            "SELECT text, tags, emotion, confidence, week, submitted_at "
            "FROM posts WHERE anon_user_id=? ORDER BY submitted_at", (anon_id,)
        ).fetchall()
        reports = conn.execute(
            "SELECT week, quiz_score, emotion_score, combined_score, risk_tier, created_at "
            "FROM weekly_reports WHERE anon_user_id=? ORDER BY created_at", (anon_id,)
        ).fetchall()
    return {
        "notice":  "This is all data WithUs holds about you. Your real identity is never stored.",
        "account": dict(user) if user else {},
        "posts":   [dict(r) for r in posts],
        "reports": [dict(r) for r in reports],
    }


def _now() -> str:
    # Return the current UTC time as an ISO string
    return datetime.now(timezone.utc).isoformat()
