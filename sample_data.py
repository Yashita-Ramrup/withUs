# =====================================================
# FILE: sample_data.py
# OWNER: Team (shared demo utility)
# PURPOSE: Seed the database with realistic fake data for day-one testing
# Game of Code 2026 - Towards Recovery
# =====================================================

"""
Run this ONCE before the demo to pre-populate the database.
It bypasses detection.py entirely, so the AI model does NOT need to be
downloaded before you can show the app end-to-end.

    python3 sample_data.py

Then launch the app and log in as 'demo_user'.
"""

from __future__ import annotations

from datetime import datetime

import scoring
import storage

# Matches app.py's WEEK_ID format so posts appear in the current week
WEEK_ID = datetime.now().strftime("%Y-W%W")

# Pre-labelled posts — emotions assigned by hand to bypass detection model
FAKE_POSTS: list[tuple[str, str, float]] = [
    ("Had a great time with friends this evening, feeling really good!", "joy", 0.91),
    ("Couldn't sleep again last night. Worried about everything.", "sadness", 0.87),
    ("The meeting went okay, I was a bit nervous going in.", "fear", 0.62),
    ("Really frustrated with how things are going at work.", "anger", 0.79),
    ("Just a normal day, nothing particular to report.", "neutral", 0.83),
    ("Feeling really low this week — hard to find motivation for anything.", "sadness", 0.93),
    ("Got some unexpected good news today, I was genuinely surprised.", "surprise", 0.76),
    ("Anxious about tomorrow, keep running the scenarios in my head.", "fear", 0.71),
    ("Went for a walk outside. Helped a little.", "joy", 0.65),
    ("Things have been quite tough lately. Feeling a bit hopeless.", "sadness", 0.88),
    ("Had a moment of real peace this morning — just sat with my coffee.", "joy", 0.70),
    ("Keep dreading things that haven't happened yet.", "fear", 0.77),
]

# Quiz answers: mix of 1s and 2s → produces a medium-tier combined score
FAKE_QUIZ_ANSWERS = [2, 1, 2, 1, 2, 1, 2, 1]

DEMO_USERNAME = "demo_user"


def seed() -> None:
    storage.init_db()

    anon_id = storage.anonymize(DEMO_USERNAME)
    storage.get_or_create_user(anon_id)
    storage.set_consent(anon_id, True)

    for text, emotion, confidence in FAKE_POSTS:
        storage.save_post(anon_id, text, emotion, confidence, WEEK_ID)

    weekly_posts = storage.get_posts_this_week(anon_id, WEEK_ID)
    report = scoring.build_weekly_report(WEEK_ID, FAKE_QUIZ_ANSWERS, weekly_posts)
    storage.save_report(anon_id, report)

    print(f"Demo user seeded: '{DEMO_USERNAME}'")
    print(f"  Anon ID : {anon_id[:16]}…")
    print(f"  Posts   : {len(FAKE_POSTS)}")
    print(f"  Report  : week={report['week']}, tier={report['risk_tier']}, "
          f"combined={report['combined_score']:.1f}, review={report['needs_human_review']}")
    print()
    print("Run 'python3 app.py' and log in as 'demo_user' to see the full demo.")


if __name__ == "__main__":
    seed()
