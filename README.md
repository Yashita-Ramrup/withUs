# WithUs (safeguard-mauritius) — "Towards Recovery"

**Game of Code 2026 submission.** A private, anonymised wellbeing companion that pairs a short weekly substance-use screening quiz with lightweight emotion analysis of a user's own journal/feed posts, to surface gentle, supportive feedback — and quietly flag medium/high concern cases for a human reviewer. No automated or punitive action is ever taken on a user.

---

## Table of contents

- [Concept](#concept)
- [Architecture](#architecture)
- [Project layout](#project-layout)
- [Setup](#setup)
- [Running the app](#running-the-app)
- [Walkthrough](#walkthrough)
- [Data model](#data-model)
- [Scoring methodology](#scoring-methodology)
- [Privacy & anonymisation](#privacy--anonymisation)
- [API reference](#api-reference)
- [Known issues / things to fix before a real demo](#known-issues--things-to-fix-before-a-real-demo)
- [Team ownership](#team-ownership)

---

## Concept

A user logs in with a username/password (no real identity is ever stored), posts short free-text entries into a shared social-style feed, and once a week answers an 8-question ASSIST-inspired substance-use screening quiz. The app combines:

- **Quiz score** (self-report, weighted 60%)
- **Emotion score** (inferred from the week's posts via a text-classification model, weighted 40%)

into a single **combined risk tier** (`low` / `medium` / `high`). The user only ever sees a warm, supportive message — never the raw tier or score. Reports flagged `medium` or `high` go into a private human-review queue, keyed only by the user's anonymous hash, for a trained reviewer to action (e.g. offer a referral). Nothing is ever auto-escalated to authorities or used punitively.

## Architecture

```
┌─────────────┐      POST /api/post        ┌──────────────┐
│   Browser   │ ───────────────────────────▶│   app.py     │
│  (index.html)│      POST /api/checkin     │  (Flask API) │
└─────────────┘ ◀───────────────────────────└──────┬───────┘
                                                     │
                       ┌─────────────────────────────┼─────────────────────────────┐
                       ▼                             ▼                             ▼
               detection.py                    scoring.py                   storage.py
       (j-hartmann emotion model)      (quiz + risk-tier logic)      (SQLite persistence,
                                                                       SHA-256 anonymisation)
                                                                              │
                                                                              ▼
                                                                       safeguard.db
```

- **`app.py`** — Flask routes, session-based auth, request/response shaping for the SPA frontend (templates not included in this bundle — see [Known issues](#known-issues--things-to-fix-before-a-real-demo)).
- **`detection.py`** — wraps a Hugging Face `transformers` pipeline (`j-hartmann/emotion-english-distilroberta-base`) to classify free text into one of 7 emotions with a confidence score. Loaded lazily, in a background thread, so the app boots instantly and falls back to a keyword heuristic until the model is ready.
- **`scoring.py`** — pure functions: scores the quiz, aggregates negative-emotion signal from a week of posts, combines both into a 0–24 risk score, and builds the user-facing weekly report.
- **`storage.py`** — SQLite (WAL mode) schema and queries for `users`, `posts`, and `weekly_reports`; also owns the one-way anonymisation (SHA-256) of usernames and password hashing.
- **`sample_data.py`** — one-off seeding script so the app can be demoed without waiting for the ML model to download.

## Project layout

```
.
├── app.py              # Flask app & routes (Person 3)
├── detection.py         # Emotion classification (Person 1)
├── scoring.py           # Quiz + risk-tier logic (Person 2)
├── storage.py           # SQLite persistence & anonymisation (Person 4)
├── sample_data.py       # Demo data seeder (shared)
├── requirements.txt
├── safeguard.db          # SQLite database (WAL mode — ships with demo data)
├── safeguard.db-shm / -wal
└── README.md
```

> Note: `app.py` renders `login.html`, `consent.html`, and `index.html` via `render_template`, but no `templates/` directory was included in this upload — see [Known issues](#known-issues--things-to-fix-before-a-real-demo).

## Setup

Requires Python 3.10+ (uses `from __future__ import annotations` and `list[int]`-style generics throughout).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pulls in Flask, `transformers`, `torch`, `accelerate`, and `matplotlib`. The emotion model (~260 MB) is **not** bundled — it downloads automatically the first time `detection._get_pipeline()` runs.

## Running the app

**Option A — instant demo, no model download:**

```bash
python3 sample_data.py     # seeds a 'demo_user' with realistic fake posts + a report
python3 app.py
```

Then open `http://localhost:5000` and log in as `demo_user` (the seeder doesn't set a password, so check `storage.get_or_create_user` defaults / set one via the login flow on first run).

**Option B — full pipeline with live emotion detection:**

```bash
python3 app.py
```

The model loads in a background thread; `model_ready` is passed to the template so the UI can indicate loading state. Until it's ready, `/api/post` falls back to a simple keyword-based heuristic (see `app.py`'s `api_post`).

## Walkthrough

1. **Login / sign-up** (`/login`) — same form for both; a new account is created on first sight of a username. Passwords are SHA-256 hashed (see [Known issues](#known-issues--things-to-fix-before-a-real-demo) re: salting).
2. **Consent** (`/consent`) — required before first use; gates access to `/app`.
3. **Main app** (`/app`) — shows the social feed (all users' posts, newest first), the current week's mood data, and the latest private report (if any).
4. **Posting** (`POST /api/post`) — free text + optional tags; classified into an emotion + confidence, saved, and echoed back to render in the feed.
5. **Weekly check-in** (`POST /api/checkin`) — submits 8 quiz answers; combined with the week's post emotions into a `WeeklyReport`; only the `supportive_message` is meant to be shown to the user, with `risk_tier` and scores kept server-side/private.

## Data model

Three SQLite tables (see `storage.init_db`):

| Table | Key columns | Purpose |
|---|---|---|
| `users` | `anon_id` (PK, SHA-256 of username), `display_name`, `password_hash`, `consent_given`, `created_at` | One row per anonymised account |
| `posts` | `anon_user_id`, `display_name`, `text`, `tags` (JSON), `emotion`, `confidence`, `week`, `submitted_at` | Every feed post, tagged with its classified emotion |
| `weekly_reports` | `anon_user_id`, `week`, `quiz_score`, `emotion_score`, `combined_score`, `risk_tier`, `needs_human_review`, `report_json`, `reviewer_action`, `reviewed_at` | One private report per user per week |

`storage._migrate()` adds `display_name`/`password_hash`/`tags` columns idempotently, so older databases (without those columns) are upgraded in place on `init_db()`.

## Scoring methodology

Defined entirely in `scoring.py`:

- **Quiz**: 8 ASSIST-inspired substance-use questions, each answered 0–3 (`Never` → `Often/Daily`). Summed to a raw score of 0–24. Higher = more concern; no inversion needed since the questions are framed negatively.
- **Emotion**: for each post in the current week, if its top emotion is one of `anger`, `disgust`, `fear`, `sadness`, its confidence is added to a running total. That total is divided by the number of posts (so a single low-confidence negative post barely moves the needle, but consistent negative posts push it up) and scaled to a 0–24 range.
- **Combine**: `combined = 0.6 × quiz_score + 0.4 × emotion_score`, clamped to [0, 24].
- **Tiering**: `0–8 → low`, `9–16 → medium`, `17–24 → high`.
- **Output**: a `WeeklyReport` containing the scores, an emotion-count summary, a `supportive_message` (tier-specific, written to never feel clinical or punitive), and `needs_human_review` (true for `medium`/`high`).

Run `python3 scoring.py` directly for a smoke test against three hand-built scenarios (low/medium/high) plus an empty-posts edge case.

## Privacy & anonymisation

- Usernames are **never stored**. `storage.anonymize()` SHA-256-hashes the raw username into a 64-character `anon_id`, which is the only identifier persisted or used to key the review queue.
- Passwords are SHA-256 hashed (unsalted — see below).
- Post **text** is stored as-is, since the user typed it themselves under consent (gated by the `/consent` flow before any posting is possible).
- The risk tier and combined score are explicitly documented as private/internal — the UI is expected to surface only `supportive_message`.
- Human reviewers only ever see the anonymous hash, never the original username, when working the review queue (`storage.get_review_queue`).

## API reference

All endpoints (other than `/login`) require an active session (`@login_required`).

| Method & path | Body | Response |
|---|---|---|
| `POST /api/post` | `{ "text": str, "tags": [str] }` | The saved post, shaped for the feed UI (id, avatar initials, emotion, score, etc.) |
| `POST /api/checkin` | `{ "answers": [int × 8] }` (each 0–3) | `{ risk_tier, combined_score, supportive_message, needs_human_review, quiz_score, emotion_score, week }` |

Page routes: `/`, `/login` (GET/POST), `/consent` (GET/POST), `/app` (GET), `/logout` (GET).

## Known issues / things to fix before a real demo

These were spotted while reading the code — worth triaging before judging/demo day:

1. **Missing templates.** `app.py` calls `render_template("login.html" | "consent.html" | "index.html", ...)`, but no `templates/` directory is in this upload. The Flask app will throw `TemplateNotFound` until those are added.
2. **`sample_data.py` calls `storage.save_post` with the old 5-argument signature** (`anon_id, text, emotion, confidence, week_id`), but `storage.save_post` now requires 7 positional args (`anon_user_id, display_name, text, tags, emotion, confidence, week_id`). Running `python3 sample_data.py` as-is will raise a `TypeError`.
3. **Quiz answer labels are inconsistent across files.** `scoring.ANSWER_LABELS` defines `0=Never, 1=Rarely, 2=Sometimes, 3=Often/Daily`, but the `__main__` smoke test docstring at the bottom of `scoring.py` prints `0=Never 1=Sometimes 2=Often 3=Almost always`. Pick one wording and use it consistently in the actual quiz UI.
4. **Unsalted password hashing.** `storage.hash_password` is a bare SHA-256 with no per-user salt, called out in the module docstring as a "production" gap — fine for a hackathon demo, but flag it explicitly if asked about security in judging.
5. **Hardcoded Flask secret key** (`app.secret_key = "safeguard-mauritius-goc2026-change-in-prod"`) — again, fine for a demo, should come from an environment variable in any real deployment.
6. **`api_post`'s keyword fallback** is a coarse stand-in for the real model and will misclassify anything outside its small word lists — worth mentioning as a known limitation rather than letting it look like the "real" detector during a live demo if the model hasn't finished loading.

## Team ownership

| Area | File(s) | Owner |
|---|---|---|
| Backend / API / SPA shell | `app.py` | Person 3 |
| Emotion detection | `detection.py` | Person 1 |
| Quiz & risk scoring | `scoring.py` | Person 2 |
| Persistence & anonymisation | `storage.py` | Person 4 |
| Demo data | `sample_data.py` | Shared |

---

## Attribution & Acknowledgements

### Origin of each component

| Component | File(s) | Origin | Notes |
|---|---|---|---|
| System architecture & design | all | **Team (original)** | Consent-first flow, anonymous hashing, quiz/emotion blend, human-review queue — all our own design |
| Flask routes & API | `app.py` | **Team (original)** | Written by Person 3; AI rewrote sections to fix integration bugs (see below) |
| Domain ensemble & keyword rules | `detection.py` | **Team (original)** | The ensemble wrapper, `_KEYWORD_RULES`, `_TAG_RULES`, and `FeedbackCalibrator` are ours; AI rewrote the file to fix bugs |
| Quiz & scoring logic | `scoring.py` | **Team (original)** | Written by Person 2; AI rewrote to fix errors |
| Database & anonymisation | `storage.py` | **Team (original)** | Written by Person 4; AI rewrote to fix bugs |
| Frontend SPA | `templates/index.html` | **Team (original)** | UI design and JS state machine are ours; AI fixed specific bugs in the JS |
| Emotion model weights | `detection.py` (`_get_pipeline`) | **External — HuggingFace** | `j-hartmann/emotion-english-distilroberta-base` — we call it, we do not own it |
| Screening questions | `scoring.py` (`QUIZ_QUESTIONS`) | **External inspiration — WHO ASSIST** | Wording and scoring formula adapted by us; not a clinical reproduction |
| Bug fixes & code cleanup | all | **AI-generated (Claude)** | Full list below |

### What the AI (Claude) generated

The base code for every file was written by the team. When bugs were found during testing, we described them to Claude (Anthropic) and the AI produced corrected versions of the relevant sections. The following were **written or rewritten by AI**:

| What | Where | Why AI was used |
|---|---|---|
| Comments backend (DB table + 2 API routes + JS modal) | `storage.py`, `app.py`, `index.html` | Backend was missing entirely; AI built it from our description |
| `switchTab()` auto-refresh logic | `index.html` | Tab switching wasn't fetching fresh data; AI rewrote the function |
| `giveFeedback()` fix | `index.html` | Thumbs-down never registered because `false` is falsy in a ternary; AI fixed by switching to `"up"`/`"down"` strings |
| Post ID collision fix | `index.html` | Demo posts (IDs 1–6) were blocking real DB posts from loading; AI changed them to negative IDs (-1 to -6) |
| `state.weekData` live update | `index.html`, `app.py` | `WEEK_DATA` constant never updated after posting; AI moved it into mutable state and added `/api/week-data` endpoint |
| `state.feedbackTotal/Correct` live update | `index.html`, `app.py` | Same pattern — moved from constant to mutable state |
| Report card CSS border fix | `index.html` | Double-border from conflicting `border` + `border-left` declarations; AI replaced with `box-shadow: inset` |
| `matplotlib` chart routes | `app.py` | Server-side PNG generation for the AI tab; AI wrote the chart code to our visual spec |
| GAN explainer section | `index.html` | AI wrote the 4-card GAN description to our spec |
| RL calibration UI | `index.html` | Progress bar, "Run RL calibration" button, and result display; AI built to our spec |
| Comment cleanup across all `.py` files | `detection.py`, `scoring.py`, `storage.py`, `app.py` | AI stripped verbose docstrings and replaced with single-line human descriptions per function |

All AI-generated code was reviewed, tested, and accepted by the team before being included. No AI output was blindly committed.

### Third-party model

**`j-hartmann/emotion-english-distilroberta-base`** (Hugging Face)
> Jochen Hartmann, "Emotion English DistilRoBERTa-base". https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/, 2022.

We call this model via the `transformers` `pipeline` API. We do not own it, we did not train it, and we did not modify its weights. Our contribution is the domain ensemble layer on top (`classify_with_ensemble` in `detection.py`) which adds substance-misuse keyword rules and tag signals, pushing accuracy from the model's published 66% F1 to ~83% on our target domain.

### Quiz instrument

The 8 screening questions (`QUIZ_QUESTIONS` in `scoring.py`) are inspired by the ASSIST screening tool:
> World Health Organization. *The Alcohol, Smoking and Substance Involvement Screening Test (ASSIST): Manual for use in primary care.* Geneva: WHO, 2010.

The exact wording and scoring formula are our own adaptation for a weekly self-check-in context and are not a clinical reproduction of ASSIST.

### Libraries & frameworks

| Library | Use | Licence |
|---|---|---|
| [Flask](https://flask.palletsprojects.com/) | Web framework | BSD-3-Clause |
| [Transformers](https://github.com/huggingface/transformers) | HuggingFace model pipeline | Apache-2.0 |
| [PyTorch](https://pytorch.org/) | Transformer inference backend | BSD-style |
| [matplotlib](https://matplotlib.org/) | Server-side chart generation | PSF/BSD |
| [SQLite](https://www.sqlite.org/) | Embedded database | Public domain |
