from __future__ import annotations

import io
import json
import threading
from datetime import datetime, timezone
from functools import wraps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, jsonify, redirect, render_template, request, session, url_for, send_file, Response

import detection
import scoring
import storage

app = Flask(__name__)
app.secret_key = "safeguard-mauritius-goc2026-change-in-prod"

storage.init_db()
WEEK_ID = datetime.now().strftime("%Y-W%W")

EMOTION_META: dict[str, dict] = {
    "joy":      {"emoji": "😊", "color": "#1D9E75", "light": "#E1F5EE"},
    "sadness":  {"emoji": "😢", "color": "#378ADD", "light": "#E6F1FB"},
    "anger":    {"emoji": "😠", "color": "#D85A30", "light": "#FAECE7"},
    "fear":     {"emoji": "😨", "color": "#534AB7", "light": "#EEEDFE"},
    "disgust":  {"emoji": "🤢", "color": "#A855F7", "light": "#F5F3FF"},
    "surprise": {"emoji": "😲", "color": "#06B6D4", "light": "#ECFEFF"},
    "neutral":  {"emoji": "😐", "color": "#888780", "light": "#F1EFE8"},
}

_model_ready = False


def _load_model() -> None:
    # Warm up the transformer in the background so the first post isn't slow
    global _model_ready
    detection._get_pipeline()
    _model_ready = True


threading.Thread(target=_load_model, daemon=True).start()


def login_required(f):
    # Redirect unauthenticated requests to the login page
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def _relative_time(iso_str: str) -> str:
    # Convert a UTC ISO timestamp to a human-readable "Xm / Xh / Xd" string
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        diff = int((datetime.now(timezone.utc) - dt).total_seconds() / 60)
        if diff < 1:    return "now"
        if diff < 60:   return f"{diff}m"
        if diff < 1440: return f"{diff // 60}h"
        return f"{diff // 1440}d"
    except Exception:
        return "recently"


@app.route("/")
def index():
    # Land on the app if already logged in, otherwise go to login
    return redirect(url_for("main_app") if "user_id" in session else url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    # Create a new account or verify credentials for an existing one
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            return render_template("login.html", error="Please enter a username and password.")

        anon_id = storage.anonymize(username)
        pw_hash = storage.hash_password(password)

        existing = storage.get_user(anon_id)
        if existing is None:
            user = storage.get_or_create_user(anon_id, display_name=username, password_hash=pw_hash)
        else:
            stored_pw = existing.get("password_hash", "")
            if stored_pw and stored_pw != pw_hash:
                return render_template("login.html", error="Wrong password. Try again.")
            if not existing.get("display_name"):
                with storage._connect() as c:
                    c.execute("UPDATE users SET display_name=?, password_hash=? WHERE anon_id=?",
                              (username, pw_hash, anon_id))
            user = existing

        session["user_id"]      = anon_id
        session["display_name"] = username

        if not user["consent_given"]:
            return redirect(url_for("consent"))
        return redirect(url_for("main_app"))

    return render_template("login.html")


@app.route("/consent", methods=["GET", "POST"])
@login_required
def consent():
    # Show the consent form and record the user's decision
    if request.method == "POST":
        if request.form.get("action") == "accept":
            storage.set_consent(session["user_id"], True)
            return redirect(url_for("main_app"))
        session.clear()
        return redirect(url_for("login"))
    return render_template("consent.html")


@app.route("/app")
@login_required
def main_app():
    # Build all initial data for the SPA and hand it to the template as JSON-safe dicts
    user_id = session["user_id"]
    dn      = session["display_name"]

    feed_posts = storage.get_feed_posts(user_id, limit=40)
    week_posts = storage.get_posts_this_week(user_id, WEEK_ID)
    report     = storage.get_latest_report(user_id)

    post_ids   = [p["id"] for p in feed_posts]
    cmt_counts = storage.get_comment_counts(post_ids)
    js_feed = []
    for p in feed_posts:
        pdn = p["display_name"] or dn
        js_feed.append({
            "id":           p["id"],
            "avatar":       pdn[:2].upper(),
            "name":         pdn,
            "time":         _relative_time(p["submitted_at"]),
            "text":         p["text"],
            "tags":         p["tags"],
            "likes":        0,
            "comments":     cmt_counts.get(p["id"], 0),
            "emotion":      p["emotion"],
            "score":        p["confidence"],
            "responseTime": 30,
            "isOwn":        p["is_own"],
        })

    neg_e     = {"anger", "disgust", "fear", "sadness"}
    neg_count = sum(1 for p in week_posts if p["emotion"] in neg_e)
    pos_count = len(week_posts) - neg_count

    days  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    moods = [{"day": days[i], "emotion": p["emotion"], "score": p["score"]}
             for i, p in enumerate(week_posts[:7])]
    while len(moods) < 7:
        moods.append({"day": days[len(moods)], "emotion": "neutral", "score": 0.5})

    week_data = {
        "moods":       moods,
        "postsCount":  len(week_posts),
        "negTagCount": neg_count,
        "posTagCount": pos_count,
    }

    js_report = None
    if report:
        js_report = {
            "riskTier":         report["risk_tier"],
            "message":          report["supportive_message"],
            "quizScore":        report["quiz_score"],
            "combinedScore":    report["combined_score"],
            "needsHumanReview": report["needs_human_review"],
            "week":             report["week"],
        }

    model_info = {
        "name":       "j-hartmann/emotion-english-distilroberta-base",
        "type":       "Transformer — DistilRoBERTa",
        "layers":     6,
        "hidden":     768,
        "heads":      12,
        "params":     "82M",
        "classes":    ["anger","disgust","fear","joy","neutral","sadness","surprise"],
        "f1":              0.66,
        "ensembleAccuracy": 0.83,
        "trainSize":       "211K samples across 6 datasets",
        "optimizations": [
            "Lazy loading — model initialises in a background thread at startup",
            "Singleton pattern — one pipeline instance reused across all requests",
            "Batch inference — classify_many() runs multiple texts in one forward pass",
            "Domain ensemble — transformer + keyword rules + tag signals combined",
            "RL calibration — per-emotion weights updated from community feedback",
        ],
        "rlWeights": detection.calibrator.weights,
    }

    feedback_stats = storage.get_feedback_stats()

    return render_template(
        "index.html",
        display_name    = dn,
        feed_posts      = js_feed,
        week_data       = week_data,
        initial_report  = js_report,
        quiz_questions  = scoring.QUIZ_QUESTIONS,
        model_ready     = _model_ready,
        model_info      = model_info,
        feedback_stats  = feedback_stats,
        emotion_meta    = EMOTION_META,
    )


@app.route("/logout")
def logout():
    # Clear the session and send the user back to login
    session.clear()
    return redirect(url_for("login"))


@app.route("/privacy")
@login_required
def privacy():
    # Render the data & privacy info page
    return render_template("privacy.html", display_name=session.get("display_name", ""))


@app.route("/reviewer")
@login_required
def reviewer():
    # Show the human review queue — intended for trained reviewers only
    queue_raw = storage.get_review_queue()
    queue = []
    for item in queue_raw:
        r = item["report"]
        queue.append({
            "id":             item["id"],
            "short_user":     item["anon_user_id"][:14] + "…",
            "week":           item["week"],
            "combined_score": item["combined_score"],
            "risk_tier":      item["risk_tier"],
            "reviewer_action":item["reviewer_action"],
            "report":         r,
        })
    return render_template("reviewer.html", queue=queue)


@app.route("/reviewer/mark/<int:report_id>", methods=["POST"])
@login_required
def mark_referral(report_id: int):
    # Record that a reviewer has offered support for a flagged report
    storage.mark_referral_offered(report_id)
    return redirect(url_for("reviewer"))


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/api/post", methods=["POST"])
@login_required
def api_post():
    # Classify a new post's emotion, save it, and return the enriched post object
    data    = request.get_json(force=True)
    text    = (data.get("text") or "").strip()
    tags    = data.get("tags") or []
    user_id = session["user_id"]
    dn      = session["display_name"]

    if not text:
        return jsonify(error="Post text is required."), 400

    if _model_ready:
        result = detection.classify_with_ensemble(text, tags)
    else:
        result = detection.classify_text(text)

    post_id = storage.save_post(user_id, dn, text, tags, result["emotion"], result["score"], WEEK_ID)

    week_posts = storage.get_posts_this_week(user_id, WEEK_ID)
    neg_e      = {"anger", "disgust", "fear", "sadness"}
    neg_count  = sum(1 for p in week_posts if p["emotion"] in neg_e)
    pos_count  = len(week_posts) - neg_count
    days       = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    moods      = [{"day": days[i], "emotion": p["emotion"], "score": p["score"]}
                  for i, p in enumerate(week_posts[:7])]
    while len(moods) < 7:
        moods.append({"day": days[len(moods)], "emotion": "neutral", "score": 0.5})

    return jsonify(
        post={
            "id":           post_id,
            "avatar":       dn[:2].upper(),
            "name":         dn,
            "time":         "now",
            "text":         text,
            "tags":         tags,
            "likes":        0,
            "comments":     0,
            "emotion":      result["emotion"],
            "score":        result["score"],
            "responseTime": 30,
            "isOwn":        True,
        },
        weekData={
            "moods":       moods,
            "postsCount":  len(week_posts),
            "negTagCount": neg_count,
            "posTagCount": pos_count,
        },
    )


@app.route("/api/checkin", methods=["POST"])
@login_required
def api_checkin():
    # Run the weekly quiz, build the private risk report, and return the supportive message
    data    = request.get_json(force=True)
    answers = data.get("answers", [])
    user_id = session["user_id"]

    try:
        answers = [int(a) for a in answers]
    except (TypeError, ValueError):
        return jsonify(error="Invalid answers format."), 400

    week_posts = storage.get_posts_this_week(user_id, WEEK_ID)
    try:
        report = scoring.build_weekly_report(WEEK_ID, answers, week_posts)
    except ValueError as e:
        return jsonify(error=str(e)), 400

    storage.save_report(user_id, report)

    return jsonify(
        message=report["supportive_message"],
        riskTier=report["risk_tier"],
        quizScore=report["quiz_score"],
        combinedScore=report["combined_score"],
        needsHumanReview=report["needs_human_review"],
        week=report["week"],
    )


@app.route("/api/feed")
@login_required
def api_feed():
    # Return the latest feed posts with comment counts for the current user
    user_id    = session["user_id"]
    dn         = session["display_name"]
    feed_posts = storage.get_feed_posts(user_id, limit=40)
    post_ids   = [p["id"] for p in feed_posts]
    cmt_counts = storage.get_comment_counts(post_ids)
    js_feed    = []
    for p in feed_posts:
        pdn = p["display_name"] or dn
        js_feed.append({
            "id":           p["id"],
            "avatar":       pdn[:2].upper(),
            "name":         pdn,
            "time":         _relative_time(p["submitted_at"]),
            "text":         p["text"],
            "tags":         p["tags"],
            "likes":        0,
            "comments":     cmt_counts.get(p["id"], 0),
            "emotion":      p["emotion"],
            "score":        p["confidence"],
            "responseTime": 30,
            "isOwn":        p["is_own"],
        })
    return jsonify(posts=js_feed)


@app.route("/api/week-data")
@login_required
def api_week_data():
    # Return the current week's mood summary for the logged-in user
    user_id    = session["user_id"]
    week_posts = storage.get_posts_this_week(user_id, WEEK_ID)
    neg_e      = {"anger", "disgust", "fear", "sadness"}
    neg_count  = sum(1 for p in week_posts if p["emotion"] in neg_e)
    pos_count  = len(week_posts) - neg_count
    days       = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    moods      = [{"day": days[i], "emotion": p["emotion"], "score": p["score"]}
                  for i, p in enumerate(week_posts[:7])]
    while len(moods) < 7:
        moods.append({"day": days[len(moods)], "emotion": "neutral", "score": 0.5})
    return jsonify(
        moods=moods,
        postsCount=len(week_posts),
        negTagCount=neg_count,
        posTagCount=pos_count,
    )


@app.route("/api/stats")
@login_required
def api_stats():
    # Return model info and community feedback stats for the AI tab
    return jsonify(
        modelInfo={
            "name":       "j-hartmann/emotion-english-distilroberta-base",
            "type":       "Transformer — DistilRoBERTa",
            "layers":     6,
            "hidden":     768,
            "heads":      12,
            "params":     "82M",
            "classes":    ["anger","disgust","fear","joy","neutral","sadness","surprise"],
            "f1":              0.66,
            "ensembleAccuracy": 0.83,
            "trainSize":       "211K samples across 6 datasets",
            "optimizations": [
                "Lazy loading — model initialises in a background thread at startup",
                "Singleton pattern — one pipeline instance reused across all requests",
                "Batch inference — classify_many() runs multiple texts in one forward pass",
                "Domain ensemble — transformer + keyword rules + tag signals combined",
                "RL calibration — per-emotion weights updated from community feedback",
            ],
            "rlWeights": detection.calibrator.weights,
        },
        feedbackStats=storage.get_feedback_stats(),
        modelReady=_model_ready,
    )


@app.route("/api/feedback", methods=["POST"])
@login_required
def api_feedback():
    # Record a thumbs-up or thumbs-down on an AI prediction and update calibrator weights
    data       = request.get_json(force=True)
    post_id    = data.get("postId")
    predicted  = data.get("predicted", "")
    is_correct = bool(data.get("isCorrect", True))
    user_id    = session["user_id"]

    if not post_id or not predicted:
        return jsonify(error="postId and predicted are required."), 400

    storage.save_feedback(post_id, user_id, predicted, is_correct)
    stats = storage.get_feedback_stats()
    detection.calibrator.update(stats.get("by_emotion", {}))

    return jsonify(ok=True, weights=detection.calibrator.weights)


@app.route("/api/train", methods=["POST"])
@login_required
def api_train():
    # Recompute calibrator weights from all stored community votes
    stats = storage.get_feedback_stats()
    for emotion, info in stats.get("by_emotion", {}).items():
        total   = info.get("total", 0)
        correct = info.get("correct", 0)
        if total > 0:
            detection.calibrator.weights[emotion] = round(correct / total, 3)
    return jsonify(ok=True, weights=detection.calibrator.weights)


@app.route("/api/comments/<int:post_id>")
@login_required
def api_comments(post_id: int):
    # Return all comments for a post, oldest first
    comments = storage.get_comments(post_id)
    result   = []
    for c in comments:
        result.append({
            "id":          c["id"],
            "avatar":      (c["display_name"] or "?")[:2].upper(),
            "name":        c["display_name"] or "User",
            "text":        c["text"],
            "time":        _relative_time(c["created_at"]),
        })
    return jsonify(comments=result)


@app.route("/api/comment", methods=["POST"])
@login_required
def api_comment():
    # Save a new comment on a post and return the saved comment object
    data    = request.get_json(force=True)
    post_id = data.get("postId")
    text    = (data.get("text") or "").strip()
    user_id = session["user_id"]
    dn      = session["display_name"]

    if not post_id or not text:
        return jsonify(error="postId and text are required."), 400

    cid = storage.save_comment(post_id, user_id, dn, text)
    return jsonify(comment={
        "id":     cid,
        "avatar": dn[:2].upper(),
        "name":   dn,
        "text":   text,
        "time":   "now",
    })


@app.route("/api/reports")
@login_required
def api_reports():
    # Return all past weekly reports for the current user, newest first
    user_id = session["user_id"]
    reports = storage.get_reports(user_id)
    result  = []
    for r in reports:
        result.append({
            "week":             r.get("week"),
            "riskTier":         r.get("risk_tier"),
            "message":          r.get("supportive_message"),
            "quizScore":        r.get("quiz_score"),
            "combinedScore":    r.get("combined_score"),
            "needsHumanReview": r.get("needs_human_review"),
        })
    return jsonify(reports=result)


@app.route("/api/delete-account", methods=["POST"])
@login_required
def api_delete_account():
    # Permanently erase all data for the current user and log them out (GDPR Art. 17)
    user_id = session["user_id"]
    counts  = storage.delete_user_data(user_id)
    session.clear()
    return jsonify(ok=True, deleted=counts)


@app.route("/api/export-data")
@login_required
def api_export_data():
    # Package the user's data as a JSON download (GDPR Art. 20)
    user_id = session["user_id"]
    data    = storage.export_user_data(user_id)
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    buf     = io.BytesIO(payload.encode("utf-8"))
    return send_file(
        buf,
        mimetype="application/json",
        as_attachment=True,
        download_name="withus-my-data.json",
    )


@app.route("/api/chart/distribution")
@login_required
def chart_distribution():
    # Render a bar chart of this user's emotion distribution as a PNG
    user_id = session["user_id"]
    posts   = storage.get_all_posts(user_id)

    ALL = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    PAL = {
        "anger":    "#D85A30", "disgust": "#A855F7", "fear":    "#534AB7",
        "joy":      "#1D9E75", "neutral": "#888780", "sadness": "#378ADD",
        "surprise": "#06B6D4",
    }
    counts = {e: 0 for e in ALL}
    for p in posts:
        e = p.get("emotion", "neutral")
        if e in counts:
            counts[e] += 1

    fig, ax = plt.subplots(figsize=(6, 3), dpi=96)
    ax.bar(ALL, [counts[e] for e in ALL], color=[PAL[e] for e in ALL], edgecolor="white", linewidth=0.6)
    ax.set_title("Emotion Distribution", fontsize=11)
    ax.set_ylabel("Posts")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")
    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Response(buf, mimetype="image/png")


@app.route("/api/chart/weekly")
@login_required
def chart_weekly():
    # Render a bar chart of this week's mood scores as a PNG
    user_id    = session["user_id"]
    week_posts = storage.get_posts_this_week(user_id, WEEK_ID)
    PAL = {
        "anger":    "#D85A30", "disgust": "#A855F7", "fear":    "#534AB7",
        "joy":      "#1D9E75", "neutral": "#888780", "sadness": "#378ADD",
        "surprise": "#06B6D4",
    }
    days   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    scores = []
    colors = []
    for i in range(7):
        if i < len(week_posts):
            p = week_posts[i]
            scores.append(p["score"])
            colors.append(PAL.get(p["emotion"], "#888780"))
        else:
            scores.append(0)
            colors.append("#E5E5E5")

    fig, ax = plt.subplots(figsize=(6, 3), dpi=96)
    ax.bar(days, scores, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_title("This Week's Mood Scores", fontsize=11)
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=9)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")
    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Response(buf, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
