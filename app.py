from __future__ import annotations

import threading
from datetime import datetime, timezone
from functools import wraps

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

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
        display_name   = dn,
        feed_posts     = js_feed,
        week_data      = week_data,
        initial_report = js_report,
        quiz_questions = scoring.QUIZ_QUESTIONS,
        model_ready    = _model_ready,
        model_info     = model_info,
        feedback_stats = feedback_stats,
    )


@app.route("/logout")
def logout():
    # Clear the session and send the user back to the login page
    session.clear()
    return redirect(url_for("login"))


@app.route("/api/feedback", methods=["POST"])
@login_required
def api_feedback():
    # Save a thumbs-up/down on an emotion prediction and return the updated community accuracy
    data       = request.get_json() or {}
    post_id    = data.get("post_id")
    predicted  = data.get("predicted_emotion", "")
    is_correct = bool(data.get("is_correct", True))
    storage.save_feedback(post_id, session["user_id"], predicted, is_correct)
    stats = storage.get_feedback_stats()
    return jsonify({"ok": True, "community_accuracy": stats["accuracy"]})


@app.route("/api/feed")
@login_required
def api_feed():
    # Return the latest feed posts so the JS can refresh without a page reload
    user_id    = session["user_id"]
    dn         = session["display_name"]
    feed_posts = storage.get_feed_posts(user_id, limit=40)
    post_ids   = [p["id"] for p in feed_posts]
    cmt_counts = storage.get_comment_counts(post_ids)
    js_feed = []
    for p in feed_posts:
        pdn = p["display_name"] or dn
        js_feed.append({
            "id":       p["id"],
            "avatar":   pdn[:2].upper(),
            "name":     pdn,
            "time":     _relative_time(p["submitted_at"]),
            "text":     p["text"],
            "tags":     p["tags"],
            "likes":    0,
            "comments": cmt_counts.get(p["id"], 0),
            "emotion":  p["emotion"],
            "score":    p["confidence"],
            "responseTime": 30,
            "isOwn":    p["is_own"],
        })
    return jsonify(js_feed)


@app.route("/api/chart/distribution")
@login_required
def api_chart_distribution():
    # Render a matplotlib bar chart of emotion distribution for the current user and return it as PNG
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    posts = storage.get_all_posts(session["user_id"])

    EMOTION_COLORS = {
        "joy": "#1D9E75", "sadness": "#378ADD", "fear": "#534AB7",
        "anger": "#D85A30", "disgust": "#A855F7", "surprise": "#06B6D4", "neutral": "#888780",
    }
    EMOTION_LABELS = {
        "joy": "Joy", "sadness": "Sadness", "fear": "Anxiety",
        "anger": "Anger", "disgust": "Disgust", "surprise": "Surprise", "neutral": "Neutral",
    }

    counts: dict[str, int] = {e: 0 for e in EMOTION_COLORS}
    for p in posts:
        e = p.get("emotion", "neutral")
        if e in counts:
            counts[e] += 1

    items = [(e, counts[e]) for e in EMOTION_COLORS if counts[e] > 0] or [("neutral", 0)]
    emotions, vals = zip(*items)
    colors = [EMOTION_COLORS[e] for e in emotions]
    labels = [EMOTION_LABELS[e] for e in emotions]

    BG = "#F5F4F0"
    fig, ax = plt.subplots(figsize=(5, max(2, len(items) * 0.55)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bars = ax.barh(labels, vals, color=colors, height=0.55, edgecolor="none")
    ax.spines[:].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelsize=9, colors="#888780")
    ax.set_xticks([])
    ax.invert_yaxis()

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_width() + max(vals) * 0.02, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", color="#2c2c2a", fontsize=9, fontweight="bold",
        )

    ax.set_xlim(0, max(vals) * 1.18)
    plt.tight_layout(pad=0.4)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return app.response_class(buf.read(), mimetype="image/png",
                              headers={"Cache-Control": "no-store"})


@app.route("/api/chart/weekly")
@login_required
def api_chart_weekly():
    # Render a matplotlib line chart of this week's daily risk signal and return it as PNG
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    week_posts = storage.get_posts_this_week(session["user_id"], WEEK_ID)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    moods = [{"day": days[i], "emotion": p["emotion"], "score": p["score"]}
             for i, p in enumerate(week_posts[:7])]
    while len(moods) < 7:
        moods.append({"day": days[len(moods)], "emotion": "neutral", "score": 0.5})

    EMOTION_COLORS = {
        "joy": "#1D9E75", "sadness": "#378ADD", "fear": "#534AB7",
        "anger": "#D85A30", "disgust": "#A855F7", "surprise": "#06B6D4", "neutral": "#CCCCCC",
    }
    RISK_SCORE = {"anger": 1.0, "disgust": 0.9, "fear": 0.8, "sadness": 0.7,
                  "neutral": 0.4, "surprise": 0.3, "joy": 0.1}

    scores     = [RISK_SCORE.get(m["emotion"], 0.5) * m["score"] for m in moods]
    dot_colors = [EMOTION_COLORS.get(m["emotion"], "#888") for m in moods]
    xlabels    = [m["day"] for m in moods]

    BG = "#F5F4F0"
    fig, ax = plt.subplots(figsize=(5, 2.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.fill_between(range(7), scores, alpha=0.12, color="#534AB7")
    ax.plot(range(7), scores, color="#534AB7", linewidth=1.5, zorder=2)
    for i, (s, c) in enumerate(zip(scores, dot_colors)):
        ax.scatter(i, s, color=c, s=55, zorder=3, edgecolors="white", linewidths=1)

    ax.set_xticks(range(7))
    ax.set_xticklabels(xlabels, fontsize=9, color="#888780")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["Low", "Mid", "High"], fontsize=8, color="#888780")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E5E3DB")
    ax.spines["bottom"].set_color("#E5E3DB")
    ax.tick_params(left=False, bottom=False)
    ax.set_title("Weekly risk signal", fontsize=10, color="#2c2c2a", pad=6, loc="left")

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return app.response_class(buf.read(), mimetype="image/png",
                              headers={"Cache-Control": "no-store"})


@app.route("/api/week-data")
@login_required
def api_week_data():
    # Return fresh week stats so the JS can update the post count and mood chart without reloading
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
    return jsonify({
        "moods":       moods,
        "postsCount":  len(week_posts),
        "negTagCount": neg_count,
        "posTagCount": pos_count,
    })


@app.route("/api/stats")
@login_required
def api_stats():
    # Return live community feedback stats for the AI tab
    return jsonify(storage.get_feedback_stats())


@app.route("/api/train", methods=["POST"])
@login_required
def api_train():
    # Run one RL calibration step — recomputes per-emotion confidence weights from community votes
    stats       = storage.get_feedback_stats()
    new_weights = detection.calibrator.update(stats["by_emotion"])
    total       = stats["total"]
    accuracy    = stats["accuracy"]
    return jsonify({
        "ok":           True,
        "samples_used": total,
        "accuracy":     accuracy,
        "calibrated_emotions": len(new_weights),
        "weights":      new_weights,
        "message":      (
            f"Calibrated {len(new_weights)} emotion(s) from {total} training samples. "
            f"Community accuracy: {accuracy}%." if total else
            "No feedback yet — rate predictions in the Feed to start training."
        ),
    })


@app.route("/api/comments/<int:post_id>")
@login_required
def api_get_comments(post_id: int):
    # Return all comments for a post formatted for the UI
    comments = storage.get_comments(post_id)
    return jsonify([
        {
            "id":     c["id"],
            "avatar": (c["display_name"] or "?")[:2].upper(),
            "name":   c["display_name"] or "User",
            "text":   c["text"],
            "time":   _relative_time(c["created_at"]),
        }
        for c in comments
    ])


@app.route("/api/comment", methods=["POST"])
@login_required
def api_post_comment():
    # Save a new comment and return it ready to render, plus the new total count
    data    = request.get_json() or {}
    post_id = data.get("post_id")
    text    = (data.get("text") or "").strip()
    if not post_id or not text:
        return jsonify({"error": "post_id and text required"}), 400
    dn = session["display_name"]
    storage.save_comment(post_id, session["user_id"], dn, text)
    count = len(storage.get_comments(post_id))
    return jsonify({
        "ok":    True,
        "avatar": dn[:2].upper(),
        "name":   dn,
        "text":   text,
        "time":   "now",
        "count":  count,
    })


@app.route("/privacy")
@login_required
def privacy():
    # Serve the privacy / data rights page
    return render_template("privacy.html", display_name=session["display_name"])


@app.route("/api/delete-account", methods=["POST"])
@login_required
def api_delete_account():
    # Permanently delete the current user's data and log them out (GDPR Art. 17)
    counts = storage.delete_user_data(session["user_id"])
    session.clear()
    return jsonify({"deleted": counts})


@app.route("/api/export-data")
@login_required
def api_export_data():
    # Bundle everything the app holds about this user into a downloadable JSON file (GDPR Art. 20)
    import json as _json
    data = storage.export_user_data(session["user_id"])
    return app.response_class(
        response=_json.dumps(data, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=withus-my-data.json"},
    )


@app.route("/api/post", methods=["POST"])
@login_required
def api_post():
    # Classify the submitted text with the ensemble, save the post, and return it ready to render
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    tags = data.get("tags", [])

    if not text:
        return jsonify({"error": "No text"}), 400

    if _model_ready:
        entry   = detection.classify_with_ensemble(text, tags)
        emotion = entry["emotion"]
        score   = entry["score"]
    else:
        lower = text.lower()
        neg   = ["sad","tired","anxious","scared","hopeless","alone","empty","dread","hurt",
                 "craving","relapsed","ashamed","withdrawal","triggered"]
        pos   = ["happy","good","great","grateful","proud","excited","peaceful","love","amazing",
                 "sober","clean","resisted","recovery","strong"]
        if   sum(1 for w in neg if w in lower) > sum(1 for w in pos if w in lower):
            emotion, score = "sadness", 0.68
        elif any(w in lower for w in pos):
            emotion, score = "joy", 0.68
        else:
            emotion, score = "neutral", 0.60

    dn = session["display_name"]
    storage.save_post(session["user_id"], dn, text, tags, emotion, score, WEEK_ID)

    return jsonify({
        "id":           int(datetime.now().timestamp() * 1000) % 9_000_000 + 1_000_000,
        "avatar":       dn[:2].upper(),
        "name":         dn,
        "time":         "now",
        "text":         text,
        "tags":         tags,
        "likes":        0,
        "comments":     0,
        "emotion":      emotion,
        "score":        round(score, 3),
        "responseTime": 30,
        "isOwn":        True,
    })


@app.route("/api/checkin", methods=["POST"])
@login_required
def api_checkin():
    # Run the weekly screening quiz and save the private risk report
    data    = request.get_json() or {}
    answers = data.get("answers", [])

    try:
        report = scoring.build_weekly_report(
            WEEK_ID, answers,
            storage.get_posts_this_week(session["user_id"], WEEK_ID),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    storage.save_report(session["user_id"], report)
    return jsonify({
        "risk_tier":          report["risk_tier"],
        "combined_score":     report["combined_score"],
        "supportive_message": report["supportive_message"],
        "needs_human_review": report["needs_human_review"],
        "quiz_score":         report["quiz_score"],
        "emotion_score":      report["emotion_score"],
        "week":               report["week"],
    })


if __name__ == "__main__":
    print("Starting WithUs — open http://localhost:5000")
    app.run(debug=True, port=5000)
