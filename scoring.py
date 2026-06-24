from __future__ import annotations

from typing import TypedDict


class EmotionEntry(TypedDict):
    emotion: str
    score: float


class WeeklyReport(TypedDict):
    week: str
    emotion_summary: dict[str, int]
    quiz_score: int
    emotion_score: float
    combined_score: float
    risk_tier: str
    supportive_message: str
    needs_human_review: bool


# Quiz weight is higher because what the user says directly is more reliable than what the AI infers
QUIZ_WEIGHT = 0.60
EMOTION_WEIGHT = 0.40

# Combined score runs 0–24; these cut-offs define low / medium / high risk
MEDIUM_THRESHOLD = 9
HIGH_THRESHOLD = 17

NEGATIVE_EMOTIONS = {"anger", "disgust", "fear", "sadness"}

# Messages the user actually sees — warm and supportive, never a diagnosis
SUPPORTIVE_MESSAGES = {
    "low": (
        "Your responses this week suggest you're managing well. Keep building on the "
        "routines and connections that are helping you stay grounded."
    ),
    "medium": (
        "It looks like this week had some difficult moments. That takes courage to "
        "acknowledge. Consider talking to someone you trust, or reaching out to one "
        "of the support services below — you don't have to face this alone."
    ),
    "high": (
        "Your check-in suggests this has been a tough week. Please know that support "
        "is available and reaching out is a sign of strength, not weakness. "
        "A trained reviewer has been notified using your anonymous code only — "
        "no punitive action will ever be taken."
    ),
}

# Eight ASSIST-inspired questions — higher answer means more substance use, no inversion needed
QUIZ_QUESTIONS: list[str] = [
    "In the past 7 days, how often did you use any substance (alcohol, cannabis, tobacco, or other drugs)?",
    "How often did you feel a strong urge or craving to use a substance?",
    "How often did substance use get in the way of your school, work, or home responsibilities?",
    "How often did substance use cause problems with your friends, family, or relationships?",
    "How often did you feel physically unwell as a result of substance use (hangover, withdrawal, or side effects)?",
    "How often did you find it hard to say no when substances were offered to you?",
    "How often did you use a substance to cope with stress, anxiety, or difficult emotions?",
    "How often did you need alcohol or another substance to help you sleep?",
]

ANSWER_LABELS = {
    0: "Never",
    1: "Rarely",
    2: "Sometimes",
    3: "Often / Daily",
}


def score_quiz(answers: list[int]) -> int:
    # Add up 8 answers (each 0–3) and return the total out of 24
    if len(answers) != 8:
        raise ValueError(f"Expected 8 answers, got {len(answers)}.")
    for i, val in enumerate(answers):
        if not isinstance(val, int) or val < 0 or val > 3:
            raise ValueError(f"Answer {i + 1} is '{val}' — must be an integer between 0 and 3.")
    return sum(answers)


def score_emotions(weekly_posts: list[EmotionEntry]) -> tuple[float, dict[str, int]]:
    # Count negative emotion signals from the week's posts and scale them to 0–24
    summary: dict[str, int] = {e: 0 for e in ("anger", "disgust", "fear", "sadness")}

    if not weekly_posts:
        return 0.0, summary

    raw_negative = 0.0
    for entry in weekly_posts:
        emotion = entry["emotion"].lower()
        conf = float(entry["score"])
        if emotion in NEGATIVE_EMOTIONS:
            raw_negative += conf
            if emotion in summary:
                summary[emotion] += 1

    max_possible = float(len(weekly_posts))
    proportion = min(raw_negative / max_possible, 1.0)
    emotion_score = round(proportion * 24, 2)
    return emotion_score, summary


def combine_and_tier(quiz_score: int | float, emotion_score: float) -> tuple[float, str]:
    # Blend quiz (60%) and emotion (40%) scores, then place the result in a risk tier
    quiz_risk = float(quiz_score)
    combined = round(QUIZ_WEIGHT * quiz_risk + EMOTION_WEIGHT * float(emotion_score), 2)
    combined = max(0.0, min(24.0, combined))

    if combined >= HIGH_THRESHOLD:
        tier = "high"
    elif combined >= MEDIUM_THRESHOLD:
        tier = "medium"
    else:
        tier = "low"

    return combined, tier


def build_weekly_report(
    week_id: str,
    answers: list[int],
    weekly_posts: list[EmotionEntry],
) -> WeeklyReport:
    # Run the quiz, score the week's posts, combine them, and return the full private report
    quiz_score = score_quiz(answers)
    emotion_score, emotion_summary = score_emotions(weekly_posts)
    combined_score, risk_tier = combine_and_tier(quiz_score, emotion_score)

    return WeeklyReport(
        week=week_id,
        emotion_summary=emotion_summary,
        quiz_score=quiz_score,
        emotion_score=emotion_score,
        combined_score=combined_score,
        risk_tier=risk_tier,
        supportive_message=SUPPORTIVE_MESSAGES[risk_tier],
        needs_human_review=(risk_tier in {"medium", "high"}),
    )
