from __future__ import annotations

import threading
from typing import TypedDict


class EmotionEntry(TypedDict):
    emotion: str
    score: float
    method: str


_pipe = None
_pipe_lock = threading.Lock()

# Substance-misuse keywords that boost emotion confidence when found in post text
_KEYWORD_RULES: dict[str, list[str]] = {
    "sadness": ["ashamed", "shame", "relapsed", "relapse", "slipped", "slip", "failed",
                "withdrawal", "withdrawing", "hopeless", "alone", "isolated", "guilty",
                "disappoint", "worthless", "regret"],
    "fear":    ["craving", "crave", "triggered", "trigger", "tempt", "urge", "struggling",
                "withdrawal", "scared", "anxious", "anxiety", "dread", "panic", "nervous",
                "resisting", "pressure"],
    "joy":     ["sober", "clean", "recovery", "resisted", "proud", "milestone", "weeks clean",
                "days clean", "month clean", "grateful", "strong", "hopeful", "better",
                "improving", "made it"],
    "anger":   ["angry", "furious", "frustrated", "unfair", "blame", "rage", "hate"],
}

# Recovery tags carry very strong signals — almost as good as a direct label
_TAG_RULES: dict[str, list[str]] = {
    "sadness": ["#relapsed", "#ashamed", "#withdrawing", "#isolated", "#seeking-help"],
    "fear":    ["#craving", "#triggered", "#peer-pressure", "#withdrawing"],
    "joy":     ["#sober-today", "#clean-week", "#resisted", "#proud",
                "#supported", "#strong", "#recovery"],
    "anger":   [],
}


def _keyword_vote(text: str, tags: list[str]) -> dict[str, float]:
    # Score each emotion based on how many matching keywords and tags appear
    lower = text.lower()
    votes: dict[str, float] = {}
    for emotion, words in _KEYWORD_RULES.items():
        hits = sum(1 for w in words if w in lower)
        if hits:
            votes[emotion] = votes.get(emotion, 0.0) + hits * 0.12
    for emotion, tag_list in _TAG_RULES.items():
        hits = sum(1 for t in (tags or []) if t in tag_list)
        if hits:
            votes[emotion] = votes.get(emotion, 0.0) + hits * 0.28
    return votes


class FeedbackCalibrator:
    # Adjusts how confident the model is per emotion, based on community thumbs-up/down votes
    MIN_SAMPLES = 5

    def __init__(self) -> None:
        self._weights: dict[str, float] = {}
        self._lock = threading.Lock()

    def update(self, by_emotion: dict[str, dict]) -> dict[str, float]:
        # Rebuild calibration weights from the latest feedback stats
        new_weights: dict[str, float] = {}
        for emotion, stats in by_emotion.items():
            total = stats.get("total", 0)
            correct = stats.get("correct", 0)
            if total >= self.MIN_SAMPLES:
                new_weights[emotion] = round(correct / total, 4)
        with self._lock:
            self._weights = new_weights
        return dict(new_weights)

    def calibrate(self, entry: EmotionEntry) -> EmotionEntry:
        # Scale the confidence score up or down based on community accuracy for that emotion
        with self._lock:
            factor = self._weights.get(entry["emotion"])
        if factor is None:
            return entry
        return EmotionEntry(
            emotion=entry["emotion"],
            score=round(min(1.0, entry["score"] * factor), 4),
            method=entry.get("method", "transformer") + "+rl",
        )

    @property
    def weights(self) -> dict[str, float]:
        with self._lock:
            return dict(self._weights)


# Shared calibrator — updated when the user clicks "Run RL calibration step"
calibrator = FeedbackCalibrator()


def _get_pipeline():
    # Load the transformer model once and reuse it for every request
    global _pipe
    if _pipe is None:
        with _pipe_lock:
            if _pipe is None:
                from transformers import pipeline as hf_pipeline
                print("[detection] Loading model — downloads automatically on first run...")
                _pipe = hf_pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=1,
                )
                print("[detection] Model ready.")
    return _pipe


def _extract(raw) -> EmotionEntry:
    # Normalise the raw pipeline output into a clean EmotionEntry dict
    item = raw[0] if isinstance(raw, list) else raw
    if isinstance(item, list):
        item = item[0]
    return EmotionEntry(
        emotion=item["label"].lower(),
        score=round(float(item["score"]), 4),
        method="transformer",
    )


def classify_text(text: str) -> EmotionEntry:
    # Run the transformer on a single piece of text and return the top emotion
    pipe = _get_pipeline()
    result = pipe(text, top_k=1)
    return _extract(result)


def classify_with_ensemble(text: str, tags: list[str] | None = None) -> EmotionEntry:
    # Combine transformer + keyword rules + tag signals for higher domain accuracy
    pipe_result = classify_text(text)
    votes = _keyword_vote(text, tags or [])

    if not votes:
        return calibrator.calibrate(pipe_result)

    top_kw_emotion = max(votes, key=lambda e: votes[e])
    kw_weight = votes[top_kw_emotion]

    if top_kw_emotion == pipe_result["emotion"]:
        # Both sources agree — boost the confidence
        boosted = min(1.0, pipe_result["score"] + kw_weight)
        result = EmotionEntry(emotion=top_kw_emotion, score=round(boosted, 4), method="ensemble")
    elif kw_weight >= 0.30:
        # Strong keyword signal overrides the transformer
        base_score = min(1.0, 0.65 + kw_weight)
        result = EmotionEntry(emotion=top_kw_emotion, score=round(base_score, 4), method="ensemble")
    else:
        # Weak disagreement — trust the transformer
        result = EmotionEntry(emotion=pipe_result["emotion"], score=pipe_result["score"], method="ensemble")

    return calibrator.calibrate(result)


def classify_many(texts: list[str]) -> list[EmotionEntry]:
    # Classify a batch of texts in one forward pass for speed
    if not texts:
        return []
    pipe = _get_pipeline()
    results = pipe(texts, top_k=1)
    return [_extract(r) for r in results]
