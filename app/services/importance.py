from __future__ import annotations

import re


PATTERN_WEIGHTS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"(지침\s*변경|instruction\s*change|new\s*policy)", re.IGNORECASE), 0.45),
    (re.compile(r"(정정|수정|교정|오류|실수|fix|wrong|correct)", re.IGNORECASE), 0.35),
    (re.compile(r"(확정|결정|합의|final\s+decision|adopt)", re.IGNORECASE), 0.30),
]


def score_turn_importance(text: str) -> float:
    score = 0.0
    for pattern, weight in PATTERN_WEIGHTS:
        if pattern.search(text):
            score += weight
    return min(1.0, round(score, 6))
