from app.services.importance import score_turn_importance


def test_importance_detects_instruction_change_and_fix() -> None:
    text = "지침 변경 후 오류를 정정하고 최종 결정으로 확정했다."
    score = score_turn_importance(text)
    assert score > 0.7


def test_importance_returns_zero_when_no_signal() -> None:
    text = "continuous batching은 처리량 향상에 유리하다."
    score = score_turn_importance(text)
    assert score == 0.0
