"""Offline tests: fake judges stand in for a real model."""

from blind_judge import blind_judge, position_bias_check

Q = "Is the Earth flat?"
ANSWERS = {"gemini": "No. Satellite imagery and ship hulls vanishing first show curvature.",
           "mistral": "Yes, it looks flat from the ground."}


def content_judge(prompt: str) -> str:
    # Picks whichever label holds the satellite answer, wherever it sits.
    for label in "AB":
        block = prompt.split(f"Model {label}:")[1].split("Model ")[0]
        if "Satellite" in block:
            return f"WINNER: {label}\nREASON: Model {label} cites evidence."
    return "WINNER: A\nREASON: none"


def first_slot_judge(prompt: str) -> str:
    # Always picks whatever is shown first: pure position bias.
    return "WINNER: A\nREASON: Model A went first."


def test_names_hidden_from_judge():
    seen = {}
    def spy(prompt):
        seen["p"] = prompt
        return "WINNER: A\nREASON: ok"
    blind_judge(Q, ANSWERS, spy, seed=1)
    assert "gemini" not in seen["p"] and "mistral" not in seen["p"]


def test_winner_mapped_back_to_real_name():
    v = blind_judge(Q, ANSWERS, content_judge, seed=7)
    assert v.winner == "gemini"
    assert "gemini" in v.reason


def test_fair_judge_is_consistent():
    r = position_bias_check(Q, "gemini", ANSWERS["gemini"], "mistral", ANSWERS["mistral"], content_judge)
    assert r["consistent"]


def test_position_biased_judge_is_caught():
    r = position_bias_check(Q, "gemini", ANSWERS["gemini"], "mistral", ANSWERS["mistral"], first_slot_judge)
    assert not r["consistent"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
