"""
Blind LLM-as-judge.

The judge model never sees which model wrote which answer. Answers are
shuffled and relabelled "Model A", "Model B", ... before judging, and the
real names are mapped back only after the verdict has been parsed. This is
the same idea Ramdos Arena uses in production so that no model can favour
its own family.

`position_bias_check` runs the same pair in both orders. A fair judge picks
the same winner both times; a judge that flips is reacting to position,
not content.

The judge is any function that takes a prompt string and returns text, so
it works with a local llama.cpp model, Gemini, Claude, OpenAI or Groq.
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass
from typing import Callable

JudgeFn = Callable[[str], str]

PROMPT = """You are an impartial judge. Several anonymous models answered the
same question. Judge ONLY on accuracy, reasoning and evidence. Ignore length,
style and confidence.

Question:
{question}

{answers}

Reply in exactly this format:
WINNER: <letter>
REASON: <one or two sentences>"""


@dataclass
class Verdict:
    winner: str | None      # real model name, or None if the reply was unparseable
    reason: str             # reason text with letters swapped back to real names
    label_map: dict         # e.g. {"A": "gemini", "B": "mistral"}
    raw: str                # untouched judge output


def _build_prompt(question: str, labelled: list[tuple[str, str]]) -> str:
    blocks = "\n\n".join(f"Model {label}:\n{text.strip()}" for label, text in labelled)
    return PROMPT.format(question=question.strip(), answers=blocks)


def _humanize(reason: str, label_map: dict) -> str:
    for label, name in label_map.items():
        reason = re.sub(rf"\b(Model|Answer|Response)\s+{label}\b", name, reason)
    return reason


def blind_judge(question: str, answers: dict[str, str], judge: JudgeFn,
                seed: int | None = None, order: list[str] | None = None) -> Verdict:
    """Judge `answers` ({model_name: answer}) without revealing model names."""
    names = list(order) if order else list(answers)
    if not order:
        random.Random(seed).shuffle(names)
    labels = string.ascii_uppercase[: len(names)]
    label_map = dict(zip(labels, names))

    raw = judge(_build_prompt(question, [(l, answers[n]) for l, n in label_map.items()]))

    m = re.search(r"WINNER:\s*(?:Model\s+)?([A-Z])\b", raw, re.I)
    r = re.search(r"REASON:\s*(.+)", raw, re.I | re.S)
    winner = label_map.get(m.group(1).upper()) if m else None
    reason = _humanize(r.group(1).strip(), label_map) if r else ""
    return Verdict(winner, reason, label_map, raw)


def position_bias_check(question: str, a_name: str, a_text: str,
                        b_name: str, b_text: str, judge: JudgeFn) -> dict:
    """Judge the pair in both orders and report whether the winner holds."""
    answers = {a_name: a_text, b_name: b_text}
    first = blind_judge(question, answers, judge, order=[a_name, b_name])
    second = blind_judge(question, answers, judge, order=[b_name, a_name])
    return {
        "winner_a_first": first.winner,
        "winner_b_first": second.winner,
        "consistent": first.winner is not None and first.winner == second.winner,
    }
