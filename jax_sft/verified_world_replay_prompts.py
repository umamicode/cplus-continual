"""Shared deterministic broad-prompt catalog for frozen-base replay.

Both the legacy generator and the standalone verified-world producer import
this module. Keeping one catalog prevents nominally shared replay arms from
silently drifting because prompt constants were copied into two producers.
"""

from __future__ import annotations

from typing import Any

import numpy as np


GENERAL_SYNTHETIC_TASKS = [
    (
        "math",
        "Create and solve one small quantitative example involving {topic}. "
        "Show the key step.",
    ),
    (
        "coding",
        "Write a concise Python example related to {topic}. Include code and "
        "one sentence explaining it.",
    ),
    (
        "debugging",
        "Diagnose one plausible implementation mistake involving {topic}, "
        "then give a concise fix.",
    ),
    (
        "science",
        "Explain the central mechanism behind {topic} in a scientifically "
        "careful way.",
    ),
    (
        "writing",
        "Write a polished short paragraph about {topic} with a clear opening "
        "and conclusion.",
    ),
    (
        "reasoning",
        "Compare two plausible claims about {topic}, identify the stronger "
        "claim, and justify the choice.",
    ),
    (
        "instruction_following",
        "Produce a concise, ordered checklist for working with {topic}. "
        "Follow the requested format exactly.",
    ),
    (
        "factual_qa",
        "Answer one concrete factual question about {topic}, including one "
        "important qualification.",
    ),
    (
        "planning",
        "Give a practical three-step plan for learning or applying {topic}.",
    ),
    (
        "editing",
        "Rewrite a vague explanation of {topic} so it is precise, concise, "
        "and easy to verify.",
    ),
]


GENERAL_SYNTHETIC_TOPICS = [
    "why regularization can reduce overfitting",
    "how to compute the area of a triangle from base and height",
    "when a hash map is preferable to a list",
    "why a binary search requires sorted input",
    "how gradient descent updates parameters",
    "the difference between precision and recall",
    "why antibiotics do not treat viral infections",
    "how vaccines train immune memory",
    "what comparative advantage means",
    "why correlation does not imply causation",
    "how photosynthesis stores energy",
    "the role of mitochondria in cells",
    "how to handle a division-by-zero bug",
    "how to reverse a string in Python",
    "how to parse a small CSV file",
    "why unit tests help refactoring",
    "how to estimate probability from repeated trials",
    "how supply and demand affect price",
    "why active reading improves comprehension",
    "how to evaluate a source for credibility",
    "how to prioritize tasks under a deadline",
    "why clear variable names matter",
    "how evaporation cools a surface",
    "why randomized controlled trials are useful",
    "how caching changes a system's time-memory tradeoff",
    "why database indexes speed up some queries",
    "how interest compounds over repeated periods",
    "why sampling bias distorts an estimate",
    "how plate tectonics changes Earth's surface",
    "why seasons result from axial tilt",
    "how constitutional checks and balances work",
    "why opportunity cost matters in a decision",
    "how a feedback loop can stabilize a system",
    "why accessibility improves product design",
    "how version control supports collaboration",
    "why a control group strengthens an experiment",
    "how a metaphor can clarify technical writing",
    "why an algorithm's worst case can matter",
    "how a budget constraint changes a plan",
    "why explicit assumptions improve an argument",
]


GENERAL_SYNTHETIC_FRAMINGS = [
    "Aim the response at a curious high-school student.",
    "Aim the response at a first-year university student.",
    "Aim the response at a software engineer outside the specialty.",
    "Aim the response at a busy general reader.",
    "Use a concrete example before the general rule.",
    "State the general rule before giving an example.",
    "Distinguish the most common misconception from the correct account.",
    "Include one boundary case where the usual rule needs care.",
    "Use no more than five compact sentences.",
    "Use a short heading followed by two bullet points.",
    "Use a numbered response with exactly three items.",
    "End with one sentence that can be checked independently.",
    "Prefer plain language and define any necessary technical term.",
    "Make the answer useful for a small real-world decision.",
    "Contrast a correct approach with one tempting failure mode.",
    "Separate observation, inference, and recommendation.",
    "Include one explicit assumption and explain why it matters.",
    "Give a minimal answer first, then one optional detail.",
    "Use an analogy, then state where the analogy stops working.",
    "Organize the response as problem, mechanism, and takeaway.",
    "Make uncertainty explicit without becoming vague.",
    "Provide one example and one counterexample.",
    "Focus on causal explanation rather than a list of facts.",
    "Conclude with a one-line practical test of understanding.",
]


GENERAL_REQUIRED_REPLAY_FAMILIES = frozenset(
    {"reasoning", "factual_qa", "coding", "writing", "instruction_following"}
)


def general_instruction_catalog_capacity() -> int:
    return (
        len(GENERAL_SYNTHETIC_TASKS)
        * len(GENERAL_SYNTHETIC_TOPICS)
        * len(GENERAL_SYNTHETIC_FRAMINGS)
    )


def make_general_instruction_rows(
    n_rows: int,
    seed: int,
    *,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    if n_rows < 0 or start_index < 0:
        raise ValueError("general replay row counts and offsets must be nonnegative")
    capacity = general_instruction_catalog_capacity()
    if start_index + n_rows > capacity:
        raise ValueError(
            "requested general replay prompts exceed the deterministic "
            f"catalog capacity ({start_index + n_rows}>{capacity})"
        )
    rng = np.random.default_rng(seed)
    topic_order = rng.permutation(len(GENERAL_SYNTHETIC_TOPICS)).tolist()
    framing_order = rng.permutation(len(GENERAL_SYNTHETIC_FRAMINGS)).tolist()
    rows: list[dict[str, Any]] = []
    for local_index in range(n_rows):
        prompt_index = start_index + local_index
        task_index = prompt_index % len(GENERAL_SYNTHETIC_TASKS)
        combination_index = prompt_index // len(GENERAL_SYNTHETIC_TASKS)
        topic_slot = combination_index % len(GENERAL_SYNTHETIC_TOPICS)
        framing_slot = (
            combination_index // len(GENERAL_SYNTHETIC_TOPICS)
        ) % len(GENERAL_SYNTHETIC_FRAMINGS)
        task_family, template = GENERAL_SYNTHETIC_TASKS[task_index]
        topic = GENERAL_SYNTHETIC_TOPICS[topic_order[topic_slot]]
        framing = GENERAL_SYNTHETIC_FRAMINGS[framing_order[framing_slot]]
        instruction = f"{template.format(topic=topic)}\n{framing}"
        rows.append(
            {
                "instruction": instruction,
                "synthetic_prompt": True,
                "synthetic_prompt_id": prompt_index,
                "synthetic_prompt_method": "general_seed",
                "synthetic_topic": topic,
                "synthetic_task_family": task_family,
                "synthetic_prompt_framing": framing,
                "synthetic_prompt_catalog_index": prompt_index,
            }
        )
    return rows
