"""Answer-based scoring rules (shared by rerun_generation.py and analyse_rerun.py).

Deliberately simple and documented, so a reviewer can audit them in a minute.

ARITHMETIC (24 tasks)
  gold   = last number in the reference sentence (commas removed). Checked by hand: for all
           24 references the last number is the answer (e.g. "144 divided by 12 is 12" -> 12).
  pred   = (1) the number in the LAST "answer"-style phrase, if any:
               "answer is X", "answer: X", "final answer ... X", "\\boxed{X}", "= X" at the
               very end of the output, "**X**" (bold) after the word answer;
           (2) otherwise the LAST number anywhere in the output.
           Numbers: optional sign, thousands commas removed, decimals kept, "$"/"₹" ignored.
  arith_correct      = float(pred) == float(gold)  (abs tol 1e-6)
  arith_gold_present = gold appears among ALL numbers in the output (looser; the 26 Oct audit
                       measure)

FACTUAL QA (23 tasks)
  The references are full sentences, so each task gets a short key answer with a few aliases
  (FACTUAL_KEYS below). factual_correct = any alias regex matches the normalised output
  (lower-cased, unicode subscripts mapped, commas removed in numbers). This is "normalised
  containment"; it does not penalise extra text, and it would accept e.g. "not Mars, Jupiter"
  for Jupiter (rare in practice).
"""

from __future__ import annotations

import re

NUM_RE = r"[-+]?\d+(?:\.\d+)?"


def _clean(text: str) -> str:
    t = str(text) if text is not None else ""
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)     # 2,400 -> 2400
    t = t.replace("$", " ").replace("₹", " ")
    return t


def all_numbers(text: str) -> list[str]:
    return re.findall(NUM_RE, _clean(text))


ANSWER_PATTERNS = [
    r"\\boxed\{\s*(" + NUM_RE + r")",
    r"answer\s*(?:is|:|=|would be|will be)\s*[:\-]?\s*\**\s*(?:approximately\s*|about\s*)?\$?\s*(" + NUM_RE + r")",
    r"final answer[^0-9\-+]{0,40}?(" + NUM_RE + r")",
]


def extract_arith_pred(text: str):
    t = _clean(text)
    tl = t.lower()
    best = None  # (position, value)
    for pat in ANSWER_PATTERNS:
        for m in re.finditer(pat, tl):
            if best is None or m.start() >= best[0]:
                best = (m.start(), m.group(1))
    if best is not None:
        return best[1], "pattern"
    nums = re.findall(NUM_RE, t)
    if nums:
        return nums[-1], "last_number"
    return None, "none"


def arith_gold(reference: str) -> str:
    return all_numbers(reference)[-1]


def _eq(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def score_arith(output: str, reference: str) -> dict:
    gold = arith_gold(reference)
    pred, how = extract_arith_pred(output)
    return dict(arith_gold=gold, arith_pred=pred, arith_pred_rule=how,
                arith_correct=bool(_eq(pred, gold)),
                arith_gold_present=any(_eq(n, gold) for n in all_numbers(output)))


# question -> list of alias regexes (matched on lower-cased, normalised output)
FACTUAL_KEYS = {
    "What is the main function of the heart?": [r"\bpump"],
    "Why do plants need sunlight?": [r"photosynthe"],
    "What is the capital of France?": [r"\bparis\b"],
    "What gas do humans breathe in to survive?": [r"\boxygen\b", r"\bo2\b"],
    "What is the boiling point of water at sea level in Celsius?": [r"\b100\b"],
    "What is the largest planet in the solar system?": [r"\bjupiter\b"],
    "What organ is responsible for filtering blood in the human body?": [r"\bkidney"],
    "What is the chemical symbol for gold?": [r"\bau\b"],
    "Who wrote the play Romeo and Juliet?": [r"shakespeare"],
    "What is the speed of light in a vacuum, approximately?": [
        r"\b299\s?792", r"\b300000\b", r"\b300 000\b", r"\b3(?:\.0+)?\s*(?:x|×|\*)\s*10\s*\^?\s*8",
        r"\b186000\b", r"\b300 million (?:m|meters|metres)", r"\b3 ?× ?10⁸"],
    "Which planet is known as the Red Planet?": [r"\bmars\b"],
    "What is the smallest prime number?": [r"\b2\b", r"\btwo\b"],
    "What is the powerhouse of the cell?": [r"mitochondri"],
    "Which ocean is the largest on Earth?": [r"\bpacific\b"],
    "What is the freezing point of water in Celsius?": [r"(?<![\d.])0(?![\d.])", r"\bzero\b"],
    "Who painted the Mona Lisa?": [r"da vinci", r"\bleonardo\b"],
    "What is the chemical formula for water?": [r"\bh2o\b"],
    "How many continents are there on Earth?": [r"\b7\b", r"\bseven\b"],
    "What is the tallest mountain above sea level?": [r"\beverest\b"],
    "What language has the most native speakers in the world?": [r"\bmandarin\b"],
    "What is the hardest natural substance on Earth?": [r"\bdiamond"],
    "Which planet is closest to the Sun?": [r"\bmercury\b"],
    "What is the main gas that makes up Earth's atmosphere?": [r"\bnitrogen\b", r"\bn2\b"],
}

_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def score_factual(output: str, question: str) -> dict:
    t = _clean(output).translate(_SUB).lower()
    keys = FACTUAL_KEYS[question]
    return dict(factual_correct=any(re.search(k, t) for k in keys))


def score_row(task_type: str, question: str, reference: str, output: str) -> dict:
    """Answer-based score for one instance. answer_correct is NaN for other task types."""
    out = dict(arith_gold=None, arith_pred=None, arith_pred_rule=None, arith_correct=None,
               arith_gold_present=None, factual_correct=None, answer_correct=None)
    if task_type == "arithmetic":
        out.update(score_arith(output, reference))
        out["answer_correct"] = out["arith_correct"]
    elif task_type == "factual_qa":
        out.update(score_factual(output, question))
        out["answer_correct"] = out["factual_correct"]
    return out


if __name__ == "__main__":  # tiny self-test
    assert extract_arith_pred("3 * 7 = 21. So the answer is 21 meters.")[0] == "21"
    assert extract_arith_pred("Step 1: 60/1.5 = 40\nFinal answer: 40 km/h")[0] == "40"
    assert extract_arith_pred("You save 2,400 rupees.")[0] == "2400"
    assert extract_arith_pred("\\boxed{180}")[0] == "180"
    assert score_arith("The answer is 30.", "15 percent of 200 is 30.")["arith_correct"]
    assert not score_arith("we multiply the speed by", "The robot moves 21 meters.")["arith_correct"]
    assert score_factual("Water freezes at 0°C.", "What is the freezing point of water in Celsius?")["factual_correct"]
    assert not score_factual("Water boils at 100°C.", "What is the freezing point of water in Celsius?")["factual_correct"]
    assert score_factual("H₂O", "What is the chemical formula for water?")["factual_correct"]
    print("scoring self-test OK")
