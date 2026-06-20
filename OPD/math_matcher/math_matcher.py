"""
Robust math-answer matcher built on top of `math_verify`.

Why this exists
---------------
Calling `math_verify.parse/verify` naively produces a LOT of false negatives
on real datasets (intervals, sets, tuples, multi-answers, plain-text answers,
`x \\in ...` prefixes, `\\mathbb{R}` vs `(-\\infty,\\infty)`, `\\dfrac` vs
`\\frac`, etc.). This module wraps `math_verify` with:

  1. correct call posture  -> wrap both sides in `$...$` and enable
     Latex/Expr/String extraction configs;
  2. answer-prefix stripping -> drop `x \\in`, `var =` assignment heads;
  3. set equivalence         -> `\\mathbb{R}` <-> `(-\\infty,\\infty)`;
  4. multi-answer handling   -> ground truth "A or B": match any branch;
  5. string fallback         -> normalized literal compare for text answers.

Public API
----------
    extract_boxed(text)              -> str        # last \\boxed{...} content
    match(prediction, ground_truth)  -> bool       # compare two answer strings
    match_solution(text, gt)         -> bool       # extract \\boxed then match

Dependency: pip install math-verify
"""

import re
import warnings

warnings.filterwarnings("ignore")

from math_verify import parse as _mv_parse, verify as _mv_verify
from math_verify import (
    LatexExtractionConfig as _LCfg,
    ExprExtractionConfig as _ECfg,
    StringExtractionConfig as _SCfg,
)

# Enable all three extraction styles so tuples/sets/strings parse correctly.
_MV_CFG = [_LCfg(), _ECfg(), _SCfg()]


# --------------------------------------------------------------------------- #
# answer extraction / normalization
# --------------------------------------------------------------------------- #
def extract_boxed(text: str) -> str:
    """Return the content of the LAST \\boxed{...} in `text` (brace-balanced)."""
    text = text or ""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return ""
    i = text.find("{", idx)
    if i == -1:
        return ""
    depth = 0
    out = []
    for ch in text[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out).strip()


def norm_ans(s: str) -> str:
    """Aggressive literal normalization, used as a last-resort string compare."""
    s = (s or "").strip()
    for a in ("\\(", "\\)", "\\[", "\\]", "$$", "$"):
        s = s.replace(a, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    # strip a leading "var=" or "(vars)=" assignment so the bare value remains
    s = re.sub(r"^[a-zA-Z]\w*=", "", s)
    s = re.sub(r"^\([a-zA-Z,]+\)=", "", s)
    return s


def _strip_answer_prefix(s: str) -> str:
    """Drop math delimiters and a leading 'x \\in' / 'var =' assignment prefix."""
    s = (s or "").strip()
    for a in ("\\(", "\\)", "\\[", "\\]"):
        s = s.replace(a, "")
    s = s.strip()
    # 'x \in <set>'  ->  '<set>'
    s = re.sub(r"^\s*[\(\)a-zA-Z,\\\s]+\\in\s*", "", s)
    # 'a = ...' single-var assignment prefix
    s = re.sub(r"^\s*[a-zA-Z]\w*\s*=\s*(?![=<>])", "", s)
    # R  <->  (-inf, inf)
    if re.fullmatch(r"\\mathbb\{R\}|\\mathbb R", s.strip()):
        s = "(-\\infty,\\infty)"
    return s.strip()


# --------------------------------------------------------------------------- #
# core symbolic compare
# --------------------------------------------------------------------------- #
def _mv_eq(a: str, b: str) -> bool:
    """Symbolic equality via math_verify, trying raw and $-wrapped forms."""
    for x, y in ((a, b), ("$" + a + "$", "$" + b + "$")):
        try:
            ga = _mv_parse(x, extraction_config=_MV_CFG)
            gb = _mv_parse(y, extraction_config=_MV_CFG)
            if ga and gb and (_mv_verify(ga, gb) or _mv_verify(gb, ga)):
                return True
        except Exception:
            pass
    return False


# --------------------------------------------------------------------------- #
# public matchers
# --------------------------------------------------------------------------- #
def match(prediction: str, ground_truth: str) -> bool:
    """Return True if `prediction` is mathematically equivalent to `ground_truth`.

    Both arguments are bare answer strings (NOT full solutions). Use
    `match_solution` if you have a full CoT containing \\boxed{...}.
    """
    pred = (prediction or "").strip()
    gt = (ground_truth or "").strip()
    if not pred or not gt:
        return False
    # 1) direct symbolic compare
    if _mv_eq(gt, pred):
        return True
    # 2) compare after stripping 'x \in' / 'var =' prefixes (interval/set answers)
    p2, t2 = _strip_answer_prefix(pred), _strip_answer_prefix(gt)
    if _mv_eq(t2, p2):
        return True
    # 3) multi-answer ground truth ("A or B"): match any branch
    for part in re.split(r"\\?\bor\b|，|,or,", gt):
        if part.strip() and _mv_eq(_strip_answer_prefix(part), p2):
            return True
    # 4) normalized string fallback (pure-text answers: words, 'No', units...)
    if norm_ans(gt) and norm_ans(gt) == norm_ans(pred):
        return True
    return False


def match_solution(solution_text: str, ground_truth: str) -> bool:
    """Extract the last \\boxed{...} from a full solution, then `match`."""
    pred = extract_boxed(solution_text)
    if not pred:
        return False
    return match(pred, ground_truth)


if __name__ == "__main__":
    # quick self-test
    cases = [
        ("[0,2)", "x \\in [0,2)", True),
        ("\\mathbb{R}", "(-\\infty,\\infty)", True),
        ("\\dfrac{1}{2}", "\\frac{1}{2}", True),
        ("166464", "166464 or 646416", True),
        ("Saturday", "Saturday", True),
        ("(2,8)", "(2, 8)", True),
        ("2", "4", False),
    ]
    ok = 0
    for p, g, exp in cases:
        got = match(p, g)
        flag = "OK " if got == exp else "XX "
        ok += got == exp
        print(f"{flag} match({p!r}, {g!r}) = {got}  (expected {exp})")
    print(f"\n{ok}/{len(cases)} self-tests passed")
