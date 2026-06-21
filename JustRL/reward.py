"""Binary DAPO-style rule-based reward for JustRL. No SymPy."""
import re
from typing import Optional


def _last_boxed(text: str) -> Optional[str]:
    """Return the last '\\boxed{...}' / '\\fbox{...}' substring, or None."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    i, opens, close = idx, 0, None
    while i < len(text):
        if text[i] == "{":
            opens += 1
        if text[i] == "}":
            opens -= 1
            if opens == 0:
                close = i
                break
        i += 1
    return text[idx:close + 1] if close is not None else None


def _remove_boxed(s: str) -> Optional[str]:
    """Strip the '\\boxed{' prefix and trailing '}'."""
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left and s[-1] == "}"
        return s[len(left):-1]
    except Exception:
        return None


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the content of the LAST \\boxed{} in `text`."""
    boxed = _last_boxed(text)
    if boxed is None:
        return None
    return _remove_boxed(boxed)


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    parts = string.split("\\sqrt")
    out = parts[0]
    for p in parts[1:]:
        out += "\\sqrt" + (p if p[:1] == "{" else "{" + p[0] + "}" + p[1:])
    return out


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr[:1] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    new_str += "{" + a + "}{" + b + "}" + substr[2:]
                else:
                    new_str += "{" + a + "}" + b + substr[2:]
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a_int, b_int = int(a), int(b)
        if string == "{}/{}".format(a_int, b_int):
            return "\\frac{" + str(a_int) + "}{" + str(b_int) + "}"
        return string
    except Exception:
        return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


_UNIT_WORDS = [
    "degree", "cm", "centimeter", "meter", "mile", "second", "minute",
    "hour", "day", "week", "month", "year", "foot", "feet", "inch", "yard",
]


def _strip_unit_words(string: str) -> str:
    for unit in _UNIT_WORDS:
        string = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", string)
    return string


def _strip_properly_formatted_commas(expr: str) -> str:
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        nxt = p1.sub(r"\1\3\4", expr)
        if nxt == expr:
            return nxt
        expr = nxt


def _str_is_int(x: str) -> bool:
    try:
        v = float(_strip_properly_formatted_commas(x))
        return abs(v - int(round(v))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    return int(float(x.replace(",", "")))


def _strip_string(string: str) -> str:
    s = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "")
    s = _remove_right_units(s)
    s = _strip_unit_words(s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s and s[0] == ".":
        s = "0" + s
    if len(s.split("=")) == 2 and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    s = _fix_sqrt(s).replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = _fix_a_slash_b(s)
    if _str_is_int(s):
        s = str(_str_to_int(s))
    return s


def mathd_normalize(answer: Optional[str]) -> Optional[str]:
    """Normalize a math answer for string comparison (Hendrycks-style)."""
    if answer is None:
        return None
    answer = answer.strip()
    m = re.search(r"^\\text\{(?P<t>.+?)\}$", answer)
    if m is not None:
        answer = m.group("t").strip()
    try:
        return _strip_string(answer)
    except Exception:
        return answer


def _to_float(s: str) -> Optional[float]:
    try:
        return float(_strip_properly_formatted_commas(s).strip())
    except Exception:
        return None


_TUPLE_CHARS = "()[]"


def _split_tuple(expr: str) -> list:
    """Split a tuple/interval '(1, 2)' / '[3, 4]' into elements."""
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in _TUPLE_CHARS
        and expr[-1] in _TUPLE_CHARS
        and all(ch not in expr[1:-1] for ch in _TUPLE_CHARS)
    ):
        return [e.strip() for e in expr[1:-1].split(",")]
    return [expr]


def grade_answer(
    given: Optional[str],
    truth: Optional[str],
    rel_tol: float = 1e-3,
) -> bool:
    """Return True iff `given` matches `truth` after normalization.

    Checks: exact normalized equality, numeric equality within rel_tol,
    and element-wise tuple/interval comparison."""
    if given is None or truth is None or not given or not truth:
        return False
    g = mathd_normalize(given)
    t = mathd_normalize(truth)
    if g is None or t is None:
        return False
    if g == t:
        return True
    g_elems, t_elems = _split_tuple(g), _split_tuple(t)
    if len(g_elems) > 1 or len(t_elems) > 1:
        if len(g_elems) != len(t_elems):
            return False
        return all(grade_answer(a, b, rel_tol) for a, b in zip(g_elems, t_elems))
    gf, tf = _to_float(g), _to_float(t)
    if gf is not None and tf is not None:
        return abs(gf - tf) <= rel_tol * max(1.0, abs(tf))
    return False


def dapo_binary_reward(prompts, completions, answer, **kwargs) -> list:
    """Return 1.0 for a correct \\boxed{} answer, else 0.0."""
    rewards = []
    for comp, gt in zip(completions, answer):
        text = comp[0]["content"] if comp else ""
        gt_clean = extract_boxed_answer(gt) if "\\boxed" in (gt or "") else gt
        given = extract_boxed_answer(text)
        rewards.append(1.0 if grade_answer(given, gt_clean) else 0.0)
    return rewards


if __name__ == "__main__":
    cases = [
        ("The answer is \\boxed{42}.", "42", True),
        ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}", True),
        ("\\boxed{0.5}", "\\frac{1}{2}", True),
        ("\\boxed{2.500}", "2.5", True),
        ("\\boxed{100}", "101", False),
        ("no box here", "42", False),
        ("\\boxed{\\frac{1}{\\sqrt{2}}}", "\\frac{\\sqrt{2}}{2}", False),
        ("\\boxed{\\frac1{72}}", "\\frac{1}{72}", True),
        ("\\boxed{12\\text{ cm}}", "12", True),
        ("\\boxed{01/02}", "01/02", True),
        ("\\boxed{5 meters}", "5", True),
        ("\\boxed{2.0}", "2", True),
        ("\\boxed{(1, 2)}", "[1, 2]", True),
        ("\\boxed{(1, 2)}", "(1, 3)", False),
        ("\\boxed{(1,2,3)}", "(1,2)", False),
        ("\\boxed{1,234}", "1234", True),
    ]
    for resp, gt, expect in cases:
        got = dapo_binary_reward([None], [[{"content": resp}]], [gt])[0]
        flag = "OK " if got == float(expect) else "FAIL"
        print(f"{flag} resp={resp!r:45} gt={gt!r:20} -> {got} (expect {expect})")
