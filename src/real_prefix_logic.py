import re
from decimal import Decimal


def classify_prefix(prefix: str) -> str:
    
    if prefix == "":
        return "start"

    if prefix == "-":
        return "sign"

    if re.fullmatch(r"-?\d+", prefix):
        return "integer"

    if re.fullmatch(r"-?\d+\.", prefix):
        return "dot"

    if re.fullmatch(r"-?\d+\.\d+", prefix):
        return "fraction"

    raise ValueError(f"Invalid prefix: {prefix!r}")


def valid_next_tokens(prefix: str, decimals: int, allow_negative: bool = True) -> list[str]:

    kind = classify_prefix(prefix)
    digits = [str(i) for i in range(10)]

    if kind == "start":
        return (["-"] if allow_negative else []) + digits

    if kind == "sign":
        return digits

    if kind == "integer":

        if prefix in {"0", "-0"}:
            return ["."]
        return digits + ["."]

    if kind == "dot":
        return digits

    if kind == "fraction":
        frac_len = len(prefix.split(".", 1)[1])
        if frac_len < decimals:
            return digits
        return []

    return []


def ordered_interval(x: float, y: float) -> tuple[float, float]:
    return min(x, y), max(x, y)


def successor_value(prefix: str) -> float:

    if "." in prefix:
        int_part, frac_part = prefix.split(".")
        places = len(frac_part)
        step = Decimal(10) ** Decimal(-places)

        x = Decimal(prefix)
        if prefix.startswith("-"):
            return float(x - step)
        return float(x + step)

    x = int(prefix)
    if prefix.startswith("-"):
        return float(x - 1)
    return float(x + 1)

