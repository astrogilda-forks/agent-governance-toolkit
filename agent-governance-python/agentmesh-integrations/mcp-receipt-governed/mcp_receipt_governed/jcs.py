# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""RFC 8785 JSON Canonicalization Scheme (JCS) for receipt payloads.

``json.dumps(sort_keys=True, separators=(",", ":"))`` is not RFC 8785. It
differs in three places that change the bytes a receipt signs:

* numbers: Python writes ``1.0`` and ``1e-07`` where JCS requires ``1`` and
  ``1e-7`` (the ECMAScript ``Number.prototype.toString`` form, RFC 8785
  section 3.2.2.3);
* member order: Python sorts keys by code point, JCS by UTF-16 code unit
  (section 3.2.3), which differs once a key holds a character outside the
  Basic Multilingual Plane;
* out-of-range values: Python writes ``NaN``, ``Infinity`` and integers of
  any size, none of which have an RFC 8785 form.

A verifier written in another language that follows RFC 8785 computes
different bytes for any of those inputs, so the signature and the chain hash
stop reproducing across implementations. This module produces the RFC 8785
bytes and refuses a value that has none.
"""

from __future__ import annotations

import json
import math
from typing import Any

__all__ = ["canonicalize"]

# RFC 8785 section 3.2.2.3 and I-JSON (RFC 7493) section 2.2: integers beyond
# this magnitude cannot round-trip through an IEEE 754 double, so two distinct
# integers would share one canonical form.
_INT_LIMIT = 2**53 - 1


def canonicalize(value: Any) -> str:
    """Return the RFC 8785 canonical JSON text of ``value``.

    Raises:
        ValueError: ``value`` holds a non-finite float, an integer outside
            +/-(2**53 - 1), or a string that is not valid Unicode.
        TypeError: ``value`` holds a type JSON cannot represent, or an object
            key that is not a string.
    """
    parts: list[str] = []
    _write(value, parts)
    return "".join(parts)


def _write(value: Any, out: list[str]) -> None:
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, int):
        if not -_INT_LIMIT <= value <= _INT_LIMIT:
            raise ValueError(f"integer {value} is outside the RFC 8785 range of +/-(2**53 - 1)")
        out.append(str(value))
    elif isinstance(value, float):
        out.append(_number(value))
    elif isinstance(value, str):
        out.append(_string(value))
    elif isinstance(value, (list, tuple)):
        out.append("[")
        for index, item in enumerate(value):
            if index:
                out.append(",")
            _write(item, out)
        out.append("]")
    elif isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise TypeError(f"object key {key!r} is not a string")
        out.append("{")
        for index, key in enumerate(sorted(value, key=_utf16_order)):
            if index:
                out.append(",")
            out.append(_string(key))
            out.append(":")
            _write(value[key], out)
        out.append("}")
    else:
        raise TypeError(f"{type(value).__name__} has no JSON representation")


def _utf16_order(key: str) -> bytes:
    # Big-endian UTF-16 bytes compare in code-unit order (RFC 8785 section 3.2.3).
    return key.encode("utf-16-be", "surrogatepass")


def _string(text: str) -> str:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("string contains a lone surrogate, which has no UTF-8 form") from exc
    # With ensure_ascii=False, json.dumps escapes exactly the set RFC 8785
    # section 3.2.2.2 requires: '"', '\\', the short forms \b \t \n \f \r, and
    # every other control character as a lowercase \u00XX escape.
    return json.dumps(text, ensure_ascii=False)


def _number(number: float) -> str:
    """Serialize a double the way ECMAScript ``Number.prototype.toString`` does."""
    if not math.isfinite(number):
        raise ValueError(f"{number!r} has no JSON representation")
    if number == 0:
        return "0"  # covers -0.0, which RFC 8785 writes as 0
    sign = "-" if number < 0 else ""
    # repr() gives the shortest digit string that round-trips, which is the
    # digit string ECMAScript chooses; only the layout differs.
    mantissa, _, exponent = repr(abs(number)).partition("e")
    whole, _, fraction = mantissa.partition(".")
    significant = (whole + fraction).lstrip("0")
    digits = significant.rstrip("0")
    count = len(digits)
    # ECMAScript writes the value as 0.<digits> x 10**point.
    point = int(exponent or 0) - len(fraction) + len(significant)
    if count <= point <= 21:
        text = digits + "0" * (point - count)
    elif 0 < point <= 21:
        text = digits[:point] + "." + digits[point:]
    elif -6 < point <= 0:
        text = "0." + "0" * -point + digits
    else:
        power = point - 1
        fraction_part = "." + digits[1:] if count > 1 else ""
        text = f"{digits[0]}{fraction_part}e{'+' if power >= 0 else '-'}{abs(power)}"
    return sign + text
