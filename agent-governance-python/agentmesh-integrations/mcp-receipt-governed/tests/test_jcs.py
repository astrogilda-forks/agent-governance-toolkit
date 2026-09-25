# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""RFC 8785 conformance of the receipt canonicalizer.

The number rows are RFC 8785 Appendix B, Table 1. Rows 11, 12 and 15 and the
UTF-16 member-order case are also signed-record conformance vectors in
https://github.com/probityai/agent-evidence-vectors (``vectors-ai-agent-action``
members ``vad17f7bf034c8dde``, ``vb7dd5fb8dcb6e345``, ``v6d21ff18cfa69ef8`` and
``vb1f091f42efd0715``), so an implementation of this receipt format in another
language can run the same cases.
"""

import hashlib
import json
import struct

import pytest

from mcp_receipt_governed.jcs import canonicalize
from mcp_receipt_governed.receipt import GovernanceReceipt, hash_tool_args

APPENDIX_B = [
    ("0000000000000000", "0"),
    ("8000000000000000", "0"),
    ("0000000000000001", "5e-324"),
    ("8000000000000001", "-5e-324"),
    ("7fefffffffffffff", "1.7976931348623157e+308"),
    ("ffefffffffffffff", "-1.7976931348623157e+308"),
    ("4340000000000000", "9007199254740992"),
    ("c340000000000000", "-9007199254740992"),
    ("4430000000000000", "295147905179352830000"),
    ("44b52d02c7e14af5", "9.999999999999997e+22"),
    ("44b52d02c7e14af6", "1e+23"),
    ("44b52d02c7e14af7", "1.0000000000000001e+23"),
    ("444b1ae4d6e2ef4e", "999999999999999700000"),
    ("444b1ae4d6e2ef4f", "999999999999999900000"),
    ("444b1ae4d6e2ef50", "1e+21"),
    ("3eb0c6f7a0b5ed8c", "9.999999999999997e-7"),
    ("3eb0c6f7a0b5ed8d", "0.000001"),
    ("41b3de4355555553", "333333333.3333332"),
    ("41b3de4355555554", "333333333.33333325"),
    ("41b3de4355555555", "333333333.3333333"),
    ("41b3de4355555556", "333333333.3333334"),
    ("41b3de4355555557", "333333333.33333343"),
    ("becbf647612f3696", "-0.0000033333333333333333"),
    ("43143ff3c1cb0959", "1424953923781206.2"),
]


@pytest.mark.parametrize("ieee, expected", APPENDIX_B)
def test_appendix_b_number_serialization(ieee, expected):
    value = struct.unpack(">d", bytes.fromhex(ieee))[0]
    assert canonicalize(value) == expected


@pytest.mark.parametrize("ieee", ["7fffffffffffffff", "7ff0000000000000", "fff0000000000000"])
def test_non_finite_numbers_are_refused(ieee):
    with pytest.raises(ValueError):
        canonicalize(struct.unpack(">d", bytes.fromhex(ieee))[0])


def test_integral_float_has_no_fraction():
    # json.dumps writes 1.0; RFC 8785 writes the double 1.0 as 1.
    assert canonicalize({"temperature": 1.0}) == '{"temperature":1}'


def test_small_exponent_uses_ecmascript_form():
    # json.dumps writes 1e-07; RFC 8785 writes 1e-7.
    assert canonicalize([1e-7]) == "[1e-7]"


def test_members_sort_by_utf16_code_unit():
    # U+1F680 is D83D DE80 in UTF-16, so it sorts before U+FF3A by code unit
    # and after it by code point.
    assert canonicalize({"\uff3a": 1, "\U0001f680": 2}) == '{"\U0001f680":2,"\uff3a":1}'


def test_rfc_8785_section_3_2_3_sorting_example():
    members = {
        "\u20ac": "Euro Sign",
        "\r": "Carriage Return",
        "\ufb33": "Hebrew Letter Dalet With Dagesh",
        "1": "One",
        "\U0001f600": "Emoji: Grinning Face",
        "\u0080": "Control",
        "\u00f6": "Latin Small Letter O With Diaeresis",
    }
    order = [key for key, _ in _members(canonicalize(members))]
    assert order == ["\r", "1", "\u0080", "\u00f6", "\u20ac", "\U0001f600", "\ufb33"]


def test_string_escapes():
    assert (
        canonicalize('\u001f"\\\b\f\n\r\t\u007f\u2028')
        == '"\\u001f\\"\\\\\\b\\f\\n\\r\\t\u007f\u2028"'
    )


def test_integers_outside_the_double_range_are_refused():
    assert canonicalize(2**53 - 1) == "9007199254740991"
    with pytest.raises(ValueError):
        canonicalize(2**53)
    with pytest.raises(ValueError):
        canonicalize(-(2**53))


def test_lone_surrogate_is_refused():
    with pytest.raises(ValueError):
        canonicalize({"a": "\ud800"})


def test_non_string_key_is_refused():
    with pytest.raises(TypeError):
        canonicalize({1: "one"})


def test_tool_args_hash_uses_raw_utf8():
    args = {"path": "/donn\u00e9es/rapport.csv"}
    expected = hashlib.sha256('{"path":"/donn\u00e9es/rapport.csv"}'.encode()).hexdigest()
    assert hash_tool_args(args) == expected


def test_receipt_payload_writes_integral_timestamp_without_fraction():
    payload = GovernanceReceipt(receipt_id="id", timestamp=1758830400.0).canonical_payload()
    assert '"timestamp":1758830400,' in payload


def _members(text):
    """Split a flat canonical object into (key, value) pairs, in order."""
    decoder = json.JSONDecoder(object_pairs_hook=list)
    return decoder.decode(text)
