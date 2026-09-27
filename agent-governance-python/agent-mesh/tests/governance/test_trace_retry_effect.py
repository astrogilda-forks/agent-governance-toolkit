# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""A retried tool call: what the Trust Record says, and what only effect evidence can.

The case: a tool completes a write, its response is lost, and the agent retries.
AGT's audit trail records what the agent attempted, so the TRACE Trust Record for
that session carries two tool invocations and one error. It cannot say how many
writes landed, or whether the timed-out attempt failed. Those are questions about
the external effect, and they are answered by a record from an observer below the
agent.

The effect records come from the ``agent-evidence-vectors`` observed-effect corpus,
five members built for exactly this case, and are judged by that package's
reference reader. The tests pin two things:

* one AGT session maps to one Trust Record whichever way the effect went, so the
  Trust Record alone must not be read as the effect count;
* the effect evidence separates the three outcomes (one write, a duplicate write,
  not yet witnessed) and refuses the two records that would hide them.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from importlib import resources

import pytest

from agentmesh.governance.audit import AuditEntry
from agentmesh.governance.trace_model import (
    TraceModelConfig,
    TraceSession,
    session_to_trust_record,
)

observedeffect = pytest.importorskip(
    "agent_evidence_vectors.observedeffect",
    reason="agent-evidence-vectors needs Python 3.13 or later",
)

_ZEROS = "0" * 64
_TARGET = "/srv/app/orders/ord-0017.json"

_CONFIG = TraceModelConfig(
    model={
        "provider": "example",
        "model_id": "example-model",
        "version": "1.0",
        "weights_digest": f"sha256:{_ZEROS}",
    },
    runtime={"platform": "software-only", "measurement": f"sha256:{_ZEROS}"},
    enforcement_mode="enforce",
    build_provenance={
        "slsa_level": 2,
        "builder": "github-actions",
        "digest": f"sha256:{_ZEROS}",
    },
    verifier="https://verifier.agentrust-io.com",
)


def _corpus_dir():
    return (
        resources.files("agent_evidence_vectors")
        / "corpora"
        / "vectors-observed-effect"
    )


def _manifest() -> dict:
    return json.loads((_corpus_dir() / "MANIFEST.json").read_text(encoding="utf-8"))


def _member(slug: str) -> tuple[dict, bytes]:
    for entry in _manifest()["vectors"]:
        if entry["slug"] == slug:
            return entry, (_corpus_dir() / entry["file"]).read_bytes()
    raise AssertionError(f"observed-effect corpus has no member {slug!r}")


def _judge(slug: str):
    manifest = _manifest()
    policy = observedeffect.Policy(
        predicate_type=manifest["predicateType"],
        observer_public_key=manifest["keys"]["observer"]["publicKey"],
    )
    _, raw = _member(slug)
    return observedeffect.verify(raw, policy)


def _writes(slug: str) -> list[dict]:
    _, raw = _member(slug)
    payload = base64.b64decode(json.loads(raw)["payload"])
    return json.loads(payload)["predicate"]["writes"]


def _retried_session() -> TraceSession:
    """Two attempts at one operation; the first response was lost."""
    t0 = datetime(2026, 9, 19, 0, 0, 1, tzinfo=UTC)
    t1 = datetime(2026, 9, 19, 0, 0, 3, tzinfo=UTC)
    common = {
        "event_type": "tool_invocation",
        "agent_did": "did:mesh:orders-agent",
        "action": "write_order",
        "resource": _TARGET,
    }
    return TraceSession(
        agent_did="did:mesh:orders-agent",
        audit_entries=[
            AuditEntry(
                **common,
                entry_id="audit_op0017_attempt1",
                timestamp=t0,
                outcome="error",
                data={"operation_id": "op-0017", "attempt": 1, "error": "timeout"},
            ),
            AuditEntry(
                **common,
                entry_id="audit_op0017_attempt2",
                timestamp=t1,
                outcome="success",
                data={"operation_id": "op-0017", "attempt": 2},
            ),
        ],
        data_class="internal",
    )


def test_trust_record_counts_attempts_not_effects():
    record = session_to_trust_record(_retried_session(), _CONFIG)

    # Two attempts, and nothing in the record says how many writes landed.
    assert record["tool_transcript"]["call_count"] == 2
    assert record["appraisal"]["status"] == "affirming"

    # The same session is consistent with one write and with two: the record is a
    # function of the session, and the session is identical in both worlds.
    assert session_to_trust_record(_retried_session(), _CONFIG) == record
    assert len(_writes("retry-one-write-across-two-attempts")) == 1
    assert len(_writes("retry-duplicated-the-write")) == 2


def test_one_write_across_two_attempts_is_established():
    report = _judge("retry-one-write-across-two-attempts")
    assert report.verdict == "valid"
    assert report.effects_independently_observed
    # Authoritative: nothing else landed in scope, so the operation completed once
    # and the timed-out attempt was not a failed operation.
    assert report.absence_established
    assert [w["path"] for w in _writes("retry-one-write-across-two-attempts")] == [
        _TARGET
    ]


def test_duplicate_write_is_visible_and_cannot_be_folded_into_one():
    report = _judge("retry-duplicated-the-write")
    assert report.verdict == "valid"
    assert report.absence_established
    assert [w["path"] for w in _writes("retry-duplicated-the-write")] == [
        _TARGET,
        _TARGET,
    ]

    folded = _judge("retry-duplicate-reported-as-one")
    assert folded.verdict == "malformed"
    assert folded.codes == ["dual-value-not-recomputable"]


def test_outcome_is_unknown_until_the_witness_arrives():
    pending = _judge("retry-effect-not-yet-witnessed")
    assert pending.verdict == "valid"
    # The observer was blind where the write would land: no write is shown, and
    # no absence is established either. Unknown, not failed.
    assert not pending.absence_established
    assert _writes("retry-effect-not-yet-witnessed") == []

    promoted = _judge("retry-timeout-read-as-no-write")
    assert promoted.verdict == "invalid"
    assert promoted.codes == ["authoritative-coverage-incomplete"]


@pytest.mark.parametrize(
    "slug",
    [
        "retry-one-write-across-two-attempts",
        "retry-duplicated-the-write",
        "retry-duplicate-reported-as-one",
        "retry-effect-not-yet-witnessed",
        "retry-timeout-read-as-no-write",
    ],
)
def test_reader_matches_the_published_expectation(slug):
    entry, _ = _member(slug)
    report = _judge(slug)
    assert report.verdict == entry["expected"]["verdict"]
    assert report.codes == entry["expected"]["codes"]
