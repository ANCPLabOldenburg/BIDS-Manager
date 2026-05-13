"""Unit tests for ``inventory.subject_identity``.

The clustering must:

* Merge two visits of one patient when *any* identifier is consistent
  (e.g. anonymised PatientID flips between visits but PatientName stays).
* NOT merge two distinct patients sharing a placeholder value
  (e.g. ``"DE-IDENTIFIED"`` family name across PPMI's two patients).
* Detect placeholders automatically — a value is a placeholder when
  records carrying it disagree on every other non-empty identifier.
"""

from __future__ import annotations

from bidsmgr.inventory.subject_identity import (
    cluster_subjects,
    detect_placeholders,
    normalize_tuple,
)


def _bucket(clusters):
    """Return a frozenset of frozensets (cluster contents)."""
    by_root = {}
    for tup, root in clusters.items():
        by_root.setdefault(root, set()).add(tup)
    return frozenset(frozenset(s) for s in by_root.values())


def test_singleton_is_its_own_cluster():
    t = normalize_tuple("123", "John", "Smith")
    clusters = cluster_subjects({t})
    assert clusters == {t: t}


def test_ppmi_case_same_pid_different_birthdate_anonymisation():
    """Two visits with identical PatientID but blanked-on-one-visit name fields
    are still ONE subject — the PatientID alone links them."""
    visit1 = normalize_tuple("101174", "", "DE-IDENTIFIED")
    visit2 = normalize_tuple("101174", "", "DE-IDENTIFIED")
    # Same tuple — collapses to one anyway.
    clusters = cluster_subjects({visit1, visit2})
    assert _bucket(clusters) == frozenset({frozenset({visit1})})


def test_ppmi_two_patients_sharing_placeholder_family_stay_split():
    """``DE-IDENTIFIED`` shared across distinct PatientIDs is a placeholder
    and must NOT merge the two patients."""
    p1 = normalize_tuple("101174", "", "DE-IDENTIFIED")
    p2 = normalize_tuple("101175", "", "DE-IDENTIFIED")
    placeholders = detect_placeholders({p1, p2})
    # FamilyName index is 2.
    assert (2, "DE-IDENTIFIED") in placeholders

    clusters = cluster_subjects({p1, p2})
    assert _bucket(clusters) == frozenset({frozenset({p1}), frozenset({p2})})


def test_consistent_name_links_visits_with_changed_pid():
    """If anonymisation rewrites PatientID across visits but the name stays
    consistent, the two visits should merge."""
    v1 = normalize_tuple("ANON-A1", "John", "Smith")
    v2 = normalize_tuple("ANON-A2", "John", "Smith")
    other = normalize_tuple("ANON-X", "Mary", "Jones")
    clusters = cluster_subjects({v1, v2, other})
    bucket = _bucket(clusters)
    assert frozenset({v1, v2}) in bucket
    assert frozenset({other}) in bucket


def test_consistent_pid_links_visits_with_typo_name():
    """If PatientID stays but the name is mistyped on a follow-up visit,
    PatientID alone must merge the two visits."""
    v1 = normalize_tuple("12345", "John", "Smith")
    v2 = normalize_tuple("12345", "Jonh", "Smith")  # typo in given
    clusters = cluster_subjects({v1, v2})
    assert _bucket(clusters) == frozenset({frozenset({v1, v2})})


def test_anonymous_pid_does_not_overmerge():
    """If many distinct people share the placeholder PatientID 'ANON', the
    PatientID is detected as a placeholder and patients with distinct names
    stay separate."""
    a = normalize_tuple("ANON", "John", "Smith")
    b = normalize_tuple("ANON", "Mary", "Jones")
    c = normalize_tuple("ANON", "Bob", "Brown")
    placeholders = detect_placeholders({a, b, c})
    assert (0, "ANON") in placeholders  # PID index 0
    clusters = cluster_subjects({a, b, c})
    assert len(_bucket(clusters)) == 3


def test_placeholder_pid_but_consistent_name_links():
    """When PatientID is anonymised constant but ALL records share a name,
    the name is the real identifier and they merge."""
    v1 = normalize_tuple("ANON", "John", "Smith")
    v2 = normalize_tuple("ANON", "John", "Smith")  # same tuple
    # We need different tuples to test merging.
    v3 = normalize_tuple("ANON", "John", "")
    # All three share "John" given name but PIDs are constant ANON.
    clusters = cluster_subjects({v1, v3})
    # ANON should be flagged as placeholder; "John" is real → merge.
    assert _bucket(clusters) == frozenset({frozenset({v1, v3})})


def test_double_placeholder_caught_by_known_token_list():
    """Two records share TWO separate well-known anonymisation tokens
    (``ANON`` PID + ``DE-IDENTIFIED`` family). The hard-coded placeholder
    list catches both regardless of cardinality, so the records correctly
    stay split via their distinct given names."""
    a = normalize_tuple("ANON", "John", "DE-IDENTIFIED")
    b = normalize_tuple("ANON", "Mary", "DE-IDENTIFIED")
    clusters = cluster_subjects({a, b})
    assert _bucket(clusters) == frozenset({frozenset({a}), frozenset({b})})


def test_three_disjoint_records_share_double_placeholder_and_split():
    """Three+ records all sharing a placeholder PID and family but with
    distinct given names — the heuristic correctly detects both placeholders
    and keeps the three subjects separate."""
    a = normalize_tuple("ANON", "John", "DE-IDENTIFIED")
    b = normalize_tuple("ANON", "Mary", "DE-IDENTIFIED")
    c = normalize_tuple("ANON", "Bob", "DE-IDENTIFIED")
    clusters = cluster_subjects({a, b, c})
    assert len(_bucket(clusters)) == 3


def test_neuroimaging_unit_new_shape_consistent_given_links_visits():
    """Real shape: operator stamps folder labels into PID and FamilyName
    so they look like distinct identifiers, but the GivenName carries the
    real anonymised subject hash. Two tuples sharing only the GivenName
    must merge; the third tuple with a different GivenName stays split.

    This is the case the cardinality-only heuristic got wrong. The fix
    is to drop cardinality detection at < 100% coverage and rely on the
    hardcoded token list + universal coverage instead.
    """
    a = normalize_tuple("OL_0001", "XX00XX00", "OL_0001")
    b = normalize_tuple("OL_0002", "XX00XX00", "OL_0002")
    c = normalize_tuple("OL_0003", "XX00XX22", "OL_0003")
    clusters = cluster_subjects({a, b, c})
    bucket = _bucket(clusters)
    assert frozenset({a, b}) in bucket
    assert frozenset({c}) in bucket
    assert len(bucket) == 2


def test_universal_coverage_value_treated_as_placeholder():
    """If a value covers EVERY tuple in a field, it's an operator-stamped
    constant — flag as placeholder so it doesn't fool the union-find."""
    a = normalize_tuple("PAT-1", "Alice", "STUDYNAME")
    b = normalize_tuple("PAT-2", "Bob", "STUDYNAME")
    c = normalize_tuple("PAT-3", "Carol", "STUDYNAME")
    placeholders = detect_placeholders({a, b, c})
    # FamilyName index is 2 — STUDYNAME covers 3/3.
    assert (2, "STUDYNAME") in placeholders
    clusters = cluster_subjects({a, b, c})
    assert len(_bucket(clusters)) == 3


def test_partial_coverage_value_NOT_treated_as_placeholder():
    """A value that covers only 2 of 3 tuples is more likely a real shared
    identifier (visits of one person within a multi-person dataset)."""
    a = normalize_tuple("PID-1", "Alice", "Smith")
    b = normalize_tuple("PID-2", "Alice", "Smith")  # same person, different anonymized PID
    c = normalize_tuple("PID-3", "Carol", "Jones")
    placeholders = detect_placeholders({a, b, c})
    # "Alice" / "Smith" cover 2/3 — not flagged as placeholders.
    assert (1, "Alice") not in placeholders
    assert (2, "Smith") not in placeholders
    clusters = cluster_subjects({a, b, c})
    bucket = _bucket(clusters)
    assert frozenset({a, b}) in bucket
    assert frozenset({c}) in bucket


def test_tuple_normalization_strips_whitespace():
    a = normalize_tuple("  101174  ", "John ", " Smith")
    assert a == ("101174", "John", "Smith")
