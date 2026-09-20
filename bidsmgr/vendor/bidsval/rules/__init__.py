"""Apply the schema's rules to a file context and produce findings.

The engine walks the schema's rule groups, evaluates each rule's selectors to
decide whether it applies, and then runs its checks and sidecar-field
requirements through the expression evaluator. The rules come entirely from the
schema, so the validator covers whatever the schema covers.
"""

from __future__ import annotations

from .engine import apply_rules, validate_basename

__all__ = ["apply_rules", "validate_basename"]
