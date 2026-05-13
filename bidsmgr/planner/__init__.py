"""``Classification + user edits -> EntityPlan``.

Reference: architecture.md §5.

The only place user agency lives. Pure functional state machine
``(state, edit) -> state``, which makes event sourcing
(``bidsmgr.project``) free.

Refuses to produce a plan that the schema rejects.

Stub — not yet implemented.
"""
