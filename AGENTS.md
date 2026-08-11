# Project Notes

## Product Specifications

Before every task in this repository, use the `$specs-author` skill to read the entire root `SPECS.md`. Before finishing, reread it and check the task and conversation for new or changed stakeholder intent.

- Treat `SPECS.md` as the persistent source of stakeholder requirements that cannot be inferred reliably from code or remembered conversations.
- Apply the scope test to proposed and existing requirements: root `SPECS.md` contains only project-wide intent; scoped intent belongs in its nearest authoritative specification and must not be broadened to fit the root.
- If the task, repository, or user request contradicts, omits, or ambiguously interprets the specification, tell the user. Continue safe exploration and work that does not depend on resolving the issue, but never silently choose an interpretation.
- Never edit `SPECS.md` from inference. Propose the exact change, explain why it reflects stakeholder intent, and edit the file only after the user explicitly approves that exact change.
- Keep `SPECS.md` complete, concise, and compacted. It must contain stakeholder intent rather than implementation, architecture, operations, or transient project detail.

## Mario State Generation

Use the repo-local `generate-mario-states` skill when creating, validating, or deploying `SuperMarioBros-Nes-v0` `Level*.state` files.

Key guardrails:

- Do not overwrite existing Mario state files unless explicitly requested.
- Validate generated screenshots against NESMaps before treating a state as correct.
- When checking a consumer repo's installed package, run Python from that consumer repo's cwd so this checkout does not shadow the installed wheel.
