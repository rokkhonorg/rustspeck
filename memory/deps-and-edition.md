---
name: deps-and-edition
description: User's preferences for Rust edition and how external crates get added
metadata:
  type: feedback
---

Keep `edition = "2024"` in Cargo.toml for the rustspek project — do not downgrade it.

Do NOT edit the `[dependencies]` section of Cargo.toml yourself. When a new external crate is needed, tell the user the exact `cargo add` command(s) and let them run it manually.

**Why:** The user wants full control over their dependency surface and adds deps via `cargo add` themselves.
**How to apply:** Write code that uses the crate, but instead of editing Cargo.toml, list the `cargo add ...` commands for the user to run.
