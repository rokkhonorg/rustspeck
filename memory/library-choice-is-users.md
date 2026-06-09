---
name: library-choice-is-users
description: The user chooses and adds libraries; implement against what they added, don't assume
metadata:
  type: feedback
---

The user picks the external library and adds it themselves (`cargo add`); my job is to implement against whatever they added — NOT to pre-decide the stack, recommend one, and start coding to it.

Concretely: for the GUI I assumed eframe/egui and wrote a whole egui implementation + told them to `cargo add eframe`. They had added **floem**. They were (rightly) angry. This pattern held for prior steps too — they said "I added rustfft" / "I added symphonia" and I implemented to those.

**Why:** They control the dependency surface and the architecture; assuming a library wastes effort and imposes choices they didn't make. See [[deps-and-edition]].

**How to apply:** When a task needs an external crate, ask which library they want (or check Cargo.toml for what they've already added) BEFORE writing code against any specific one. If they haven't added it yet, wait or ask — do not pick one and run. Then read the *installed* version's API from the registry source before coding (these crates drift between versions).
