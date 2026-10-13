# Contributing

Thanks for helping. Read [AGENTS.md](AGENTS.md) first: it holds the rules every change follows,
whether a person or an agent writes it.

1. Branch from `main`, keep a pull request to one topic.
2. Run `cargo xtask ci` before pushing; CI runs the same gates.
3. User-visible behaviour is a command (`crates/engine/src/cmd/`) with tests, including hostile
   parameters.
4. Meshing and solver algorithms cite the paper they come from in the module docs.
5. Never contribute code, text or data from GPL-family projects or commercial CAE products
   (see [docs/adr/0001-libraries.md](docs/adr/0001-libraries.md)).

By contributing you agree your work is licensed under MIT OR Apache-2.0.
