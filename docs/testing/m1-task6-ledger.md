# M1 task 6 execution ledger

Base: PR #8 was reviewed and merged as 18f462a before this task began.
Task: real C ABI conformance and isolated native builds, per the approved M1 plan.

RED starts with executable layout/export gate tests against incomplete gates.
The initial C command rejects rather than treating missing consumers as success.
The frozen header is an exact copy of the development header blob
405989ed7024c32fa904e02662b35863ff7d9f13, not an already released ABI guarantee.

Planned acceptance: Rust-produced target layouts versus C compiler layouts for
all 10 public records and 75 fields, constants, all 11 public entry points,
controlled BUSY from native threads, frozen-header regression, exact symbols,
normal/fixture isolation, dynamic C on Linux/Windows and static C on Linux.
No runtime architecture, library license or trained-model behavior changes.

Local authoring has no Rust toolchain/network; use isolated remote Git trees,
a leased feature branch and actual GitHub Actions results. No local compilation
or Windows execution is claimed until corresponding commands run successfully.
No security suppression, package publication or extra runtime dependency.
AI-assisted; no contributor certification asserted.
