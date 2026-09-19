# RS-2026-002 Design

The trainer keeps a device-resident sticky status across forward, loss,
backward, topology, and optimizer. A failed step gates subsequent device work;
the host reads status only at declared safety points. Workspace ownership is
explicit and must not use thread-local raw pointers. Reports distinguish GPU
completion timing from CPU submission timing and carry adapter, split, and
binary identity when available.

The change is staged as P0 correctness, P1 ownership and measurement, and P2
quality/reproducibility. Existing P0 and P1.1 evidence in `tasks.md` is the
baseline; future work starts at the first unchecked item.
