# Fork PR #220 port to cand-6

The port starts from `07665c3` and uses venglov’s changes as its source. [Fork PR #220](https://github.com/fxceless/victory-fork/pull/220)

## Indirect budget

- Set `VICTORY_INDIRECT_GROUP_BUDGET=1` to enable grouped charging and the device-local list budget. The switch defaults to off because of the device losses reported on one machine. The default uses the existing per-record clamp and CPU-mapped list budget. The disabled clamp compiles to byte-for-byte identical DXBC with diagnostics both off and on.
- The opt-in shader reduces costs in group-shared memory with two full-group barriers, so the 64-thread group can span either one 64-lane wave or two 32-lane waves. One lane reserves the group's charge with an atomic read and one successful compare-and-swap, retrying on contention. Costs are applied in lane order within each group, preserving saturation and the existing bounds. Groups beyond the effective count leave the budget untouched.
- With immediate-write support on direct command lists, the list zeroes its device-local budget using `WriteBufferImmediate` before its first clamp. Diagnostics builds copy back the two mesh and two compute counters at list close. The next submission reads those counters as before. Adapters without immediate-write support, or lists whose device-local allocation fails, use mapped memory with grouped charging. [D3D12 immediate-write requirements](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist2-writebufferimmediate)

## Window checking and scratch barriers

- `VICTORY_INDIRECT_WINDOW_CHECK=1` enables fresh comparisons on hits with `VICTORY_INDIRECT_WINDOW=1`. The check clamps into private scratch with a separate, explicitly initialized mapped budget. It records the first differing record and counts comparisons for logging at the next checked submission of the list. A list-budget difference can also produce a mismatch because the fresh clamp has its own budget. The comparison supports records up to 64 KiB, which covers the multi-record window steps of up to 256 bytes.
- The upstream scratch-barrier implementation and its barrier-specific tests were left out because fork PR #780 already supplies that behavior through `VICTORY_INDIRECT_KEEP=1`. The comparison restores the scratch state expected by either setting. The upstream `VICTORY_INDIRECT_CARD_BUDGET` switch was left out because device-local charging belongs to the new opt-in switch.
- The ignored `indirect::bench` module compares per-record charging, PR #780’s keep behavior, grouped charging, and grouped charging with the device-local budget. It uses the production clamp variants and budget initialization.

## Verification

All Cargo commands used `CARGO_TARGET_DIR=../VIctory-fx/target-q`.

| Check | Result |
| --- | --- |
| `cargo check -p victory-gpu --features diagnostics` | Passed |
| `cargo test -p victory-gpu --features diagnostics --lib -- indirect` | 14 passed, 2 ignored |
| `cargo clippy -p victory-gpu --features diagnostics --all-targets -- -D warnings` | Passed with no warnings |
| Indirect suite on WARP, D3D12 debug layer requested, grouped budget off | 14 passed, 2 ignored |
| Same WARP suite, grouped budget and window checks on | 14 passed, 2 ignored |
| Same WARP suite, also with `VICTORY_INDIRECT_KEEP=1` | 14 passed, 2 ignored |
| Non-diagnostics check and indirect suite with grouped budget on for the tests | Check passed. 14 passed, 2 ignored |
| Ignored benchmark on WARP, D3D12 debug layer requested | Passed all eight cases |

Tests cover lane order across the 32-lane boundary, saturation across groups, count-buffer tails, budget reset across submissions, both diagnostic counter pairs, and detection of a GPU write under a window. The window test also executes the original window record after the comparison.

The benchmark ran on Microsoft Basic Render Driver. These are median GPU milliseconds for 3,200 draws in the diagnostics build with the debug layer requested.

| Mode | Single records | Windows of 32 |
| --- | ---: | ---: |
| Per-record charge | 122.40 | 68.32 |
| Per-record charge, keep scratch | 122.65 | 30.80 |
| Grouped charge, keep scratch | 154.46 | 29.91 |
| Grouped charge, keep scratch, device-local budget | 153.31 | 30.34 |

These software-adapter timings do not measure discrete-GPU or PCIe performance. The game was not run and the branch was not pushed.
