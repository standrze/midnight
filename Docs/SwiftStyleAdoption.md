# Swift style adoption in Midnight

[SwiftProjectStyle.md](SwiftProjectStyle.md) is the default for new and edited Swift code. Midnight predates that guide and has established runtime, protocol, benchmark, and optional backend boundaries. Keep those boundaries unless a behavior change justifies moving code.

The root `.editorconfig` and `.swift-format` set four spaces, a 120-column target, LF, and final newlines. The formatter preserves multiline string content. Run the full package check after Swift edits:

```sh
Scripts/check-swift-format.sh
```

Both scripts accept file paths for a targeted pass; with no arguments they cover the root package, the shared ModelFiles package, the optional Vision package, and Swift scripts. On 2026-09-26 the complete 191-file format check passed. The remaining 147 lines over 120 columns are primarily string literals, regular expressions, and diagnostic text whose contents must stay exact. No files have tabs or missing final newlines. A few snake-case declarations are explicitly exempted from the naming lint rule because they are JSON keys or published properties.

The formatter's `AllPublicDeclarationsHaveDocumentation` rule is disabled because it requires comments on simple public stored properties, which the style guide explicitly exempts. An initial audit with that rule enabled reported 530 undocumented stored properties and no undocumented public types, initializers, functions, or computed properties. Public operations have caller-facing summaries. Add comments to stored properties when their units or ownership are not clear from the enclosing type and property name; the audit clarified inspector token positions, inspection context capacities and trace counts, checkpoint voice positions, voice retention metadata, speech speed and pagination, runner token metrics, model-load token limits, and lifecycle generation guards.

Verification from the repository root:

```sh
Scripts/check-swift-format.sh
swift test
swift test --package-path Optional/Vision
swift build --product midnight
```

The format check and both test suites passed on 2026-09-26. The quality benchmark's command now delegates scoring, report construction, and writeout to named methods; its product build, help output, and input validation were checked after the refactor. The generation benchmark's command now delegates initial report construction, corpus reading, model loading, and per-sample execution while retaining failure-stage tracking and snapshot order; its product build, help output, and validation-only report were checked. The Talkie context probe's command now separates configuration loading, prompt selection, budget calculation, report construction, and execution; its product build, help output, and the root test suite passed after the refactor. The teacher KL benchmark now delegates teacher report construction, cache manifest writing, student scoring, final report construction, and writeout to named methods. Its student scoring loop delegates each sample's validation and measurements to a focused method while retaining aggregation and memory cleanup in the loop. The runtime benchmark's command now delegates checkpoint validation, runner setup, trial execution, report construction, and report writeout to named methods. Prompt-cache and hot-cache trial loops have their own methods, and ordinary A/B trials share one ordered mode selection through a named selector. Each A/B and cache comparison calculation has a focused method. The Talkie GQA recovery command now separates teacher-cache preparation from student training. These benchmark products built, their help output worked, the format check passed, and the root test suite passed after the refactors. Other benchmark executables retain long `main.swift` files because their command types and diagnostic helpers are coupled to each target; review their command bodies when changing benchmark behavior.

The formatter cannot decide every readability rule. A source-and-test scan found 402 single-line control-flow statements, all of which have been expanded. The change affected whitespace only in 81 Swift files when compared against the pre-expansion snapshot. A further scan found 68 single-line `else` branches, all of which were expanded. The root and optional Vision test suites passed afterward. Keep checking executable command bodies and runtime helpers for other overly compressed flow as those files are changed.

The guide's example tree illustrates small SwiftPM packages; Midnight keeps its current targets and `Docs/` directory. The user confirmed that the guides should stay in `Docs/` for now. Its package-specific sections describe Midnight's module direction, model-serving contracts, and verification commands. `Sources/ModelRunner/main.swift` is the server entrypoint; its command delegates setup to `ServerConfiguration`. Preserve public signatures and client-visible behavior during style cleanup.

## Guide audit

| Guide area | Current evidence or remaining work |
| --- | --- |
| Formatting and compressed flow | The full Swift format check passes. Single-line control-flow and `else` branches found in the source and test scans were expanded. Long literal and diagnostic lines are preserved intentionally. |
| Package boundaries | The root manifest lists executable, library, and test targets explicitly. Existing runtime, protocol, optional Vision, and shared model-file modules express ownership and dependency direction. |
| Navigation and entrypoints | The main server command and benchmark commands delegate work to named methods. Focused files coexist with larger runtime and benchmark files that reflect established model and diagnostic boundaries. Review newly changed large methods for further splitting. |
| API contracts | Public types and operations have summaries. Simple stored properties rely on their names and enclosing contracts. Ambiguous token, byte, time, position, speed, pagination, and lifecycle fields identified in the audit have explicit comments; review new fields by the same rule. |
| Project documentation | The 78-line README keeps the overview and points to [getting started](getting-started.md) and [usage and model details](usage-and-models.md). Local Markdown links in these pages, the guide, and the adoption notes resolve in this checkout. |
| Verification | The root and optional Vision test suites, formatter check, and `midnight` build are the package gates. Benchmark product builds and CLI help checks cover their source-only refactors. |

A Swift parser scan of active source found the largest remaining methods in model selection, HTTP request handling, scoring, and native generation. Their lengths alone are not a reason to introduce new targets; split methods when a coherent phase can be named without changing request ordering or model ownership. The managed HTTP dispatcher now delegates model-route leasing and forwarding to a focused method. Chat request decoding and parameter validation now run independently of the NIO channel, with focused tests for accepted limits and a conflicting-parameter error. `ModelLoader.validate` now delegates backend selection and its token/context limits to a focused resolver, preserving validation order. The `midnight` build, root tests, and 12 focused model-loader tests passed after these refactors and removal of an unreachable fallback in model selection.

## Follow-up audit: 30 September 2026

The current authored tree contains 232 Swift files across the root, shared
ModelFiles, optional Vision, and script scopes. The full strict format check
passes. All files use LF, have a final newline, and contain no tabs. The 155
remaining lines over 120 columns are formatter-preserved content, including
literal and diagnostic text. The formatter scripts now discover Swift files
recursively under `Scripts/`; the earlier explicit list missed the added-token
regex benchmark.

This pass expanded newly added or previously missed single-line control flow
in benchmarks, scripts, tests, HTTP handlers, and speech producers. It also
replaced the generation monitor's nested phase ternary with explicit branches,
documented the speech and Vision protocol requirements, and corrected a Vision
response comment. Public signatures, literal contents, request ordering, and
client-visible behavior are preserved. The manifest comment now explains why
the `@main` command in `main.swift` requires `-parse-as-library`.

Production dependencies continue to follow the guide. The short server
entrypoint delegates configuration; protocol types do not depend on MLX; and
console I/O, model execution, shared file leases, and the optional Vision worker
have explicit owners. Existing large files are not, by themselves, grounds for
new targets or a broad source move.

The remaining layout work is deliberate and separate from this consistency pass:

- `Tests/ModelRunnerProtocolTests` mixes protocol, core runtime, and backend
  coverage. Of its 92 Swift files, 44 import `ModelRunnerCore`. Split these by
  ownership in a dedicated change, preserving discovery and fixture paths.
- `Shared/ModelFiles` has no test target; its direct lease test currently runs
  in Vision's `FastVLMAdmissionTests`. Move that contract test into the shared
  package when separating the test suites. Vision HTTP tests also share the
  `VisionProtocolTests` target.
- The paired benchmark's `run()` and Vision worker's entrypoint still combine
  several setup and execution phases. Extract named phases when touching their
  behavior, retaining lease lifetimes and cleanup ordering.
- `LocalModelRunner` and `ModelHTTPServer` remain candidates for focused
  capability splits, as recorded in the [readiness audit](readiness-audit-2026-09-30.md).
- The [project layout](project-layout.md) records the sibling menu bar launcher's
  old binary default and the missing historical Chat checkout. Neither requires
  changing Midnight's module boundaries.

Verification passed on the final source state: the full format check, root Swift
tests, all 19 optional Vision tests, the `midnight` and paired benchmark product
builds, and the paired benchmark's help output. The three Swift scripts were
type-checked or compiled; the probe checks passed 38,896 growth cases, and the
regex benchmark preserved segmentation with the Talkie tokenizer fixture.
Hardware-specific tests retain their existing skip gates. No live weight-loading
or terminal smoke test was needed for this formatting and documentation pass.

This checkout has no root Git metadata. Before-copies and the audit's change
inventory are retained under `artifacts/swift-style-audit-20260930/`.

The later [ownership cleanup](project-separation-20260930.json) moved the Talkie
GQA training command into Training and the Gemma planner into the project then
named Facet, now Wick. References
above to their earlier style checks describe the audit before that relocation;
the [project layout](project-layout.md) records their current homes.

## Terminal console

The loom/weft server console requires Swift 6.4. Its state and rendering live on
the main actor in `ServerConsole.swift`; `ModelLifecycleManager` continues to own
model admission and execution. `ConsoleOutput.swift` confines descriptor
redirection and bounded log capture to the process boundary. It saves a terminal
descriptor for loom output and size queries, and restores stdout/stderr before
weft restores raw mode and the alternate screen.

Run `python3 Scripts/smoke-server-console.py .build/debug/midnight` after building
to check terminal resize, q/Ctrl-C/SIGTERM exits, failed startup selection, bind
errors and `--no-ui`. It creates temporary listeners with a test API key and
does not load weights. Policy, clipped rendering and shutdown admission are
also covered by the root Swift tests. macOS PTY behavior is verified; Linux
terminal behavior still requires validation.

The live dashboard uses a passive request-local observer in the reviewed
`mlx-swift-lm-generation-progress.patch`. `GenerationProgress.swift` keeps its
counters behind a lock so console refreshes never wait for the decoding actor.
`ServerDashboard.swift` owns the measurement layout and budget bars. The observer
counts emitted token IDs independently of text chunks without changing sampling or events.
Run `bash Tests/Shell/GenerationProgressPatchTests.sh` to verify atomic replay,
reversal and conflict rejection. To check the dashboard with a local text model,
run `python3 Scripts/smoke-server-dashboard.py /path/to/checkpoint` after building.
The latter loads the supplied weights and makes one local 256-token chat request;
its API key, listener port and policy file are temporary.
