#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != Darwin ]]; then
  echo 'SKIP: the optional pinned runtime is macOS-only.'
  exit 0
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TASK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/midnight-pinned-lazy.XXXXXXXX")"
trap 'rm -rf "$TASK_DIR"' EXIT

cat > "$TASK_DIR/Probe.swift" <<'SWIFT'
import Foundation

@main struct Probe {
  static func main() async throws {
    // This fresh process has not run any other runtime tests. Observing the
    // optional worker for cleanup must not create its permanent pthread.
    precondition(MLXPinnedRuntime.existing == nil)
    precondition(MLXPinnedRuntime.existing == nil)
    let runtime = MLXPinnedRuntime.shared
    precondition(MLXPinnedRuntime.existing === runtime)
    let thread = await runtime.runCleanup { Thread.current.name }
    precondition(thread == "midnight.mlx-pinned")
    let cancelled = Task {
      withUnsafeCurrentTask { $0?.cancel() }
      return await runtime.runCleanup { Thread.current.name }
    }
    let cleanupThread = await cancelled.value
    precondition(cleanupThread == "midnight.mlx-pinned")
    print("PASS: cleanup observation stays lazy; explicit use creates one worker; cancelled cleanup still runs")
  }
}
SWIFT

swiftc -swift-version 6 -parse-as-library -module-cache-path "$TASK_DIR/module-cache" \
  "$ROOT/Sources/ModelRunnerCore/MLXPinnedRuntime.swift" "$TASK_DIR/Probe.swift" \
  -o "$TASK_DIR/probe"
"$TASK_DIR/probe"
