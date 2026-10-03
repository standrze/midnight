#!/usr/bin/env bash
set -euo pipefail
REPLAY_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPLAY_TEST_DIR="$(mktemp -d "${TMPDIR:-/tmp}/midnight-replay-tests.XXXXXX")"
trap 'rm -rf "$REPLAY_TEST_DIR"' EXIT
"${CXX:-c++}" -std=c++17 -Wall -Wextra -Werror -pedantic -pthread \
  "$REPLAY_ROOT/tests/session_core_tests.cpp" -o "$REPLAY_TEST_DIR/session-tests"
"$REPLAY_TEST_DIR/session-tests"
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s "$REPLAY_ROOT/tests" -p 'test_*.py'
