#!/usr/bin/env bash
set -euo pipefail
KV_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KV_CHECKOUT="$KV_ROOT/.build/checkouts/mlx-swift-lm"
KV_PATCH="$KV_ROOT/Patches/mlx-swift-lm-direct-kv-slice-update.patch"
KV_TEMP="$(mktemp -d "${TMPDIR:-/tmp}/midnight-formatted-kv-patch.XXXXXX")"
trap 'rm -rf "$KV_TEMP"' EXIT
mkdir -p "$KV_TEMP/checkout"
git -C "$KV_CHECKOUT" archive 14414441fa44f45eee35a61e9fa0bab577cf9734 \
  Libraries/MLXLMCommon/KVCache.swift Tests/MLXLMTests/KVCacheTests.swift \
  | tar -xf - -C "$KV_TEMP/checkout"
# Load only the real patch function, without package resolution or live mutations.
sed -n '/^apply_dependency_patch() {/,/^}/p' "$KV_ROOT/prepare-dependencies.sh" > "$KV_TEMP/apply.sh"
source "$KV_TEMP/apply.sh"
apply_dependency_patch 'mlx-swift-lm direct KV slice update' "$KV_TEMP/checkout" "$KV_PATCH"
apply_dependency_patch 'mlx-swift-lm direct KV slice update' "$KV_TEMP/checkout" "$KV_PATCH"
sed \
  -e '/updateKVCacheSlice/s/previous \.\.< self.offset/previous..<self.offset/' \
  -e '/updateKVCacheSlice/s/idx \.\.< (idx + S)/idx..<(idx + S)/' \
  "$KV_TEMP/checkout/Libraries/MLXLMCommon/KVCache.swift" > "$KV_TEMP/formatted.swift"
cp "$KV_TEMP/formatted.swift" "$KV_TEMP/checkout/Libraries/MLXLMCommon/KVCache.swift"
apply_dependency_patch 'mlx-swift-lm direct KV slice update' "$KV_TEMP/checkout" "$KV_PATCH"
# A semantic change must not be hidden by formatting compatibility.
sed 's/range: previous..<self.offset/range: 0..<self.offset/' "$KV_TEMP/formatted.swift" \
  > "$KV_TEMP/checkout/Libraries/MLXLMCommon/KVCache.swift"
cp -R "$KV_TEMP/checkout" "$KV_TEMP/before"
if apply_dependency_patch 'mlx-swift-lm direct KV slice update' "$KV_TEMP/checkout" "$KV_PATCH"; then
  echo 'Expected altered cache ranges to fail.' >&2
  exit 1
fi
diff -qr "$KV_TEMP/before" "$KV_TEMP/checkout" >/dev/null
echo 'Formatted KV slice patch compatibility and semantic drift rejection passed.'
