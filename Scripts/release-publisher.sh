#!/usr/bin/env bash

# Publish a verified custom-scratch RTX 4090 release without copying an opaque
# binary into SwiftPM's default build cache. Callers source this file.

model_runner_release_sha256() {
  local path="$1"

  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    echo "A SHA-256 utility (sha256sum or shasum) is required." >&2
    return 1
  fi
}

model_runner_release_manifest_value() {
  local manifest="$1"
  local key="$2"
  local count

  count="$(awk -v prefix="$key=" 'index($0, prefix) == 1 {count++} END {print count + 0}' "$manifest")"
  if [[ "$count" != "1" ]]; then
    echo "Release manifest must contain exactly one $key field: $manifest" >&2
    return 1
  fi
  awk -v prefix="$key=" 'index($0, prefix) == 1 {print substr($0, length(prefix) + 1)}' "$manifest"
}

model_runner_release_reject_multiline() {
  local label="$1"
  local value="$2"

  if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
    echo "$label may not contain newline characters." >&2
    return 1
  fi
}

model_runner_release_paths() {
  local package_root="$1"

  MODEL_RUNNER_RTX4090_STABLE_PATH="$package_root/bin/midnight-rtx4090"
  MODEL_RUNNER_RTX4090_MANIFEST_PATH="$package_root/bin/midnight-rtx4090.manifest"
  MODEL_RUNNER_RTX4090_COMPAT_PATH="$package_root/.build/release/midnight"
}

model_runner_verify_rtx4090_manifest_and_binary() {
  local package_root="$1"
  local expected_scratch="${2:-}"
  local manifest_profile
  local manifest_version
  local manifest_scratch
  local manifest_source
  local manifest_stable
  local manifest_compat
  local expected_hash
  local actual_hash
  local expected_size
  local actual_size

  model_runner_release_paths "$package_root"
  if [[ ! -f "$MODEL_RUNNER_RTX4090_MANIFEST_PATH" \
    || -L "$MODEL_RUNNER_RTX4090_MANIFEST_PATH" ]]; then
    echo "Verified RTX 4090 release manifest is missing: $MODEL_RUNNER_RTX4090_MANIFEST_PATH" >&2
    return 1
  fi
  if [[ ! -f "$MODEL_RUNNER_RTX4090_STABLE_PATH" \
    || ! -x "$MODEL_RUNNER_RTX4090_STABLE_PATH" \
    || -L "$MODEL_RUNNER_RTX4090_STABLE_PATH" ]]; then
    echo "Published RTX 4090 release is missing or not a regular executable: $MODEL_RUNNER_RTX4090_STABLE_PATH" >&2
    return 1
  fi

  manifest_version="$(model_runner_release_manifest_value "$MODEL_RUNNER_RTX4090_MANIFEST_PATH" manifest_version)" || return 1
  manifest_profile="$(model_runner_release_manifest_value "$MODEL_RUNNER_RTX4090_MANIFEST_PATH" build_profile)" || return 1
  manifest_scratch="$(model_runner_release_manifest_value "$MODEL_RUNNER_RTX4090_MANIFEST_PATH" source_scratch)" || return 1
  manifest_source="$(model_runner_release_manifest_value "$MODEL_RUNNER_RTX4090_MANIFEST_PATH" source_binary)" || return 1
  manifest_stable="$(model_runner_release_manifest_value "$MODEL_RUNNER_RTX4090_MANIFEST_PATH" stable_binary)" || return 1
  manifest_compat="$(model_runner_release_manifest_value "$MODEL_RUNNER_RTX4090_MANIFEST_PATH" compatibility_link)" || return 1
  expected_hash="$(model_runner_release_manifest_value "$MODEL_RUNNER_RTX4090_MANIFEST_PATH" sha256)" || return 1
  expected_size="$(model_runner_release_manifest_value "$MODEL_RUNNER_RTX4090_MANIFEST_PATH" size_bytes)" || return 1

  if [[ "$manifest_version" != "1" ]]; then
    echo "Unsupported RTX 4090 release manifest version: $manifest_version" >&2
    return 1
  fi
  case "$manifest_profile" in
    configuration=release:cuda:sm_89:*) ;;
    *)
      echo "Published RTX 4090 manifest does not describe a release sm_89 build." >&2
      return 1
      ;;
  esac
  if [[ "$manifest_scratch" != /* || "$manifest_source" != "$manifest_scratch"/* ]]; then
    echo "Published RTX 4090 manifest has an invalid custom-scratch source." >&2
    return 1
  fi
  if [[ -n "$expected_scratch" && "$manifest_scratch" != "$expected_scratch" ]]; then
    echo "Published RTX 4090 release came from a different scratch tree." >&2
    echo "Expected: $expected_scratch" >&2
    echo "Manifest: $manifest_scratch" >&2
    return 1
  fi
  if [[ "$manifest_stable" != "$MODEL_RUNNER_RTX4090_STABLE_PATH" \
    || "$manifest_compat" != "$MODEL_RUNNER_RTX4090_COMPAT_PATH" ]]; then
    echo "Published RTX 4090 manifest paths do not match this package root." >&2
    return 1
  fi
  if [[ ! "$expected_hash" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "Published RTX 4090 manifest contains an invalid SHA-256 hash." >&2
    return 1
  fi
  if [[ ! "$expected_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "Published RTX 4090 manifest contains an invalid artifact size." >&2
    return 1
  fi
  actual_hash="$(model_runner_release_sha256 "$MODEL_RUNNER_RTX4090_STABLE_PATH")" || return 1
  if [[ "$actual_hash" != "$expected_hash" ]]; then
    echo "Published RTX 4090 release hash does not match its manifest." >&2
    echo "Expected: $expected_hash" >&2
    echo "Actual:   $actual_hash" >&2
    return 1
  fi
  actual_size="$(wc -c < "$MODEL_RUNNER_RTX4090_STABLE_PATH" | tr -d '[:space:]')"
  if [[ "$actual_size" != "$expected_size" ]]; then
    echo "Published RTX 4090 release size does not match its manifest." >&2
    return 1
  fi
}

model_runner_verify_rtx4090_publication() {
  local package_root="$1"
  local expected_scratch="${2:-}"
  local link_target

  model_runner_verify_rtx4090_manifest_and_binary "$package_root" "$expected_scratch" || return 1
  if [[ ! -L "$MODEL_RUNNER_RTX4090_COMPAT_PATH" ]]; then
    echo "Expected compatibility path is not a symlink: $MODEL_RUNNER_RTX4090_COMPAT_PATH" >&2
    return 1
  fi
  link_target="$(readlink "$MODEL_RUNNER_RTX4090_COMPAT_PATH")"
  if [[ "$link_target" != "$MODEL_RUNNER_RTX4090_STABLE_PATH" ]]; then
    echo "Compatibility link does not target the verified RTX 4090 release." >&2
    echo "Expected: $MODEL_RUNNER_RTX4090_STABLE_PATH" >&2
    echo "Actual:   $link_target" >&2
    return 1
  fi
  if [[ ! "$MODEL_RUNNER_RTX4090_COMPAT_PATH" -ef "$MODEL_RUNNER_RTX4090_STABLE_PATH" ]]; then
    echo "Compatibility link does not resolve to the verified RTX 4090 release." >&2
    return 1
  fi
}

model_runner_publish_rtx4090_release() {
  local package_root="$1"
  local scratch_path="$2"
  local source_binary="$3"
  local expected_profile="$4"
  local source_dir
  local profile_marker
  local recorded_profile
  local source_hash
  local copied_hash
  local stable_temp=""
  local manifest_temp=""
  local link_temp=""
  local release_dir
  local existing_target

  package_root="$(cd "$package_root" && pwd -P)" || return 1
  scratch_path="$(cd "$scratch_path" && pwd -P)" || return 1
  source_dir="$(cd "$(dirname "$source_binary")" && pwd -P)" || return 1
  source_binary="$source_dir/$(basename "$source_binary")"
  model_runner_release_paths "$package_root"

  model_runner_release_reject_multiline "Scratch path" "$scratch_path" || return 1
  model_runner_release_reject_multiline "Source binary path" "$source_binary" || return 1
  model_runner_release_reject_multiline "Build profile" "$expected_profile" || return 1
  if [[ "$scratch_path" == "$package_root/.build" ]]; then
    echo "RTX 4090 publisher requires an isolated custom SwiftPM scratch tree." >&2
    return 1
  fi
  case "$source_binary" in
    "$scratch_path"/*) ;;
    *)
      echo "Release source is not inside the selected custom scratch tree: $source_binary" >&2
      return 1
      ;;
  esac
  if [[ ! -f "$source_binary" || ! -x "$source_binary" || -L "$source_binary" ]]; then
    echo "Release source is missing or not a regular executable: $source_binary" >&2
    return 1
  fi
  case "$expected_profile" in
    configuration=release:cuda:sm_89:*) ;;
    *)
      echo "RTX 4090 publisher requires a release CUDA sm_89 build profile." >&2
      return 1
      ;;
  esac

  profile_marker="$scratch_path/.model-runner-profile"
  if [[ ! -f "$profile_marker" || -L "$profile_marker" ]]; then
    echo "Release build profile marker is missing: $profile_marker" >&2
    return 1
  fi
  if [[ "$(wc -l < "$profile_marker" | tr -d '[:space:]')" != "1" ]]; then
    echo "Release build profile marker must contain exactly one line: $profile_marker" >&2
    return 1
  fi
  IFS= read -r recorded_profile < "$profile_marker" || true
  if [[ "$recorded_profile" != "$expected_profile" ]]; then
    echo "Release build profile marker does not match the completed build." >&2
    echo "Expected: $expected_profile" >&2
    echo "Marker:   $recorded_profile" >&2
    return 1
  fi

  mkdir -p "$package_root/bin"
  if [[ -e "$MODEL_RUNNER_RTX4090_STABLE_PATH" \
    || -L "$MODEL_RUNNER_RTX4090_STABLE_PATH" \
    || -e "$MODEL_RUNNER_RTX4090_MANIFEST_PATH" \
    || -L "$MODEL_RUNNER_RTX4090_MANIFEST_PATH" ]]; then
    if [[ ! -f "$MODEL_RUNNER_RTX4090_STABLE_PATH" \
      || ! -f "$MODEL_RUNNER_RTX4090_MANIFEST_PATH" ]]; then
      echo "Incomplete prior RTX 4090 publication; refusing to overwrite it." >&2
      return 1
    fi
    model_runner_verify_rtx4090_manifest_and_binary "$package_root" || {
      echo "Prior RTX 4090 publication failed verification; refusing to overwrite it." >&2
      return 1
    }
  fi

  release_dir="$(dirname "$MODEL_RUNNER_RTX4090_COMPAT_PATH")"
  if [[ -L "$release_dir" && ! -d "$release_dir" ]]; then
    echo "SwiftPM release directory is a dangling symlink: $release_dir" >&2
    return 1
  fi
  mkdir -p "$release_dir"
  if [[ -e "$MODEL_RUNNER_RTX4090_COMPAT_PATH" \
    || -L "$MODEL_RUNNER_RTX4090_COMPAT_PATH" ]]; then
    if [[ ! -L "$MODEL_RUNNER_RTX4090_COMPAT_PATH" ]]; then
      echo "Refusing to replace a non-symlink SwiftPM artifact: $MODEL_RUNNER_RTX4090_COMPAT_PATH" >&2
      return 1
    fi
    existing_target="$(readlink "$MODEL_RUNNER_RTX4090_COMPAT_PATH")"
    if [[ "$existing_target" != "$MODEL_RUNNER_RTX4090_STABLE_PATH" ]]; then
      echo "Refusing to replace an unmanaged compatibility link: $MODEL_RUNNER_RTX4090_COMPAT_PATH" >&2
      return 1
    fi
  fi

  stable_temp="$(mktemp "$package_root/bin/.midnight-rtx4090.XXXXXX")" || return 1
  if ! install -m 0755 "$source_binary" "$stable_temp"; then
    rm -f "$stable_temp"
    return 1
  fi
  source_hash="$(model_runner_release_sha256 "$source_binary")" || {
    rm -f "$stable_temp"
    return 1
  }
  copied_hash="$(model_runner_release_sha256 "$stable_temp")" || {
    rm -f "$stable_temp"
    return 1
  }
  if [[ "$source_hash" != "$copied_hash" ]]; then
    echo "Staged RTX 4090 release hash does not match its source." >&2
    rm -f "$stable_temp"
    return 1
  fi

  manifest_temp="$(mktemp "$package_root/bin/.midnight-rtx4090.manifest.XXXXXX")" || {
    rm -f "$stable_temp"
    return 1
  }
  if ! {
    printf 'manifest_version=1\n'
    printf 'build_profile=%s\n' "$expected_profile"
    printf 'source_scratch=%s\n' "$scratch_path"
    printf 'source_binary=%s\n' "$source_binary"
    printf 'stable_binary=%s\n' "$MODEL_RUNNER_RTX4090_STABLE_PATH"
    printf 'compatibility_link=%s\n' "$MODEL_RUNNER_RTX4090_COMPAT_PATH"
    printf 'sha256=%s\n' "$source_hash"
    printf 'size_bytes=%s\n' "$(wc -c < "$source_binary" | tr -d '[:space:]')"
  } > "$manifest_temp"; then
    rm -f "$stable_temp" "$manifest_temp"
    return 1
  fi
  chmod 0644 "$manifest_temp"

  mv -f "$stable_temp" "$MODEL_RUNNER_RTX4090_STABLE_PATH"
  stable_temp=""
  mv -f "$manifest_temp" "$MODEL_RUNNER_RTX4090_MANIFEST_PATH"
  manifest_temp=""
  model_runner_verify_rtx4090_manifest_and_binary "$package_root" "$scratch_path" || return 1

  link_temp="$(mktemp "$release_dir/.midnight-link.XXXXXX")" || return 1
  rm -f "$link_temp"
  if ! ln -s "$MODEL_RUNNER_RTX4090_STABLE_PATH" "$link_temp"; then
    rm -f "$link_temp"
    return 1
  fi
  mv -f "$link_temp" "$MODEL_RUNNER_RTX4090_COMPAT_PATH"
  link_temp=""
  model_runner_verify_rtx4090_publication "$package_root" "$scratch_path"
}
