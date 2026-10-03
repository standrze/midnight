#!/usr/bin/env bash
set -euo pipefail

# Public bootstrap: resolve the newest published release, including prereleases.
# No version is pinned here. GitHub returns public releases newest first.
main() {
  local repo=standrze/midnight platform version archive stage expected actual
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --help|-h) echo 'Usage: install.sh (prompts to add Midnight to PATH)'; return ;;
      *) echo "Unknown option: $1" >&2; return 2 ;;
    esac
  done
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)
      platform=macos-arm64
      if [[ "$(sw_vers -productVersion | cut -d. -f1)" -lt 26 ]]; then
        echo 'The prebuilt Metal runner requires macOS 26 or newer.' >&2
        return 1
      fi
      ;;
    Linux-x86_64)
      platform=linux-x86_64-cuda13-sm89
      echo 'Linux package: RTX 4090 (sm_89), CUDA 13, cuDNN 9; tested on Ubuntu 24.04.'
      ;;
    *)
      echo 'Prebuilt downloads support Apple silicon on macOS 26+ and x86-64 Linux/CUDA 13 (sm_89).' >&2
      echo 'Linux/CUDA and other configurations: https://github.com/standrze/midnight#readme' >&2
      return 1
      ;;
  esac
  stage="$(mktemp -d)"
  trap "$(printf 'rm -rf -- %q' "$stage")" EXIT
  curl --proto '=https' --tlsv1.2 -fsSL --retry 3 \
    "https://api.github.com/repos/$repo/releases?per_page=1" -o "$stage/release.json"
  version="$(sed -n 's/^[[:space:]]*"tag_name":[[:space:]]*"\([^"]*\)".*/\1/p' "$stage/release.json")"
  if [[ ! "$version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.-]+)?$ ]]; then
    echo 'Could not determine the newest published Midnight release.' >&2
    return 1
  fi
  archive="midnight-$version-$platform.tar.gz"
  echo "Downloading Midnight $version..."
  curl --proto '=https' --tlsv1.2 -fsSL --retry 3 \
    "https://github.com/$repo/releases/download/$version/$archive" -o "$stage/$archive"
  curl --proto '=https' --tlsv1.2 -fsSL --retry 3 \
    "https://github.com/$repo/releases/download/$version/SHA256SUMS" -o "$stage/SHA256SUMS"
  expected="$(awk -v file="$archive" '$2 == file {print $1}' "$stage/SHA256SUMS")"
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$stage/$archive" | awk '{print $1}')"
  else
    actual="$(shasum -a 256 "$stage/$archive" | awk '{print $1}')"
  fi
  if [[ ! "$expected" =~ ^[0-9a-f]{64}$ || "$expected" != "$actual" ]]; then
    echo 'Release checksum verification failed; nothing was installed.' >&2
    return 1
  fi
  tar -tzf "$stage/$archive" > "$stage/files.txt"
  if grep -Eq '(^/|(^|/)\.\.(/|$))' "$stage/files.txt"; then
    echo 'Unsafe archive paths; nothing was installed.' >&2
    return 1
  fi
  mkdir "$stage/unpacked"
  tar -xzf "$stage/$archive" -C "$stage/unpacked"
  bash "$stage/unpacked/midnight/install.sh" --binary "$stage/unpacked/midnight/bin/midnight"
  "$HOME/.midnight/bin/midnight" --version
  echo 'Ready. Add ~/.midnight/bin to PATH if needed, then run:'
  echo '  midnight --list'
  echo '  midnight download'
  rm -rf "$stage"
  trap - EXIT
}
main "$@"
