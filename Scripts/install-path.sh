#!/usr/bin/env bash

midnight_configure_path() {
  local bin="$1" answer config line shell_name
  if [[ -t 1 ]] && ( : </dev/tty ) 2>/dev/null; then
    printf 'Add Midnight to PATH? [y/N] ' >/dev/tty
    IFS= read -r answer </dev/tty || answer=n
    case "$answer" in y|Y|yes|YES) ;; *) return ;; esac
  else
    echo "PATH unchanged (no interactive terminal). Add $bin to your shell configuration to use midnight."
    return
  fi
  shell_name="$(basename "${SHELL:-/bin/zsh}")"
  case "$shell_name" in
    zsh) config="${ZDOTDIR:-$HOME}/.zshrc" ;;
    bash)
      if [[ "$(uname -s)" == Darwin ]]; then config="$HOME/.bash_profile"; else config="$HOME/.bashrc"; fi
      ;;
    *) echo "PATH unchanged: add $bin using your $shell_name shell configuration."; return ;;
  esac
  printf -v line 'export PATH=%q:"$PATH"' "$bin"
  mkdir -p "$(dirname "$config")"
  if ! grep -Fqx "$line" "$config" 2>/dev/null; then
    printf '\n# Midnight runner\n%s\n' "$line" >> "$config"
  fi
  echo "PATH configured in $config. Open a new terminal to use midnight."
}
