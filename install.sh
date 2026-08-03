#!/usr/bin/env bash
set -euo pipefail

SCRIPT_VERSION="1.0.0"

OWNER="${ANVIL_GITHUB_OWNER:-BrokkAi}"
REPO="anvil"
BIN_NAME="anvil"
INSTALL_DIR="${ANVIL_INSTALL_DIR:-${INSTALL_DIR:-$HOME/.local/bin}}"

TMP_DIR=""
OS_FAMILY=""
ARCH=""
RUST_TARGET=""

log() {
  printf 'anvil-installer: %s\n' "$*"
}

warn() {
  printf 'anvil-installer: warning: %s\n' "$*" >&2
}

die() {
  printf 'anvil-installer: error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Install the released Anvil binary.

Usage:
  curl -fsSL https://raw.githubusercontent.com/BrokkAi/anvil/refs/heads/master/install.sh | bash

Platforms:
  macOS            Apple Silicon and Intel, via the universal binary.
  Linux            x86-64 and ARM64 with glibc.
  WSL              Supported as Linux. Installs the Linux binary, which runs
                   inside WSL and is not callable from Windows-native tools.
  Android          ARM64 under Termux.
  Windows          Not installed by this script. Download the release archive
                   or install from source with Cargo.

Environment:
  INSTALL_DIR              Install directory. Defaults to ~/.local/bin.
  ANVIL_INSTALL_DIR        Same as INSTALL_DIR, with higher precedence.
  ANVIL_GITHUB_OWNER       GitHub owner to download from. Defaults to BrokkAi.
  ANVIL_VERSION            Optional release tag to install, for example v0.24.2.
  GITHUB_TOKEN             Optional token for GitHub API rate limits.
  PROFILE                  Optional shell profile to update when INSTALL_DIR is not on PATH.
EOF
}

cleanup() {
  if [[ -n "$TMP_DIR" && -d "$TMP_DIR" ]]; then
    rm -rf "$TMP_DIR"
  fi
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

curl_args() {
  printf '%s\0' -fsSL --retry 3 --retry-delay 1
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    printf '%s\0' -H "Authorization: Bearer ${GITHUB_TOKEN}"
  fi
}

download_file() {
  local url="$1"
  local dest="$2"
  local -a args=()

  while IFS= read -r -d '' arg; do
    args+=("$arg")
  done < <(curl_args)

  curl "${args[@]}" -o "$dest" "$url"
}

detect_platform() {
  local uname_s uname_m uname_o=""

  uname_s="$(uname -s)"
  uname_m="$(uname -m)"
  if uname -o >/dev/null 2>&1; then
    uname_o="$(uname -o)"
  fi

  case "$uname_m" in
    x86_64 | amd64) ARCH="x86_64" ;;
    arm64 | aarch64) ARCH="aarch64" ;;
    *) die "unsupported CPU architecture: ${uname_m}" ;;
  esac

  case "$uname_s" in
    Darwin)
      OS_FAMILY="macos"
      RUST_TARGET="universal-apple-darwin"
      ;;
    Linux)
      if [[ "$uname_o" == "Android" || "${PREFIX:-}" == *com.termux* ]]; then
        OS_FAMILY="android"
        [[ "$ARCH" == "aarch64" ]] || die "no Android build is published for ${uname_m}; only aarch64-linux-android is released"
        RUST_TARGET="aarch64-linux-android"
      else
        OS_FAMILY="linux"
        RUST_TARGET="${ARCH}-unknown-linux-gnu"
      fi
      ;;
    MINGW* | MSYS* | CYGWIN*)
      die "Windows release assets are not installed by this script. Download the archive from https://github.com/${OWNER}/${REPO}/releases, install from source with 'cargo install brokk-anvil --locked', or run this script in WSL."
      ;;
    *) die "unsupported OS: ${uname_s}" ;;
  esac
}

release_endpoint() {
  if [[ -n "${ANVIL_VERSION:-}" ]]; then
    printf 'https://api.github.com/repos/%s/%s/releases/tags/%s\n' "$OWNER" "$REPO" "$ANVIL_VERSION"
  else
    printf 'https://api.github.com/repos/%s/%s/releases/latest\n' "$OWNER" "$REPO"
  fi
}

release_tag() {
  { grep -o '"tag_name"[[:space:]]*:[[:space:]]*"[^"]*"' "$1" || true; } |
    sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' |
    head -n 1
}

release_asset_urls() {
  { grep -o '"browser_download_url"[[:space:]]*:[[:space:]]*"[^"]*"' "$1" || true; } |
    sed -n 's/.*"browser_download_url"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p'
}

available_assets() {
  release_asset_urls "$1" | sed 's#.*/##' | sed '/[.]sha256$/d' | tr '\n' ' '
}

select_asset() {
  local release_file="$1" tag="$2" url name
  local expected="brokk-anvil-${tag}-${RUST_TARGET}.zip"

  while IFS= read -r url; do
    name="${url##*/}"
    if [[ "$name" == "$expected" ]]; then
      printf '%s\n' "$url"
      return 0
    fi
  done < <(release_asset_urls "$release_file")

  die "no Anvil asset found for ${OS_FAMILY}/${ARCH} in ${OWNER}/${REPO} release ${tag}. Available assets: $(available_assets "$release_file")"
}

checksum_url_for() {
  local release_file="$1" checksum_name="${2}.sha256" url

  while IFS= read -r url; do
    if [[ "${url##*/}" == "$checksum_name" ]]; then
      printf '%s\n' "$url"
      return 0
    fi
  done < <(release_asset_urls "$release_file")
}

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

verify_checksum() {
  local release_file="$1" asset_name="$2" asset_file="$3"
  local checksum_url checksum_file expected actual

  checksum_url="$(checksum_url_for "$release_file" "$asset_name" || true)"
  [[ -n "$checksum_url" ]] || die "no checksum published for ${asset_name}; refusing to install an unverified archive"

  checksum_file="${TMP_DIR}/${asset_name}.sha256"
  download_file "$checksum_url" "$checksum_file"
  expected="$(awk '{print $1}' "$checksum_file" | head -n 1)"
  [[ "$expected" =~ ^[[:xdigit:]]{64}$ ]] || die "invalid checksum published for ${asset_name}"

  actual="$(hash_file "$asset_file")"
  [[ "$expected" == "$actual" ]] || die "checksum mismatch for ${asset_name}: expected ${expected}, got ${actual}"
}

strip_quarantine() {
  if [[ "$OS_FAMILY" == "macos" ]] && command -v xattr >/dev/null 2>&1; then
    xattr -dr com.apple.quarantine "$1" >/dev/null 2>&1 || true
  fi
}

ensure_install_dir() {
  [[ -d "$INSTALL_DIR" ]] && return 0
  mkdir -p "$INSTALL_DIR" 2>/dev/null && return 0
  command -v sudo >/dev/null 2>&1 || die "cannot create ${INSTALL_DIR}; set INSTALL_DIR to a writable directory"
  sudo mkdir -p "$INSTALL_DIR"
}

install_dir_on_path() {
  case ":${PATH}:" in *":${INSTALL_DIR}:"*) return 0 ;; *) return 1 ;; esac
}

can_prompt_on_tty() {
  [[ -r /dev/tty && -w /dev/tty ]] && { : >/dev/tty; } 2>/dev/null
}

shell_quote() {
  printf "'"
  printf '%s' "$1" | sed "s/'/'\\\\''/g"
  printf "'"
}

path_export_line() {
  printf 'export PATH=%s:"$PATH"\n' "$(shell_quote "$INSTALL_DIR")"
}

default_shell_profile() {
  local shell_name="${SHELL:-}"
  [[ -n "${PROFILE:-}" ]] && { printf '%s\n' "$PROFILE"; return 0; }
  shell_name="${shell_name##*/}"
  case "$shell_name" in
    zsh) printf '%s/.zshrc\n' "$HOME" ;;
    bash) [[ "$OS_FAMILY" == "macos" ]] && printf '%s/.bash_profile\n' "$HOME" || printf '%s/.bashrc\n' "$HOME" ;;
    ksh) printf '%s/.kshrc\n' "$HOME" ;;
    sh) printf '%s/.profile\n' "$HOME" ;;
    *) return 1 ;;
  esac
}

append_install_dir_to_profile() {
  local profile="$1"
  local line="$2"
  local profile_dir

  profile_dir="$(dirname "$profile")"
  if ! mkdir -p "$profile_dir" 2>/dev/null; then
    warn "could not create ${profile_dir}; add this manually: ${line}"
    return 1
  fi

  {
    printf '\n# Added by Anvil installer\n'
    printf '%s\n' "$line"
  } >>"$profile" || {
    warn "could not update ${profile}; add this manually: ${line}"
    return 1
  }

  log "added ${INSTALL_DIR} to PATH in ${profile}"
  log "restart your shell or run: ${line}"
}

ensure_install_dir_on_path() {
  local profile line answer
  install_dir_on_path && return 0
  line="$(path_export_line)"
  profile="$(default_shell_profile || true)"
  if [[ -z "$profile" ]]; then
    warn "${INSTALL_DIR} is not on PATH"
    log "add this to your shell profile: ${line}"
    return 0
  fi
  if [[ -f "$profile" ]] && grep -Fq "$INSTALL_DIR" "$profile"; then
    warn "${INSTALL_DIR} is not on the current PATH, but it already appears in ${profile}"
    log "restart your shell or run: ${line}"
    return 0
  fi
  if ! can_prompt_on_tty; then
    warn "${INSTALL_DIR} is not on PATH"
    log "add this to ${profile}: ${line}"
    return 0
  fi
  printf 'anvil-installer: %s is not on PATH. Add it to %s? [Y/n] ' "$INSTALL_DIR" "$profile" >/dev/tty
  read -r answer </dev/tty || answer=""
  case "$answer" in
    "" | y | Y | yes | YES)
      append_install_dir_to_profile "$profile" "$line" || true
      ;;
    *)
      warn "${INSTALL_DIR} is not on PATH"
      log "add this to ${profile}: ${line}"
      ;;
  esac
}

install_binary() {
  local src="$1" dest="${INSTALL_DIR}/${BIN_NAME}"
  chmod 0755 "$src"
  strip_quarantine "$src"
  if [[ -w "$INSTALL_DIR" ]]; then
    install -m 0755 "$src" "$dest"
    strip_quarantine "$dest"
  else
    command -v sudo >/dev/null 2>&1 || die "cannot write ${INSTALL_DIR}; set INSTALL_DIR to a writable directory"
    sudo install -m 0755 "$src" "$dest"
    if [[ "$OS_FAMILY" == "macos" ]] && command -v xattr >/dev/null 2>&1; then
      sudo xattr -dr com.apple.quarantine "$dest" >/dev/null 2>&1 || true
    fi
  fi
  log "installed ${BIN_NAME} to ${dest}"
}

install_anvil() {
  local release_file="${TMP_DIR}/release.json" tag asset_url asset_name asset_file extract_dir src
  download_file "$(release_endpoint)" "$release_file"
  tag="$(release_tag "$release_file")"
  [[ -n "$tag" ]] || die "could not read ${OWNER}/${REPO} release metadata"
  asset_url="$(select_asset "$release_file" "$tag")"
  asset_name="${asset_url##*/}"
  asset_file="${TMP_DIR}/${asset_name}"
  log "downloading ${tag} (${asset_name})"
  download_file "$asset_url" "$asset_file"
  verify_checksum "$release_file" "$asset_name" "$asset_file"
  extract_dir="${TMP_DIR}/extract"
  mkdir -p "$extract_dir"
  unzip -q "$asset_file" -d "$extract_dir"
  strip_quarantine "$extract_dir"
  src="$(find "$extract_dir" -type f -name "$BIN_NAME" -print -quit)"
  [[ -n "$src" ]] || die "archive did not contain expected binary: ${BIN_NAME}"
  install_binary "$src"
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
  require_command curl
  require_command awk
  require_command grep
  require_command sed
  require_command unzip
  require_command install
  require_command find
  detect_platform
  ensure_install_dir
  TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/anvil-installer.XXXXXX")"
  trap cleanup EXIT
  log "installing for ${OS_FAMILY}/${ARCH} into ${INSTALL_DIR} (script ${SCRIPT_VERSION})"
  install_anvil
  ensure_install_dir_on_path
  log "done"
}

main "$@"
