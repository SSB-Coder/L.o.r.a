#!/bin/sh
# Install LORA on macOS or Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/SSB-Coder/L.o.r.a/master/scripts/install.sh | sh
#   curl -fsSL .../install.sh | sh -s -- nightly     # newest prerelease
#   curl -fsSL .../install.sh | sh -s -- v0.2.1      # a specific release
#
# Installs to ~/.lora/bin (override with LORA_INSTALL_DIR). Needs curl, tar, and sha256sum or shasum.
set -eu

REPO="${LORA_REPO:-SSB-Coder/L.o.r.a}"
CHANNEL="${1:-stable}"
DEST="${LORA_INSTALL_DIR:-$HOME/.lora/bin}"
API="https://api.github.com/repos/$REPO"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) target=linux-x64 ;;
  Linux-aarch64 | Linux-arm64) target=linux-arm64 ;;
  Darwin-arm64) target=darwin-arm64 ;;
  Darwin-x86_64) target=darwin-x64 ;;
  *) echo "No LORA build for $(uname -s) $(uname -m)." >&2; exit 1 ;;
esac

case "$CHANNEL" in
  stable) tag="$(curl -fsSL "$API/releases/latest" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n 1)" ;;
  nightly)
    # Releases come newest first; take the first one marked as a prerelease.
    tag="$(curl -fsSL "$API/releases?per_page=30" | awk -F'"' '/"tag_name":/ { t = $4 } /"prerelease": *true/ { print t; exit }')"
    ;;
  v*) tag="$CHANNEL" ;;
  *) echo "Channel must be stable, nightly, or a tag like v0.2.1 (got $CHANNEL)." >&2; exit 1 ;;
esac
[ -n "$tag" ] || { echo "No $CHANNEL release of LORA found." >&2; exit 1; }

version="${tag#v}"
name="lora-$version-$target"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Downloading LORA $version ($target)…"
base="https://github.com/$REPO/releases/download/$tag"
curl -fsSL -o "$tmp/$name.tar.gz" "$base/$name.tar.gz"
curl -fsSL -o "$tmp/SHA256SUMS" "$base/SHA256SUMS"

expected="$(grep " $name.tar.gz\$" "$tmp/SHA256SUMS" | cut -d' ' -f1)"
if command -v sha256sum >/dev/null; then actual="$(sha256sum "$tmp/$name.tar.gz" | cut -d' ' -f1)"
else actual="$(shasum -a 256 "$tmp/$name.tar.gz" | cut -d' ' -f1)"; fi
[ -n "$expected" ] && [ "$expected" = "$actual" ] || { echo "Checksum mismatch for $name.tar.gz; not installing." >&2; exit 1; }

tar -xzf "$tmp/$name.tar.gz" -C "$tmp"
mkdir -p "$DEST"
install -m 0755 "$tmp/$name/lora" "$DEST/lora"
echo "Installed $("$DEST/lora" --version) to $DEST/lora"

case ":$PATH:" in
  *":$DEST:"*) ;;
  *) echo "Add it to your PATH:  export PATH=\"$DEST:\$PATH\"" ;;
esac
echo "Start Ollama, pull a model (e.g. ollama pull qwen3:8b), then run: lora"
