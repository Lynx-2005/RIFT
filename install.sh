#!/bin/sh
# Install rift: download the release archive for this OS and put the binary on
# PATH. Usage:
#   curl -fsSL https://raw.githubusercontent.com/Lynx-2005/RIFT/main/install.sh | sh
set -eu

repo="Lynx-2005/RIFT"
bin_dir="${RIFT_BIN_DIR:-$HOME/.local/bin}"

os=$(uname -s)
arch=$(uname -m)
case "$os-$arch" in
  # Static musl build: runs on any x86_64 Linux, no glibc version to match.
  Linux-x86_64)  target="x86_64-unknown-linux-musl" ;;
  Darwin-arm64)  target="aarch64-apple-darwin" ;;
  Darwin-x86_64) target="x86_64-apple-darwin" ;;
  *)
    echo "rift: no build published for $os $arch" >&2
    exit 1
    ;;
esac

url="https://github.com/$repo/releases/latest/download/rift-$target.tar.gz"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "rift: downloading $target"
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$url" | tar -xz -C "$tmp"
else
  wget -qO- "$url" | tar -xz -C "$tmp"
fi

mkdir -p "$bin_dir"
install -m 755 "$tmp/rift" "$bin_dir/rift"
echo "rift: installed $bin_dir/rift"

case ":$PATH:" in
  *":$bin_dir:"*) ;;
  *) echo "rift: add it to PATH:  export PATH=\"$bin_dir:\$PATH\"" ;;
esac
