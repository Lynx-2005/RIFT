# RIFT

A terminal agent harness whose base tool is the shell, plus skills on disk and MCP servers.

## Install

Download the archive for your platform from the latest release:

- Linux x86_64: `rift-x86_64-unknown-linux-gnu.tar.gz` (glibc) or `rift-x86_64-unknown-linux-musl.tar.gz` (static; runs on any distro)
- Windows x86_64: `rift-x86_64-pc-windows-msvc.zip` — unzip and run `rift.exe` from Windows Terminal or PowerShell
- macOS: `rift-aarch64-apple-darwin.tar.gz` (Apple Silicon) or `rift-x86_64-apple-darwin.tar.gz` (Intel)

Then unpack and run:

```
tar -xzf rift-<target>.tar.gz && ./rift     # Linux and macOS
```

Set a provider key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `TOKENHARBOR_API_KEY`)
or run `/model` inside rift to pick one.

## Build from source

```
cd shell-harness
cargo build --release
```

Cross-build the Windows binary from Linux:

```
rustup target add x86_64-pc-windows-gnu
sudo apt install mingw-w64
cargo build --release --target x86_64-pc-windows-gnu
```

macOS cannot be cross-built from Linux (needs Apple's SDK); CI builds it on macOS runners.

## Layout

- `shell-harness/` — the Rust crate; the binary is `rift`
- `.github/workflows/build.yml` — tests and release binaries for Linux, Windows and macOS
