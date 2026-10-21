# RIFT

A terminal agent harness whose base tool is the shell, plus skills on disk and MCP servers.

## Install

Linux and macOS:

```
curl -fsSL https://raw.githubusercontent.com/Lynx-2005/RIFT/main/install.sh | sh
```

Windows (PowerShell):

```
irm https://raw.githubusercontent.com/Lynx-2005/RIFT/main/install.ps1 | iex
```

Both scripts download the archive for your OS from the latest release and put
`rift` on PATH (`~/.local/bin` on Unix, `%LOCALAPPDATA%\rift\bin` on Windows).
Prefer to do it by hand? Take an archive from the releases page:

- Linux x86_64: `rift-x86_64-unknown-linux-gnu.tar.gz` (glibc) or `rift-x86_64-unknown-linux-musl.tar.gz` (static; runs on any distro)
- Windows x86_64: `rift-x86_64-pc-windows-msvc.zip` — unzip and run `rift.exe` from Windows Terminal or PowerShell
- macOS: `rift-aarch64-apple-darwin.tar.gz` (Apple Silicon) or `rift-x86_64-apple-darwin.tar.gz` (Intel)

Then unpack and run:

```
tar -xzf rift-<target>.tar.gz && ./rift     # Linux and macOS
```

Set a provider key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `TOKENHARBOR_API_KEY`)
or run `/provider` inside rift to pick one.

## Providers

`/provider` opens a live picker; `/provider <name>` switches directly and asks
for a key for built-in providers, every time, so an existing key can be
replaced (Enter saves, Esc keeps the current one). Keys are entered as dots, so
the paste length is visible without printing the secret. A key saved this way
is used ahead of the provider's environment variable; the variable is the
fallback when no key is stored. Configure as many providers as you like:
`/model` lists the models from every configured provider together, and picking
one switches provider automatically.

Custom endpoints (Ollama, LM Studio, vLLM, a third-party gateway) are saved like
built-ins:

```
/provider add ollama url=http://localhost:11434/v1 type=openai model=llama3.2
/provider add proxy url=https://gateway.example/v1 type=anthropic key=<key> models=claude-a,claude-b
/provider remove ollama
```

`type=` is `openai` (the OpenAI-compatible `/chat/completions` shape) or
`anthropic` (the `/v1/messages` shape). Keys are optional for custom providers,
so local servers work without a secret; `models=` seeds the model menu when an
endpoint has no `/models` list. `ollama`, `lmstudio`, `llamacpp` and `vllm`
default to their usual local address, so `url=` can be omitted for them:

```
/provider add ollama model=llama3.2
```

### Where settings live

Per project, rift reads `.rift/`: skills in `.rift/skills/` and MCP servers in
`.rift/mcp.json`.

Globally, settings live in `~/.rift/`:

- `~/.rift/providers.json` — local and custom providers
- `~/.rift/skills/` — skills available in every project
- `~/.rift/mcp.json` — MCP servers available in every project

The older XDG locations (`~/.config/rift/`) are still read as a fallback, so
existing setups keep working. State is separate and machine-local:
`~/.local/share/rift/` holds the session database and snapshots.

### Local models from a file

For a setup you can keep in an editor or a dotfiles repo, declare providers in
`~/.rift/providers.json`. The file is read live, so an edit is picked up
without restarting; `RIFT_PROVIDERS=/path/to/file.json` points somewhere else.

```json
{
  "local": {
    "type": "openai",
    "url": "http://127.0.0.1:8787/v1",
    "model": "local-model",
    "models": ["local-model"]
  }
}
```

`type` defaults to `openai`; `anthropic` is the other wire shape. `models`
accepts a list or an object keyed by id, and the first id becomes the default
when `model` is omitted. `api_key` is optional, and `api_key_env` names an
environment variable instead. File providers appear in `/provider` and `/model`
like saved ones, but are never copied into the database: the file stays the
source of truth. A malformed file is reported at startup, and one invalid entry
is skipped without hiding the rest.

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
