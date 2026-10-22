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
for a key for hosted providers, every time, so an existing key can be replaced
(Enter saves, Esc keeps the current one). Keys are entered as dots, so the
paste length is visible without printing the secret. A key saved this way is
used ahead of the provider's environment variable; the variable is the fallback
when no key is stored. `/model` lists the models from every configured provider
together, and picking one switches provider automatically.

If `OPENAI_BASE_URL` or `OPENAI_API_BASE` points at a non-OpenAI gateway,
rift ignores `OPENAI_API_KEY` for its OpenAI provider: that key belongs to
the gateway, and sending it to api.openai.com would fail. Configure the
gateway as a custom provider, or enter a real OpenAI key.

### Local model servers

Servers running on this machine are discovered automatically, the same way
OpenCode discovers its local providers. On startup and every 30 seconds rift
probes:

- **Ollama** at `http://127.0.0.1:11434` — `/api/tags`, then `/api/show` for
  each model
- **LM Studio** at `http://127.0.0.1:1234` — `/api/v1/models`
- **vLLM** at `http://127.0.0.1:8000` — `/health`, then `/v1/models`

Their models appear in `/model` with no registration and no key. The native
metadata also supplies the context window, and Ollama's capabilities keep
embedding-only models out of the menu. A server that stops keeps its last known
list until it answers again; a successful empty list clears it.

To use a server on another machine, point to it in `providers.json` and
discovery will use that address:

```json
{
  "ollama": {
    "url": "http://gpu-box:11434/v1"
  }
}
```

llama.cpp is not probed automatically; declare it as a custom provider below.

### Custom providers from a file

Registration is file-only: `~/.rift/providers.json` is the single source of
truth, and there is no `/provider add`. The file is read live, so an edit is
picked up without restarting; `RIFT_PROVIDERS=/path/to/file.json` points
somewhere else.

```json
{
  "gateway": {
    "type": "anthropic",
    "url": "https://gateway.example/v1",
    "api_key_env": "GATEWAY_API_KEY",
    "models": ["claude-a", "claude-b"]
  },
  "llamacpp": {
    "url": "http://localhost:8080/v1",
    "models": ["qwen3-coder"]
  }
}
```

`type` is `openai` (the OpenAI-compatible `/chat/completions` shape, the
default) or `anthropic` (the `/v1/messages` shape). `models` accepts a list or
an object keyed by id, and the first id becomes the default when `model` is
omitted. An explicit `models` list is authoritative: it seeds the menu and is
not overwritten by discovery. `api_key` is optional, and `api_key_env` names an
environment variable instead. `ollama`, `lmstudio`, `llamacpp` and `vllm`
default to their usual local address, so `url` can be omitted for them.

File providers appear in `/provider` and `/model` like built-ins, but are never
copied into the database: the file stays the source of truth. A malformed file
is reported at startup, and one invalid entry is skipped without hiding the
rest.

### Where settings live

Per project, rift reads `.rift/`: skills in `.rift/skills/` and MCP servers in
`.rift/mcp.json`.

Globally, settings live in `~/.rift/`:

- `~/.rift/providers.json` — custom providers and local-runtime overrides
- `~/.rift/skills/` — skills available in every project
- `~/.rift/mcp.json` — MCP servers available in every project

The older XDG locations (`~/.config/rift/`) are still read as a fallback, so
existing setups keep working. State is separate and machine-local:
`~/.local/share/rift/` holds the session database and snapshots.

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
