# RIFT

A terminal agent harness whose base tool is the shell, plus skills on disk and MCP servers.

RIFT is a single native binary for hosted and local models. It keeps its
built-in tool surface small and can talk to common local model servers directly.

On launch the header shows the provider, model and the number of enabled MCP
servers on the right.

## Low overhead, local-friendly

RIFT is built to stay small:

- **One native binary.** No Node.js, Python, Electron, or browser runtime is
  needed. There is no separate background daemon; refreshes run on threads
  while RIFT is open.
- **Two built-in tools.** The model gets `shell` and `skill`. MCP tools are
  added only from servers you configure.
- **Skills load on demand.** Only a skill's name and description are kept in
  the prompt; the full `SKILL.md` is read when the model calls the `skill`
  tool.
- **Common local runtimes are supported.** Ollama, LM Studio, and vLLM are
  checked at their usual local addresses, and llama.cpp or another
  OpenAI-compatible server can be added in `providers.json`. Local endpoints
  normally need no API key.
- **File-based configuration, local state.** Settings live under `.rift/` and
  `~/.rift/`; sessions and snapshots stay in the local data directory.

The smaller prompt surface leaves more room for the task in a model's context
window, which helps with local models that have limited context. It does not
make inference itself cheap: speed, memory use, and tool-calling quality depend
on the model, runtime, and hardware. Optional MCP servers and large skills can
also add startup work and prompt tokens, and per-turn snapshots for undo do
filesystem work within a size budget.

## Install

Linux and macOS:

```
curl -fsSL https://raw.githubusercontent.com/Lynx-2005/RIFT/main/install.sh | sh
```

Windows (PowerShell):

```
irm https://raw.githubusercontent.com/Lynx-2005/RIFT/main/install.ps1 | iex
```

The installers download the latest release for the platform they support.
On Unix the binary goes to `~/.local/bin`; if that directory is not on PATH,
the script prints the line to add. On Windows it goes to
`%LOCALAPPDATA%\rift\bin`, and the installer adds that directory to your user
PATH. Supported install targets are Linux x86_64, Windows x86_64, and macOS on
Intel or Apple Silicon.

Prefer to do it by hand? Take an archive from the releases page:

- Linux x86_64: `rift-x86_64-unknown-linux-gnu.tar.gz` (glibc) or `rift-x86_64-unknown-linux-musl.tar.gz` (static; avoids glibc version dependencies)
- Windows x86_64: `rift-x86_64-pc-windows-msvc.zip` — unzip and run `rift.exe` from Windows Terminal or PowerShell
- macOS: `rift-aarch64-apple-darwin.tar.gz` (Apple Silicon) or `rift-x86_64-apple-darwin.tar.gz` (Intel)

Then unpack and run:

```
tar -xzf rift-<target>.tar.gz && ./rift     # Linux and macOS
```

Other targets can be built from source; see below.

For a hosted provider, set a key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `TOKENHARBOR_API_KEY`)
or run `/provider` inside rift to pick one. Local runtimes normally need no
key: run `/provider` and select the discovered server, or start with
`--provider ollama`, `--provider lmstudio`, or `--provider vllm`.

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
the gateway, and sending it to api.openai.com can fail. Configure the gateway
as a custom provider, or enter a real OpenAI key.

### Local model servers

Ollama, LM Studio, and vLLM are discovered at their usual local addresses. On
startup and every 30 seconds rift probes:

- **Ollama** at `http://127.0.0.1:11434` — `/api/tags`, then `/api/show` for
  each model
- **LM Studio** at `http://127.0.0.1:1234` — `/api/v1/models`
- **vLLM** at `http://127.0.0.1:8000` — `/health`, then `/v1/models`

The discovered models appear in `/model` without registration; the default
local addresses normally need no key. Where the runtime exposes it, native
metadata also supplies the context window, and Ollama's capabilities keep
embedding-only models out of the menu. If a server stops, its last known list
stays until the server answers again; a successful empty list clears it.

A local model still needs tool-calling support in whatever runtime serves it;
RIFT does not add that capability to a model that lacks it.

RIFT is the client only: it does not download, start, or manage model servers.
Start Ollama, LM Studio, vLLM, or another server separately.

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

Custom providers are configured in `~/.rift/providers.json`; there is no
`/provider add`. The file is read at runtime, so an edit takes effect without
restarting. `RIFT_PROVIDERS=/path/to/file.json` points somewhere else.

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

File providers appear in `/provider` and `/model` like built-ins, but are kept
in the file rather than copied into the database. A malformed file is reported
at startup, and one invalid entry is skipped without hiding the rest.

### Declaring thinking support

Local runtimes do not all expose a standard reasoning-capability endpoint. You
can declare support in `providers.json`; the block applies to every model the
entry serves:

```json
{
  "vllm": {
    "url": "http://localhost:8000/v1",
    "thinking": {
      "supported": true,
      "effort": ["none", "low", "medium", "high"]
    }
  },
  "llamacpp": {
    "url": "http://localhost:8080/v1",
    "thinking": {
      "supported": true,
      "toggle": true
    }
  }
}
```

A model entry can override the provider-wide block:

```json
"models": {
  "qwen3-coder": {
    "thinking": {
      "supported": true,
      "effort": ["none", "low", "medium", "high"]
    }
  }
}
```

- `supported`: `false` means this model explicitly cannot reason. Omit the
  whole block to leave the model unknown; it then sends no reasoning fields.
- `effort`: the levels this endpoint accepts. Include `"none"` if `off` should
  explicitly disable thinking. On OpenAI-compatible endpoints these are sent as
  `reasoning_effort` values. Support and exact behavior depend on the runtime
  and model; check the server's documentation for the values it honors.
- `toggle`: declares an on/off endpoint with no graded effort. On an
  OpenAI-compatible endpoint rift sends `none` for off and `high` for on.
- `budget`: `{"min": ..., "max": ...}` for Anthropic-wire budget control (also
  used by OpenRouter's unified reasoning object). OpenAI-compatible local
  servers use `effort` or `toggle`.

Explicit configuration wins over discovery: model entry, then provider entry,
then Ollama's `/api/show` metadata, then models.dev, then unknown. The file is
read at runtime, so an edit takes effect on the next request without restarting.

### Where settings live

Per project, rift reads `.rift/`: skills in `.rift/skills/` and MCP servers in
`.rift/mcp.json`.

Globally, settings live in `~/.rift/`:

- `~/.rift/providers.json` — custom providers and local-runtime overrides
- `~/.rift/skills/` — skills available in every project
- `~/.rift/mcp.json` — MCP servers available in every project

The older XDG locations (`~/.config/rift/`) are still read as a fallback for
existing setups. Session state is machine-local: `~/.local/share/rift/` holds
the session database and snapshots.

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

macOS builds run on macOS runners in CI, because cross-compiling from Linux
requires Apple's SDK.

## Layout

- `shell-harness/` — the Rust crate; the binary is `rift`
- `.github/workflows/build.yml` — tests and release binaries for Linux, Windows and macOS
