# Jirou Development Guide

Jirou (次郎) is a fork of [KiroCrew](https://github.com/kirodotdev/KiroCrew) that adds
local LLM support. Instead of using Amazon's kiro-cli backend, it connects to any
OpenAI-compatible API endpoint — most commonly **LM Studio** running **Qwen3 14B** locally.

**No cloud API key needed. All inference runs on your own hardware.**

---

## What this fork adds

| Component | Location | Description |
|---|---|---|
| `providers/openai_compatible.py` | New file | LLMProvider implementation for local models |
| `config/sections.py` | +1 line | Adds `"local-llm"` to provider enum |
| `config/loader.py` | +12 lines | Routes to new provider when configured |

---

## Repository layout

| Location | What it is |
|---|---|
| `~/jirou/` | This repo — KiroCrew fork |
| `~/jirou-poc/` | Old Electron + FastAPI proof-of-concept (archived) |
| `~/.kiro/jirou/` | Isolated KiroCrew data home for this fork |

---

## Prerequisites

- [LM Studio](https://lmstudio.ai) with **Qwen3 14B** loaded
- Python 3.12+ and Node.js 22+ (for rebuilding the frontend)
- macOS (tested on MacBook Air M4 32GB)

---

## LM Studio setup (critical for performance)

### 1. Download and load the model

In LM Studio, search for `Qwen3 14B` and download the GGUF Q4_K_M version
(`lmstudio-community/Qwen3-14B-GGUF`).

### 2. Set context length to 4096 ⚠️

**This is the most important performance setting.**

When loading Qwen3 14B, set **Context Length = 4096** (NOT the default 32768).

Why: LM Studio pre-allocates KV cache for the full context length. At 32768,
prefill takes 5-15 seconds even for short prompts. At 4096, prefill drops to
**0.9-1.5 seconds**.

Our context splitting code ensures we never exceed 4096 tokens, so 4096 is safe.

### 3. Enable the Local Server

In LM Studio → Local Server → Load Qwen3 14B → Start Server

Default port: `1234` (matches our default `LOCAL_LLM_BASE_URL`)

---

## Running the fork

### Step 1: Set up Python venv

```bash
cd ~/jirou
python3 -m venv .venv
source .venv/bin/activate
pip install openai
pip install -e ".[dev]"
```

### Step 2: Create isolated KiroCrew data home

```bash
mkdir -p ~/.kiro/jirou
```

Create `~/.kiro/jirou/config.json`:
```json
{
  "agent": {
    "provider": "local-llm",
    "approval_mode": "auto",
    "sandbox": "off",
    "tool_search": false,
    "log_level": "WARNING"
  },
  "dashboard": {
    "bot_name": "KiroCrew (local)"
  }
}
```

Create `~/.kiro/jirou/.env`:
```
LOCAL_LLM_BASE_URL=http://localhost:1234/v1
LOCAL_LLM_API_KEY=lm-studio
LOCAL_LLM_MODEL=qwen/qwen3-14b
LOCAL_LLM_NO_THINK=true
LOCAL_LLM_CONTEXT_WINDOW=4096
LOCAL_LLM_MAX_CONTEXT_CHARS=2000
LOCAL_LLM_SYSTEM_PROMPT=You are a helpful local AI assistant. Answer concisely. Use the available tools when needed to complete the user's request.
```

### Step 3: Build the frontend

```bash
cd ~/jirou/website
npm ci
npm run build
cd ..
cp -r website/dist src/kiro_crew/static/
```

### Step 4: Start the gateway

```bash
cd ~/jirou
KIROCREW_HOME=~/.kiro/jirou KIROCREW_PORT=5477 .venv/bin/kirocrew gateway
```

Open [http://localhost:5477](http://localhost:5477) in your browser.

---

## Git remotes

```
origin   → https://github.com/obrutjack/jirou       ← push your changes here
upstream → https://github.com/kirodotdev/KiroCrew   ← pull KiroCrew updates
```

### Syncing with upstream KiroCrew

```bash
cd ~/jirou
git fetch upstream
git merge upstream/main
```

Keep your changes in new files (`providers/openai_compatible.py`) and minimal
edits to existing files to reduce merge conflicts.

---

## Performance benchmarks (2026-09-19, M4 32GB)

| Metric | Value |
|--------|-------|
| LM Studio raw baseline (curl) | 3.3s |
| First question latency (new session) | ~34s (KiroCrew session warm-up, one-time) |
| Subsequent questions (steady state) | **0.9-1.5s** ✅ |
| Prompt size after context trimming | ~548-700 tokens |

### Key performance settings

| Setting | Value | Impact |
|---------|-------|--------|
| LM Studio context length | **4096** | Most important — prefill 10x faster than 32768 |
| LOCAL_LLM_MAX_CONTEXT_CHARS | 2000 | Trims KiroCrew's injected context to ~500 tokens |
| LOCAL_LLM_NO_THINK | true | Disables Qwen3 thinking mode (10x generation speedup) |
| tool_search | false | Removes skills index from injected context |

---

## OpenAI-compatible provider env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_LLM_BASE_URL` | `http://localhost:1234/v1` | LM Studio / Ollama endpoint |
| `LOCAL_LLM_API_KEY` | `lm-studio` | API key (any string for local) |
| `LOCAL_LLM_MODEL` | `qwen/qwen3-14b` | Model identifier |
| `LOCAL_LLM_NO_THINK` | `true` | Inject `/no_think` to disable Qwen3 thinking mode |
| `LOCAL_LLM_CONTEXT_WINDOW` | `32768` | Report to KiroCrew for context budget scaling |
| `LOCAL_LLM_MAX_CONTEXT_CHARS` | `2000` | Max chars to keep from KiroCrew's injected context |
| `LOCAL_LLM_SYSTEM_PROMPT` | (built-in) | Override system prompt entirely |
