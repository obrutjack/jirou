# Jirou Development Guide

Jirou (次郎) is a fork of [KiroCrew](https://github.com/kirodotdev/KiroCrew) that adds
local LLM support (LM Studio, Ollama, and any OpenAI-compatible endpoint).

## Repository layout

| Location | What it is |
|---|---|
| `~/_jackliu/app_project/jirou/` | This repo — KiroCrew fork |
| `~/_jackliu/app_project/jirou-poc/` | Old Electron + FastAPI proof-of-concept (archived) |

## Git remotes

```
origin   → https://github.com/obrutjack/jirou       ← push your changes here
upstream → https://github.com/kirodotdev/KiroCrew   ← pull KiroCrew updates from here
```

## Syncing with upstream KiroCrew

When KiroCrew ships a new version:

```bash
cd ~/_jackliu/app_project/jirou
git fetch upstream
git merge upstream/main      # fast-forward when there are no conflicts
# or
git rebase upstream/main     # cleaner linear history
```

Keep your changes in as few new files as possible (e.g. `src/kiro_crew/providers/openai_compatible.py`)
to minimise merge conflicts on each sync.

## Goal

Add an `openai-compatible` provider to `src/kiro_crew/providers/` so KiroCrew can use:
- **LM Studio** running locally at `http://localhost:1234/v1`
- **Ollama** at `http://localhost:11434/v1`
- Any BYOK endpoint (Anthropic, OpenAI, OpenRouter, etc.)

See `obrutjack/jirou` issues for the implementation plan.

## Benchmark reference (from Jirou POC)

Qwen3 14B via LM Studio on MacBook Air M4 32GB:
- Single-turn tool call: **96% success rate**
- Multi-step agent loop: **100% success rate**
- Average latency: **4.0s/call** (with `/no_think` system prompt)

The `/no_think` system prompt is the key unlock — disables Qwen3's thinking mode,
giving ~10x latency reduction with no meaningful accuracy loss on tool dispatch.
