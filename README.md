# Loadout

**What can my rig run?**

Pick your GPUs and Loadout ranks every popular Hugging Face model by how well it fits: which run well, which only fit if they're slow, which need RAM offload, and which won't fit at all. For anything that doesn't fit, it ranks what it would take — free settings changes first, then offload, then hardware to buy — with the arithmetic shown for every number.

**Live:** https://blaine-hiers.github.io/Loadout/

It's the GPU-first companion to [Headroom](https://github.com/blaine-hiers/Headroom). Headroom starts from one model and sizes it in depth. Loadout starts from the hardware you own and shows the whole list.

> **Estimates, not benchmarks.** Real runtimes add activation memory, allocator fragmentation and their own KV rounding. Use Loadout to shortlist, then confirm on the real stack. Every row's **Why?** panel shows its formulas with your numbers filled in.

## What it does

- **Ranked model list.** About 700 text and vision models from the Hugging Face Hub, best fit first. Also sorts by fastest, biggest that runs, most VRAM to spare, most downloaded or newest.
- **Search as you type**, with filters inside the query: `qwen coder`, `org:google`, `<35b`, `>100b moe`, `ctx>200k`, `lic:apache`, `vision`, `base`. Picking a suggestion pins that model to the top.
- **Hardware:** 60 GPUs searchable by name, VRAM or vendor (GeForce, workstation, Radeon, Arc, Apple silicon, unified-memory PCs, datacenter), plus a custom card. Mix cards, up to 8.
- **Every precision:** fp32/fp16/bf16, fp8/int8/q8_0, all GGUF k-quants and i-quants, AWQ, GPTQ, MXFP4, NVFP4, NF4 and EXL2 2.5–8 bpw. **Auto** picks the best precision that fits, down to a floor you choose. KV cache in fp32, fp16, bf16, fp8, q8_0, q5_1 or q4_0.
- **Workload sliders:** users (1–256), context (1K–1M), system RAM, minimum acceptable tok/s, usable VRAM %.
- **Runtime options:** layer split (llama.cpp, Ollama) or tensor parallel (vLLM, SGLang), RAM speed for offload, raising the macOS GPU memory limit, offload on/off.
- **What would it take?** For any model that doesn't run well: shorter context, fewer users, a smaller KV cache, a lower quant, offload, more of your card, or the smallest hardware swap that fits.
- **Biggest unlock for your money:** which upgrade adds the most models that run well.
- **GPU pages (preview):** a page per GPU, per model, and per GPU × model pair, answering the question people search for ("can a 4090 run Llama 3.3 70B?").

## Run it locally

No build step and no dependencies. The page loads `data/models.json`, so serve the folder over HTTP rather than opening the file directly:

```sh
python -m http.server 8000
# then open http://localhost:8000/
```

## The model snapshot

There's no backend, so the model list is a snapshot rather than a live query.

```sh
python scripts/hf_snapshot.py   # standard library only; writes data/models.json
```

It pulls the top text-generation and image-text-to-text models by downloads and by likes, drops pre-quantized repos (GGUF, AWQ, GPTQ, FP8, MLX…) since precision is chosen in the app, then fetches each repo's `config.json` and keeps only what the math needs. Gated repos (Meta, Google) fall back to the matching `unsloth/` mirror for their config. Repos with no standard `config.json` are skipped.

The **Refresh model snapshot** workflow re-runs it every Monday and commits only when the model list actually changed, which redeploys the site.

## The math

All sizes are decimal GB.

- **Weights** = `params × bits per weight ÷ 8`
- **KV per token per layer** = `2 × KV heads × head dim × KV bytes` (MLA models: `(kv_lora_rank + qk_rope_head_dim) × KV bytes`, no ×2)
- **KV per request** = `full layers × context + sliding layers × min(context, window)`, times the per-layer figure. Linear-attention and Mamba layers hold no KV.
- **Total** = weights + users × KV per request + runtime overhead
- **Usable VRAM** = card VRAM × usable % (default 95%); unified memory uses its allocatable share (Apple 75%, or 90% with the limit raised)
- **Speed** ≈ `0.75 × memory bandwidth ÷ (active weight bytes + KV bytes)` per token. Layer split runs cards in sequence, so it adds room but not speed; tensor parallel across 2/4/8 matching cards sums bandwidth at 85% efficiency.
- **Offload speed** splits the same read between GPU bandwidth and system RAM bandwidth.
- **MoE active params** are estimated from the expert config: total minus all expert weights plus the routed top-k.
- **Fit score** = `10·log₁₀(capability × quant quality) + 1.2·log₁₀(downloads)`, where capability is params, or `√(total × active)` for MoE, and quant quality falls from 1.0 at 8 bits to about 0.97 at q4_k_m and 0.82 at iq3_xxs.

## Layout

| Path | What |
|---|---|
| `index.html` | The whole app: markup, styles and script |
| `data/models.json` | Model snapshot (generated) |
| `scripts/hf_snapshot.py` | Builds the snapshot from the Hugging Face API |
| `.github/workflows/pages.yml` | Deploys to GitHub Pages |
| `.github/workflows/snapshot.yml` | Weekly snapshot refresh |
