"""Snapshot popular Hugging Face LLMs and parse each config.json into the fields the fit math needs."""
import datetime, json, math, os, re, sys, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

API = "https://huggingface.co/api/models"
EXP = "&".join(f"expand[]={k}" for k in ["safetensors", "downloads", "likes", "tags", "createdAt", "gated", "cardData", "pipeline_tag"])
SKIP_NAME = re.compile(r"(gguf|awq|gptq|-fp8|fp8-|-int4|-int8|bnb|4bit|8bit|mlx|exl2|nf4|-w4a16|-w8a8|nvfp4|mxfp4|-onnx|eagle|-draft)", re.I)


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "loadout-snapshot"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def listing():
    seen = {}
    for q in [
        f"pipeline_tag=text-generation&sort=downloads&direction=-1&limit=1000",
        f"pipeline_tag=text-generation&sort=likes&direction=-1&limit=600",
        f"pipeline_tag=image-text-to-text&sort=downloads&direction=-1&limit=400",
        f"pipeline_tag=image-text-to-text&sort=likes&direction=-1&limit=200",
    ]:
        for m in get(f"{API}?{q}&{EXP}", 120):
            seen.setdefault(m["id"], m)
    return list(seen.values())


def keep(m):
    st = m.get("safetensors") or {}
    if not st.get("total"):
        return False
    if SKIP_NAME.search(m["id"]):
        return False
    if any(t.startswith("base_model:quantized:") for t in m.get("tags", [])):
        return False
    return True


def params_of(st):
    p = st.get("parameters", {})
    total = st["total"]
    # packed 4-bit (MXFP4 in U8) holds two weights per byte
    return total


def fetch_cfg(mid):
    for repo in [mid, "unsloth/" + mid.split("/", 1)[1]]:
        try:
            return get(f"https://huggingface.co/{repo}/resolve/main/config.json", 30), repo != mid
        except Exception:
            continue
    return None, False


def parse(m, cfg):
    c = dict(cfg)
    if isinstance(cfg.get("text_config"), dict):
        c.update(cfg["text_config"])
    def g(*ks):
        v = next((c[k] for k in ks if c.get(k) is not None), None)
        return v[0] if isinstance(v, list) and v else v
    L = g("num_hidden_layers", "n_layer", "num_layers")
    heads = g("num_attention_heads", "n_head")
    hidden = g("hidden_size", "n_embd", "d_model")
    if not (L and heads and hidden):
        return None
    kvh = g("num_key_value_heads", "num_kv_heads", "n_head_kv") or (1 if c.get("multi_query") else heads)
    hd = g("head_dim") or hidden // heads
    ctx = g("max_position_embeddings", "n_positions", "seq_length", "max_seq_len") or 4096
    out = {"L": L, "kvh": kvh, "hd": hd, "ctx": int(ctx)}
    if c.get("kv_lora_rank"):
        out["mla"] = c["kv_lora_rank"] + (c.get("qk_rope_head_dim") or 64)
    # attention layer mix: full / sliding / no-KV (linear, mamba)
    full, slide, win = L, 0, c.get("sliding_window") or 0
    lt = c.get("layer_types")
    if isinstance(lt, list) and lt:
        full = sum(1 for t in lt if t in ("full_attention", "attention", "global"))
        slide = sum(1 for t in lt if t in ("sliding_attention", "local", "chunked_attention"))
    elif c.get("full_attention_interval"):
        full = L // c["full_attention_interval"]
    elif isinstance(c.get("layers_block_type"), list):
        full = sum(1 for t in c["layers_block_type"] if "attention" in t)
    elif isinstance(c.get("hybrid_override_pattern"), str):
        full = c["hybrid_override_pattern"].count("*")
    elif isinstance(c.get("attn_layer_indices"), list):
        full = len(c["attn_layer_indices"])
    elif c.get("sliding_window_pattern") and win:
        n = c["sliding_window_pattern"]
        slide = L - math.ceil(L / n); full = L - slide
    elif win and (c.get("use_sliding_window") is True or (c.get("model_type") == "mistral" and c.get("use_sliding_window") is not False)):
        slide, full = L, 0
    if slide and win:
        out["slide"] = [slide, win]
    if full + slide < L:
        out["noKv"] = L - full - slide
    # MoE active params
    E = g("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
    k = g("num_experts_per_tok", "experts_per_token", "moe_topk", "num_selected_experts")
    total = params_of(m["safetensors"])
    out["p"] = round(total / 1e9, 3)
    if E and k and E > 1:
        inter = g("moe_intermediate_size", "expert_intermediate_size") or c.get("intermediate_size") or 4 * hidden
        moe_layers = L - (c.get("first_k_dense_replace") or 0)
        expert_all = moe_layers * E * 3 * hidden * inter
        active = total - expert_all + moe_layers * k * 3 * hidden * inter
        active = min(total, max(active, 0.02 * total))
        out["a"] = round(active / 1e9, 2)
        out["E"] = [E, k]
    if isinstance(cfg.get("vision_config"), dict) or m.get("pipeline_tag") == "image-text-to-text":
        out["vision"] = 1
    return out


def main():
    models = [m for m in listing() if keep(m)]
    print("candidates", len(models), file=sys.stderr)
    with ThreadPoolExecutor(24) as ex:
        cfgs = list(ex.map(lambda m: fetch_cfg(m["id"]), models))
    rows = []
    for m, (cfg, mirrored) in zip(models, cfgs):
        if not cfg:
            continue
        try:
            a = parse(m, cfg)
        except Exception as e:
            print("parse fail", m["id"], e, file=sys.stderr); continue
        if not a or a["p"] < 0.1 or a["p"] > 1200:
            continue
        lic = (m.get("cardData") or {}).get("license") or next((t[8:] for t in m.get("tags", []) if t.startswith("license:")), "unknown")
        if isinstance(lic, list): lic = lic[0]
        tags = m.get("tags", [])
        a.update({
            "id": m["id"], "dl": m.get("downloads", 0), "likes": m.get("likes", 0),
            "date": (m.get("createdAt") or "")[:10], "lic": lic,
            "chat": 1 if "conversational" in tags else 0,
            "gated": 1 if m.get("gated") else 0,
        })
        rows.append(a)
    rows.sort(key=lambda r: -r["dl"])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "models.json")
    snap = {"generated": datetime.date.today().isoformat(), "models": rows}
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(snap, f, separators=(",", ":"))
    print("kept", len(rows), file=sys.stderr)


if __name__ == "__main__":
    main()
