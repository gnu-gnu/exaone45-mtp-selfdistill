"""Retrain the EXAONE-4.5 MTP head on self-generated data with the 33B target frozen.

MTP step (matches vLLM Exaone4_5MultiTokenPredictor):
  m_i = norm(layer(fc([norm_e(emb(x_{i+1})); norm_h(h_i)])))
  draft logits for x_{i+2} = lm_head(m_i)
Loss: KL(p_target(. | x_<=i+1) || p_mtp) on response positions.
Metric `acc`: E[p_target(argmax p_mtp)], the expected acceptance of a greedy draft
token under sampling; `top1`: argmax agreement (greedy decoding acceptance).
"""
import argparse, json, math, os, random, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Exaone4Config
from transformers.models.exaone4.modeling_exaone4 import (
    Exaone4DecoderLayer,
    Exaone4ForCausalLM,
    Exaone4RMSNorm,
    Exaone4RotaryEmbedding,
    apply_rotary_pos_emb,
)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/model")
ap.add_argument("--data", default="/data/gen.jsonl")
ap.add_argument("--out", default="/ckpt/run")
ap.add_argument("--max-len", type=int, default=4096)
ap.add_argument("--tokens-per-step", type=int, default=16384)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--warmup", type=int, default=30)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--eval-every", type=int, default=100)
ap.add_argument("--eval-max", type=int, default=263)
ap.add_argument("--chunk", type=int, default=1024)
ap.add_argument("--eval-only", action="store_true")
ap.add_argument("--init", default="", help="optional mtp safetensors to start from")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--K", type=int, default=3, help="recursive MTP steps trained (vLLM num_speculative_tokens)")
ap.add_argument("--beta", type=float, default=0.6, help="per-step loss weight decay")
ap.add_argument("--think-weight", type=float, default=1.0, help="loss multiplier for thinking-mode records")
ap.add_argument("--l2-init", type=float, default=0.0, help="L2 pull toward the initial MTP weights")
ap.add_argument("--loss", choices=["kl", "lk"], default="kl", help="lk: LK hybrid lam*KL + (1-lam)*TV, lam=exp(-eta*alpha)")
ap.add_argument("--eta", type=float, default=3.0)
ap.add_argument("--eval-by-src", action="store_true", help="split eval metrics by prompt source")
ap.add_argument("--mtp-src", default="", help="checkpoint dir for the initial mtp.* weights (default --model)")
ap.add_argument("--mem-frac", type=float, default=0.55)
ap.add_argument("--diag", type=int, default=0)
ap.add_argument("--simulate", default="", help="jsonl of greedy vLLM outputs; replay greedy spec decoding offline")
ap.add_argument("--regime", type=int, default=0, help="max answer tokens for the regime analysis")
args = ap.parse_args()
torch.manual_seed(args.seed)
random.seed(args.seed)
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(args.mem_frac)
os.makedirs(args.out, exist_ok=True)
logf = open(f"{args.out}/log.jsonl", "a")


def log(**kw):
    kw["t"] = round(time.time())
    print(json.dumps(kw), flush=True)
    logf.write(json.dumps(kw) + "\n")
    logf.flush()


tc = json.load(open(f"{args.model}/config.json"))["text_config"]
cfg = Exaone4Config(**{k: v for k, v in tc.items() if k not in ("architectures", "model_type", "_num_mtp_layers")})
cfg._attn_implementation = "sdpa"
idx = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]


def drop_page_cache(path):
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def load(keys_map, model_dir=None):
    model_dir = model_dir or args.model
    widx = json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]
    out, by_file = {}, {}
    for src, dst in keys_map.items():
        by_file.setdefault(widx[src], []).append((src, dst))
    for f, kv in by_file.items():
        with safe_open(f"{model_dir}/{f}", "pt", device="cuda") as fh:
            for src, dst in kv:
                out[dst] = fh.get_tensor(src)
        torch.cuda.synchronize()
        drop_page_cache(f"{model_dir}/{f}")
    return out


class FP8Linear(nn.Module):
    """compressed-tensors FP8 W8A8: per-channel weight scale, dynamic per-token activations."""

    scaled_mm = True

    def __init__(self, n_out, n_in):
        super().__init__()
        self.register_buffer("weight", torch.empty(n_out, n_in, dtype=torch.float8_e4m3fn))
        self.register_buffer("weight_scale", torch.empty(n_out, 1, dtype=torch.bfloat16))

    def forward(self, x):
        shp = x.shape
        x = x.reshape(-1, shp[-1]).to(torch.bfloat16)
        ws = self.weight_scale.float().view(1, -1)
        if FP8Linear.scaled_mm:
            xs = x.abs().amax(-1, keepdim=True).float().clamp(min=1e-12) / 448.0
            xq = (x.float() / xs).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            y = torch._scaled_mm(xq, self.weight.t(), scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16)
        else:
            y = F.linear(x, (self.weight.to(torch.bfloat16) * ws.t()).to(torch.bfloat16))
        return y.view(*shp[:-1], -1)


t0 = time.time()
with torch.device("meta"):
    tgt = Exaone4ForCausalLM(cfg)
tmap = {k: k.replace("model.language_model.", "model.") for k in idx if k.startswith("model.language_model.")}
tmap["lm_head.weight"] = "lm_head.weight"
if any(k.endswith("weight_scale") for k in tmap):
    for layer in tgt.model.layers:
        for name in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(layer.mlp, name)
            with torch.device("meta"):
                setattr(layer.mlp, name, FP8Linear(lin.out_features, lin.in_features))
tgt.load_state_dict(load(tmap), assign=True, strict=True)
tgt.model.rotary_emb = Exaone4RotaryEmbedding(cfg, device=dev)
tgt.eval().requires_grad_(False)
log(event="target_loaded", sec=round(time.time() - t0), gb=round(torch.cuda.memory_allocated() / 2**30, 1))
if any(isinstance(m, FP8Linear) for m in tgt.modules()):
    probe = tgt.model.layers[0].mlp.up_proj
    xt = torch.randn(13, probe.weight.shape[1], device=dev, dtype=torch.bfloat16)
    FP8Linear.scaled_mm = False
    ref = probe(xt).float()
    try:
        FP8Linear.scaled_mm = True
        err = ((probe(xt).float() - ref).norm() / ref.norm()).item()
        FP8Linear.scaled_mm = err < 0.05
    except RuntimeError as e:
        err, FP8Linear.scaled_mm = repr(e)[:200], False
    log(event="fp8_path", scaled_mm=FP8Linear.scaled_mm, rel_err=err)


class MTP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.layers = nn.ModuleList([Exaone4DecoderLayer(cfg, layer_idx=0)])
        self.norm = Exaone4RMSNorm(H, eps=eps)
        self.pre_fc_norm_hidden = Exaone4RMSNorm(H, eps=eps)
        self.pre_fc_norm_embedding = Exaone4RMSNorm(H, eps=eps)

    def forward(self, emb, h, pos_emb):
        x = self.fc(torch.cat([self.pre_fc_norm_embedding(emb), self.pre_fc_norm_hidden(h)], -1))
        x = self.layers[0](x, attention_mask=None, position_embeddings=pos_emb)
        if isinstance(x, tuple):
            x = x[0]
        return self.norm(x)

    def _qkv(self, x, pos_emb):
        a = self.layers[0].self_attn
        shp = (x.shape[0], -1, a.head_dim)
        q = a.q_norm(a.q_proj(x).view(shp)).transpose(0, 1)
        k = a.k_norm(a.k_proj(x).view(shp)).transpose(0, 1)
        v = a.v_proj(x).view(shp).transpose(0, 1)
        q, k = apply_rotary_pos_emb(q[None], k[None], *pos_emb)
        rep = a.num_key_value_groups
        return q[0], k[0].repeat_interleave(rep, 0), v.repeat_interleave(rep, 0)

    def _finish(self, x, attn):
        layer = self.layers[0]
        o = layer.self_attn.o_proj(attn.transpose(0, 1).reshape(x.shape[0], -1))
        h = x + layer.post_attention_layernorm(o)
        return self.norm(h + layer.post_feedforward_layernorm(layer.mlp(h)))

    def chain(self, emb, H, rotary, K, chunk=512):
        """Outputs of K recursive draft steps over a whole sequence, attending like vLLM drafting.

        Step k at root i reads (step k-1 output at i, emb(x_{i+k})) at position i+k-1. Its attention sees
        step-1 keys at roots <= i (the drafter KV cache of verified tokens) plus the keys of steps 2..k at
        root i (the draft's own earlier steps).
        """
        L = emb.shape[0]
        scale = self.layers[0].self_attn.scaling
        outs, diag, h = [], [], H
        for k in range(1, K + 1):
            n = L - k - 1
            if n <= 0:
                break
            x = self.fc(torch.cat([self.pre_fc_norm_embedding(emb[k : k + n]), self.pre_fc_norm_hidden(h[:n])], -1))
            q, kk, vv = self._qkv(x, rotary(k - 1, n))
            if k == 1:
                k1, v1 = kk, vv
                attn = F.scaled_dot_product_attention(q[None], k1[None], v1[None], is_causal=True, scale=scale)[0]
            else:
                diag.append((kk, vv))
                parts = []
                for s0 in range(0, n, chunk):
                    e0 = min(s0 + chunk, n)
                    qs = q[:, s0:e0]
                    sc = (qs @ k1[:, :e0].transpose(1, 2)).float() * scale
                    future = torch.arange(e0, device=q.device)[None] > torch.arange(s0, e0, device=q.device)[:, None]
                    sc = sc.masked_fill(future, float("-inf"))
                    ds = torch.stack([(qs * dk[:, s0:e0]).sum(-1) for dk, _ in diag], -1).float() * scale
                    w = torch.softmax(torch.cat([sc, ds], -1), -1).to(q.dtype)
                    o = w[..., :e0] @ v1[:, :e0]
                    for m, (_, dv) in enumerate(diag):
                        o = o + w[..., e0 + m, None] * dv[:, s0:e0]
                    parts.append(o)
                attn = torch.cat(parts, 1)
            h = self._finish(x, attn)
            outs.append(h)
        return outs


assert cfg.layer_types[0] == "sliding_attention"
mtp = MTP(cfg).to(dev, torch.float32)
mtp_dir = args.mtp_src or args.model
midx = json.load(open(f"{mtp_dir}/model.safetensors.index.json"))["weight_map"]
init = load({k: k[len("mtp."):] for k in midx if k.startswith("mtp.")}, mtp_dir)
for k in [k for k in init if k.endswith(".weight_scale")]:
    w = k[: -len("_scale")]
    init[w] = init[w].float() * init.pop(k).float()
if args.init:
    with safe_open(args.init, "pt", device="cuda") as fh:
        init = {k[len("mtp."):]: fh.get_tensor(k) for k in fh.keys()}
missing, unexpected = mtp.load_state_dict({k: v.float() for k, v in init.items()}, strict=True)
del init
log(event="mtp_loaded", params_m=round(sum(p.numel() for p in mtp.parameters()) / 1e6, 1))

recs = [json.loads(l) for l in open(args.data)]
recs = [r for r in recs if len(r["prompt_ids"]) + len(r["out_ids"]) <= args.max_len and len(r["out_ids"]) >= 4]
train = [r for r in recs if r["split"] == "train"]
evals = [r for r in recs if r["split"] == "eval"][: args.eval_max]
log(event="data", train=len(train), eval=len(evals),
    train_out_tok=sum(len(r["out_ids"]) for r in train))


alpha_ema = [0.0] * args.K  # running per-step soft acceptance sum(min(p, q)), sets the LK blend


def step_seq(r, train_mode):
    """Per-step [sum_kl, n, sum_acc, sum_top1] for one sequence; backprops the weighted loss if train_mode.

    Step k (1-based) mirrors vLLM drafting: input (h, emb(x_{i+k})) at position i+k-1, where h is the
    target hidden state H_i for k=1 and the previous step's (post-norm) MTP output otherwise; it predicts
    x_{i+k+1}, whose target distribution comes from H_{i+k}.
    """
    ids = torch.tensor(r["prompt_ids"] + r["out_ids"], device=dev)
    P, L = len(r["prompt_ids"]), ids.numel()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pos = torch.arange(L, device=dev)[None]
        H = tgt.model(input_ids=ids[None], position_ids=pos).last_hidden_state[0]
        emb = tgt.model.embed_tokens(ids)
    weights = [args.beta ** k for k in range(args.K)]
    weights = [w / sum(weights) * (args.think_weight if r["think"] else 1.0) for w in weights]
    lams = [math.exp(-args.eta * a) if args.loss == "lk" else 1.0 for a in alpha_ema]
    tots, outs, grads = [], [], []
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(train_mode):
        Ms = mtp.chain(emb, H, lambda off, n: tgt.model.rotary_emb(H[None], pos[:, off : off + n]), args.K)
    lo = P - 1  # drafts start at the last accepted position
    for k, M in enumerate(Ms, 1):
        n = M.shape[0]
        if n <= lo:
            break
        Md = M.detach().requires_grad_(train_mode)
        tot = [0.0, 0, 0.0, 0, 0.0]
        for s0 in range(lo, n, args.chunk):
            e0 = min(s0 + args.chunk, n)
            with torch.no_grad():
                tl = tgt.lm_head(H[s0 + k : e0 + k]).float()
                tlogp = F.log_softmax(tl, -1)
                tp = tlogp.exp()
            with torch.set_grad_enabled(train_mode):
                dl = (Md[s0:e0].to(torch.bfloat16) @ tgt.lm_head.weight.T).float()
                kl = (tp * (tlogp - F.log_softmax(dl, -1))).sum(-1)
                alpha = torch.minimum(tp, F.softmax(dl, -1)).sum(-1)
                if train_mode:
                    lam = lams[k - 1]
                    ((lam * kl.sum() + (1 - lam) * (1 - alpha).sum()) * weights[k - 1]).backward()
            with torch.no_grad():
                am = dl.argmax(-1)
                tot[0] += kl.sum().item()
                tot[1] += e0 - s0
                tot[2] += tp.gather(-1, am[:, None]).sum().item()
                tot[3] += (am == tl.argmax(-1)).sum().item()
                tot[4] += alpha.sum().item()
        tots.append(tot)
        if train_mode:
            alpha_ema[k - 1] = 0.9 * alpha_ema[k - 1] + 0.1 * tot[4] / tot[1]
            outs.append(M)
            grads.append(Md.grad)
    if train_mode and outs:
        torch.autograd.backward(outs, grads)
    return tots


def merge(agg, tots):
    for k, t in enumerate(tots):
        a = agg.setdefault(k, [0.0, 0, 0.0, 0, 0.0])
        for j in range(5):
            a[j] += t[j]


def summarize(agg):
    out = {"kl": round(agg[0][0] / agg[0][1], 4), "acc": round(agg[0][2] / agg[0][1], 4),
           "top1": round(agg[0][3] / agg[0][1], 4), "alpha": round(agg[0][4] / agg[0][1], 4), "n": agg[0][1]}
    for k in sorted(agg)[1:]:
        v = agg[k]
        out[f"acc{k + 1}"] = round(v[2] / v[1], 4)
        out[f"top1_{k + 1}"] = round(v[3] / v[1], 4)
        out[f"alpha{k + 1}"] = round(v[4] / v[1], 4)
    return out


def evaluate(tag):
    mtp.eval()
    agg = {}
    for r in evals:
        key = "think" if r["think"] else "nothink"
        merge(agg.setdefault(f"{r['src']}/{key}" if args.eval_by_src else key, {}), step_seq(r, False))
    mtp.train()
    out = {k: summarize(v) for k, v in agg.items()}
    log(event="eval", tag=tag, **out)
    return out


if args.diag:
    import collections
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{args.model}/tokenizer.json")
    for kind in ("nothink", "think"):
        dc, tc, n = collections.Counter(), collections.Counter(), 0
        for r in [r for r in evals if r["think"] == (kind == "think")][: args.diag]:
            ids = torch.tensor(r["prompt_ids"] + r["out_ids"], device=dev)
            P, L = len(r["prompt_ids"]), ids.numel()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pos = torch.arange(L, device=dev)[None]
                H = tgt.model(input_ids=ids[None], position_ids=pos).last_hidden_state[0]
                emb = tgt.model.embed_tokens(ids)
                M = mtp(emb[1:][None], H[:-1][None], tgt.model.rotary_emb(H[None], pos[:, : L - 1]))[0]
                d = tgt.lm_head(M[P - 1 : L - 2]).argmax(-1).tolist()
                t = ids[P + 1 : L].tolist()
            dc.update(d); tc.update(t); n += len(d)
            if n == len(d):
                log(event="diag_seq", kind=kind, target=tok.decode(t[:40]), draft=[tok.id_to_token(x) for x in d[:40]])
        log(event="diag_hist", kind=kind, n=n,
            draft_top=[(tok.id_to_token(k), round(v / n, 3)) for k, v in dc.most_common(12)],
            target_top=[(tok.id_to_token(k), round(v / n, 3)) for k, v in tc.most_common(12)])
    raise SystemExit
if args.regime:
    THINK, CLOSE, NL, NLNL = 54, 55, 560, 49

    def run(ids, lo, hi):
        """H for answer positions [lo, hi), all-layer cos inputs, and MTP stats predicting tokens in the answer."""
        ids_t = torch.tensor(ids, device=dev)
        L = len(ids)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pos = torch.arange(L, device=dev)[None]
            out = tgt.model(input_ids=ids_t[None], position_ids=pos, output_hidden_states=True)
            layers = [h[0, lo:hi].float() for h in out.hidden_states]
        return layers, ids_t, pos

    def mtp_stats(H, ids_t, pos, lo, hi, H_tgt=None):
        """acc/top1 of MTP drafts whose target token lies in the answer: i in [lo-1, hi-2)."""
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            L = ids_t.numel()
            emb = tgt.model.embed_tokens(ids_t)
            M = mtp(emb[1:][None], H[:-1][None], tgt.model.rotary_emb(H[None], pos[:, : L - 1]))[0]
            a, b = lo - 1, hi - 2
            tp = F.softmax(tgt.lm_head((H if H_tgt is None else H_tgt)[a + 1 : b + 1]).float(), -1)
            am = tgt.lm_head(M[a:b]).argmax(-1)
            return tp.gather(-1, am[:, None]).mean().item(), (am == tp.argmax(-1)).float().mean().item()

    def full_H(ids):
        ids_t = torch.tensor(ids, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pos = torch.arange(len(ids), device=dev)[None]
            return tgt.model(input_ids=ids_t[None], position_ids=pos).last_hidden_state[0], ids_t, pos

    pairs = []
    for r in recs:
        o = r["out_ids"]
        if not r["think"] or CLOSE not in o or r["prompt_ids"][-2:] != [THINK, NL]:
            continue
        k = o.index(CLOSE)
        if o[k + 1] != NLNL:
            continue
        ans = o[k + 2 : k + 2 + args.regime]
        if len(ans) < 64:
            continue
        a_ids = r["prompt_ids"] + o[: k + 2] + ans
        b_ids = r["prompt_ids"][:-2] + [THINK, NLNL, CLOSE, NLNL] + ans
        pairs.append((a_ids, len(a_ids) - len(ans), b_ids, len(b_ids) - len(ans)))
    log(event="regime_pairs", n=len(pairs), ans_tok=sum(len(a) - la for a, la, _, _ in pairs))

    nl = cfg.num_hidden_layers + 1
    unit = lambda x: x / x.norm(dim=-1, keepdim=True)
    layer_cos = torch.zeros(nl)
    rows, store = [], []
    for j, (a_ids, la, b_ids, lb) in enumerate(pairs):
        LA, _, _ = run(a_ids, la, len(a_ids))
        LB, _, _ = run(b_ids, lb, len(b_ids))
        for l in range(nl):
            layer_cos[l] += (unit(LA[l]) * unit(LB[l])).sum(-1).mean().item()
        HA, ia, pa = full_H(a_ids)
        HB, ib, pb = full_H(b_ids)
        accA, t1A = mtp_stats(HA, ia, pa, la, len(a_ids))
        accB, t1B = mtp_stats(HB, ib, pb, lb, len(b_ids))
        rows.append((accA, accB, t1A, t1B))
        store.append((LA[-1].cpu(), LB[-1].cpu()))
        del LA, LB
    layer_cos /= len(pairs)
    log(event="regime_layer_cos", every4=[round(layer_cos[l].item(), 4) for l in range(0, nl, 4)] + [round(layer_cos[-1].item(), 4)])
    n = len(rows)
    log(event="regime_mtp_same_tokens", n=n,
        acc_after_reasoning=round(sum(r[0] for r in rows) / n, 4), acc_after_empty_think=round(sum(r[1] for r in rows) / n, 4),
        top1_after_reasoning=round(sum(r[2] for r in rows) / n, 4), top1_after_empty_think=round(sum(r[3] for r in rows) / n, 4))

    tr, te = store[0::2], store[1::2]
    UA = lambda ss: torch.cat([unit(a) for a, _ in ss])
    UB = lambda ss: torch.cat([unit(b) for _, b in ss])
    d = UA(tr).mean(0) - UB(tr).mean(0)
    dh = d / d.norm()
    diff_te = UA(te) - UB(te)
    frac = (diff_te.mean(0).norm() ** 2 / diff_te.pow(2).sum(-1).mean()).item()
    seq_d = [unit(unit(a).mean(0) - unit(b).mean(0)) for a, b in te]
    seq_cos = torch.stack([(x * dh).sum() for x in seq_d])
    pa_tr, pb_tr = UA(tr) @ dh, UB(tr) @ dh
    thr = ((pa_tr.mean() + pb_tr.mean()) / 2).item()
    pa_te, pb_te = UA(te) @ dh, UB(te) @ dh
    probe = ((pa_te > thr).float().sum() + (pb_te <= thr).float().sum()).item() / (pa_te.numel() + pb_te.numel())
    paired = (pa_te > pb_te).float().mean().item()
    rnd = UA(te)
    perm = torch.randperm(rnd.shape[0])
    base_cos = (rnd * rnd[perm]).sum(-1).mean().item()
    log(event="regime_direction", same_token_cos_A_vs_B=round((UA(te) * UB(te)).sum(-1).mean().item(), 4),
        random_token_pair_cos=round(base_cos, 4), diff_var_explained_by_mean_dir=round(frac, 4),
        per_seq_diff_cos_to_train_dir_mean=round(seq_cos.mean().item(), 4), min=round(seq_cos.min().item(), 4),
        probe_1d_acc=round(probe, 4), paired_proj_A_gt_B=round(paired, 4),
        proj_A_mean=round(pa_te.mean().item(), 4), proj_B_mean=round(pb_te.mean().item(), 4), proj_std=round(torch.cat([pa_te, pb_te]).std().item(), 4))

    delta = (torch.cat([a for a, _ in tr]).mean(0) - torch.cat([b for _, b in tr]).mean(0)).to(dev)
    steer = {}
    for alpha in (0.0, 0.5, 1.0, 1.5):
        accs, t1s = [], []
        for j, (a_ids, la, b_ids, lb) in enumerate(pairs):
            if j % 2 == 0:
                continue
            HB0, ib, pb = full_H(b_ids)
            HB = HB0.clone()
            HB[lb - 1 :] += alpha * delta.to(HB.dtype)
            acc_, t1_ = mtp_stats(HB, ib, pb, lb, len(b_ids), H_tgt=HB0)
            accs.append(acc_)
            t1s.append(t1_)
        steer[str(alpha)] = [round(sum(accs) / len(accs), 4), round(sum(t1s) / len(t1s), 4)]
    log(event="regime_steering_empty_think_plus_alpha_delta", acc_top1=steer, delta_norm=round(delta.norm().item(), 3),
        H_norm=round(torch.cat([b for _, b in te]).norm(dim=-1).mean().item(), 3))
    raise SystemExit
if args.simulate:
    for r in [json.loads(l) for l in open(args.simulate)]:
        ids = torch.tensor(r["prompt_ids"] + r["out_ids"], device=dev)
        P, L = len(r["prompt_ids"]), ids.numel()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pos = torch.arange(L, device=dev)[None]
            H = tgt.model(input_ids=ids[None], position_ids=pos).last_hidden_state[0]
            tgt_greedy = tgt.lm_head(H).argmax(-1).tolist()
            Ms = mtp.chain(tgt.model.embed_tokens(ids), H, lambda off, n: tgt.model.rotary_emb(H[None], pos[:, off : off + n]), args.K)
            drafts = [tgt.lm_head(M).argmax(-1).tolist() for M in Ms]
        x = ids.tolist()
        acc, n_drafts, i = [0] * args.K, 0, P - 1
        while i + 2 < L:
            n_drafts += 1
            a = 0
            for k in range(1, args.K + 1):
                if i + k + 1 >= L or i >= len(drafts[k - 1]) or drafts[k - 1][i] != x[i + k + 1]:
                    break
                acc[k - 1] += 1
                a = k
            i += a + 1
        agree = sum(tgt_greedy[j] == x[j + 1] for j in range(P - 1, L - 1)) / (L - P)
        log(event="simulate", name=r.get("name"), drafts=n_drafts, acc_per_pos=acc, vllm=r.get("vllm"),
            target_greedy_agree=round(agree, 4))
    raise SystemExit
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    r0 = evals[0]
    ids0 = torch.tensor(r0["prompt_ids"] + r0["out_ids"], device=dev)[:512]
    pos0 = torch.arange(ids0.numel(), device=dev)[None]
    H0 = tgt.model(input_ids=ids0[None], position_ids=pos0).last_hidden_state[0]
    e0 = tgt.model.embed_tokens(ids0)
    ref = mtp(e0[1:-1][None], H0[:-2][None], tgt.model.rotary_emb(H0[None], pos0[:, :-2]))[0].float()
    got = mtp.chain(e0, H0, lambda off, n: tgt.model.rotary_emb(H0[None], pos0[:, off : off + n]), 1)[0].float()
    log(event="chain_check", step1_rel_err=round(((got - ref).norm() / ref.norm()).item(), 5))
evaluate("init")
if args.eval_only:
    raise SystemExit

init_params = [p.detach().to(torch.bfloat16).clone() for p in mtp.parameters()] if args.l2_init else None
opt = torch.optim.AdamW(mtp.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
total_tok = sum(len(r["out_ids"]) for r in train) * args.epochs
total_steps = max(1, int(total_tok / args.tokens_per_step))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
log(event="start", total_steps=total_steps)

order = []
while len(order) < len(train) * args.epochs:
    perm = list(range(len(train)))
    random.shuffle(perm)
    order += perm
order = order[: int(len(train) * args.epochs)]

step, acc_tok, run = 0, 0, {}
t_last = time.time()
opt.zero_grad(set_to_none=True)
for n, i in enumerate(order):
    t = step_seq(train[i], True)
    merge(run, t)
    acc_tok += t[0][1] if t else 0
    if acc_tok >= args.tokens_per_step or n == len(order) - 1:
        for p in mtp.parameters():
            if p.grad is not None:
                p.grad.div_(max(acc_tok, 1))
        if init_params:
            for p, p0 in zip(mtp.parameters(), init_params):
                if p.grad is not None:
                    p.grad.add_(p.detach() - p0.float(), alpha=2 * args.l2_init)
        gn = torch.nn.utils.clip_grad_norm_(mtp.parameters(), 1.0).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        dt = time.time() - t_last
        t_last = time.time()
        log(event="train", step=step, seqs=n + 1, **summarize(run), gn=round(gn, 3), lam=[round(math.exp(-args.eta * a), 3) for a in alpha_ema] if args.loss == "lk" else None, lr=sched.get_last_lr()[0],
            tok_s=round(acc_tok / dt))
        run, acc_tok = {}, 0
        if step % args.eval_every == 0:
            evaluate(f"step{step}")
            save_file({f"mtp.{k}": v.detach().to(torch.bfloat16).contiguous() for k, v in mtp.state_dict().items()},
                      f"{args.out}/mtp-step{step}.safetensors")
evaluate("final")
save_file({f"mtp.{k}": v.detach().to(torch.bfloat16).contiguous() for k, v in mtp.state_dict().items()},
          f"{args.out}/mtp.safetensors")
log(event="done", steps=step)
