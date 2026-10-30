"""Build a natural prompt mix for EXAONE-4.5 MTP retraining data generation."""
import argparse, glob, json, random, re

import pyarrow.parquet as pq
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="/raw")
ap.add_argument("--tokenizer", default="/model/tokenizer.json")
ap.add_argument("--out", default="/data/prompts.jsonl")
ap.add_argument("--max-prompt-tokens", type=int, default=3000)
ap.add_argument("--think-frac", type=float, default=0.2)
ap.add_argument("--eval-frac", type=float, default=0.02)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-wc-en", type=int, default=5000)
ap.add_argument("--n-wc-ko", type=int, default=2500)
ap.add_argument("--n-kovast", type=int, default=3000)
ap.add_argument("--n-kullm", type=int, default=2000)
ap.add_argument("--n-code", type=int, default=3000)
args = ap.parse_args()
rng = random.Random(args.seed)
tok = Tokenizer.from_file(args.tokenizer)
seen = set()


def ntok(msgs):
    return sum(len(tok.encode(m["content"]).ids) + 4 for m in msgs)


def ok(msgs):
    last = msgs[-1]["content"].strip()
    key = re.sub(r"\s+", " ", last.lower())[:500]
    if len(last) < 5 or key in seen:
        return False
    if ntok(msgs) > args.max_prompt_tokens:
        return False
    seen.add(key)
    return True


def cut_multiturn(conv):
    """conv: list of {role, content}; cut at a random user turn, keep history."""
    users = [i for i, m in enumerate(conv) if m["role"] == "user"]
    if not users:
        return None
    k = rng.choice(users[:3])
    msgs = [{"role": m["role"], "content": m["content"]} for m in conv[: k + 1]]
    if any(not m["content"].strip() for m in msgs):
        return None
    return msgs


def take(src, cands, n):
    rng.shuffle(cands)
    out = []
    for msgs in cands:
        if len(out) >= n:
            break
        if msgs and ok(msgs):
            out.append({"src": src, "messages": msgs})
    print(f"{src}: {len(out)} / {len(cands)} candidates")
    return out


wc = {"English": [], "Korean": []}
for f in sorted(glob.glob(f"{args.raw}/wildchat/data/*.parquet")):
    t = pq.read_table(f, columns=["conversation", "language", "toxic", "redacted"])
    for r in t.to_pylist():
        if r["language"] in wc and not r["toxic"] and not r["redacted"]:
            wc[r["language"]].append(r["conversation"])
items = take("wc_en", [cut_multiturn(c) for c in wc["English"]], args.n_wc_en)
items += take("wc_ko", [cut_multiturn(c) for c in wc["Korean"]], args.n_wc_ko)

kv = pq.read_table(glob.glob(f"{args.raw}/kovast/data/*.parquet")[0]).to_pylist()
kv = [
    cut_multiturn(
        [{"role": "user" if m["from"] in ("user", "human") else "assistant", "content": m["value"]}
         for m in r["conversations"]]
    )
    for r in rng.sample(kv, min(len(kv), 30000))
]
items += take("kovast", kv, args.n_kovast)

kl = []
with open(f"{args.raw}/kullm/kullm-v2.jsonl") as fh:
    for line in fh:
        r = json.loads(line)
        q = r["instruction"].strip() + ("\n\n" + r["input"].strip() if r["input"].strip() else "")
        kl.append([{"role": "user", "content": q}])
items += take("kullm", kl, args.n_kullm)

cf = []
with open(f"{args.raw}/codefeedback/CodeFeedback-Filtered-Instruction.jsonl") as fh:
    for line in fh:
        cf.append([{"role": "user", "content": json.loads(line)["query"]}])
items += take("code", cf, args.n_code)

rng.shuffle(items)
n_eval = int(len(items) * args.eval_frac)
with open(args.out, "w") as fh:
    for i, it in enumerate(items):
        it["id"] = f"p{i:06d}"
        it["think"] = rng.random() < args.think_frac
        it["split"] = "eval" if i < n_eval else "train"
        fh.write(json.dumps(it, ensure_ascii=False) + "\n")
print(f"total {len(items)}, eval {n_eval}, think {sum(it['think'] for it in items)}")
