"""Write Korean casual prompts (AI Hub 271 first utterances, KoAlpaca questions) to a separate prompts file."""
import argparse, json, random, re

import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--emodial", default="/data/emodial.jsonl")
ap.add_argument("--koalpaca", default="/raw/koalpaca/data/train-00000-of-00001-21df739eb88d711e.parquet")
ap.add_argument("--prompts", default="/data/prompts.jsonl", help="existing prompts; new ids must not collide")
ap.add_argument("--out", default="/data/prompts_ko.jsonl")
ap.add_argument("--n-emodial", type=int, default=800)
ap.add_argument("--n-koalpaca", type=int, default=700)
ap.add_argument("--n-eval", type=int, default=40)
ap.add_argument("--think-frac", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dry", action="store_true")
args = ap.parse_args()
rng = random.Random(args.seed)

ASK = re.compile(r"\?|줘|주세요|주실|할까|을까|ㄹ까|어때|뭐야|뭔가요|나요|까요|습니까|ㅂ니까|건가요|는지")
CONTEXT = re.compile(r"^(그래서|그럼|그러니까|그러면|근데|그런데|그건|그거|그게|그렇게|그래도|그래|아니|네[,.]|응|어\b|야\b|하지만)")
ADDRESS = re.compile(r"(형|누나|오빠|언니|엄마|아빠|팀장님|선배|선생님|사장님|자기야|여보)[!,.~ ]")
CODE = re.compile(r"```|\bdef |\bimport |#include|[{};]\s*$|https?://", re.M)

emo = []
for line in open(args.emodial):
    d = json.loads(line)
    t = d["turns"][0]["text"]
    if d["id"] == "emodial00000" or not 20 <= len(t) <= 200:
        continue
    if ASK.search(t) and not CONTEXT.search(t) and not ADDRESS.search(t):
        emo.append(t)
kq = []
for r in pq.read_table(args.koalpaca).to_pylist():
    t = (r.get("instruction") or "").strip()
    if 15 <= len(t) <= 300 and not CODE.search(t):
        kq.append(t)
print(f"candidates: emodial {len(emo)}, koalpaca {len(kq)}")

existing = {json.loads(l)["id"] for l in open(args.prompts)}
new = []
for src, pool, n, pre in [("emodial", emo, args.n_emodial, "e"), ("koalpaca", kq, args.n_koalpaca, "k")]:
    pick = rng.sample(sorted(set(pool)), n + args.n_eval)
    for i, t in enumerate(pick):
        new.append({"id": f"{pre}{i:05d}", "messages": [{"role": "user", "content": t}],
                    "split": "eval" if i < args.n_eval else "train", "src": src,
                    "think": rng.random() < args.think_frac})
assert not existing & {p["id"] for p in new}
for src in ("emodial", "koalpaca"):
    print(f"== {src} samples")
    for p in [p for p in new if p["src"] == src][args.n_eval: args.n_eval + 12]:
        print("  ", p["messages"][0]["content"][:110].replace("\n", " "))
if not args.dry:
    with open(args.out, "w") as fh:
        for p in new:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(new)} prompts to {args.out}")
