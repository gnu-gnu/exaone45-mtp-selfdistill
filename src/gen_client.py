"""Generate EXAONE-4.5 responses with exact token ids; resumable, one JSON line per prompt."""
import argparse, asyncio, hashlib, json, os, time

import aiohttp

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", default="/data/prompts.jsonl")
ap.add_argument("--out", default="/data/gen.jsonl")
ap.add_argument("--url", default="http://localhost:18001/v1/chat/completions")
ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""), help="bearer token for a server started with --api-key")
ap.add_argument("--model", default="exaone45-33b-fp8")
ap.add_argument("--concurrency", type=int, default=96)
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--max-tokens-think", type=int, default=2048)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--sys-mix", action="store_true", help="add a system prompt to train prompts: 50% none, 25% concise, 25% generic")
args = ap.parse_args()

done = set()
if os.path.exists(args.out):
    with open(args.out) as fh:
        done = {json.loads(l)["id"] for l in fh if l.strip()}
todo = [json.loads(l) for l in open(args.prompts)]
todo = [p for p in todo if p["id"] not in done]
if args.limit:
    todo = todo[: args.limit]
print(f"done {len(done)}, todo {len(todo)}", flush=True)

CONCISE = [
    "당신은 간결하게 답하는 어시스턴트입니다. 질문에 바로 답하세요. 서론, 질문 되풀이, 마무리 요약, 추가 도움 제안은 쓰지 마세요. "
    "기본은 3~5문장 이내로 답하고, 목록·제목·코드는 꼭 필요할 때만 쓰세요. 사용자가 쓴 언어로 답하세요.",
    "Be concise. Answer directly without preamble, restating the question, or closing summaries. "
    "Use lists or headings only when they clearly help.",
    "간단명료하게 답해 주세요. 핵심만 말하고 불필요한 인사말이나 요약은 빼 주세요.",
    "Keep answers short: a few sentences unless the user asks for detail or code.",
]
GENERIC = [
    "You are a helpful assistant.",
    "당신은 친절하고 유능한 AI 어시스턴트입니다.",
    "You are EXAONE, a large language model created by LG AI Research. Answer the user's questions accurately.",
    "You are a helpful assistant. Respond in the same language as the user.",
]


def pick_system(p):
    if not args.sys_mix or p["split"] != "train" or p["messages"][0]["role"] == "system":
        return None
    h = int(hashlib.md5(p["id"].encode()).hexdigest(), 16)
    r = h % 100
    if r < 50:
        return None
    pool = CONCISE if r < 75 else GENERIC
    return pool[(h // 100) % len(pool)]


stats = {"n": 0, "tok": 0, "err": 0, "t0": time.time()}


async def worker(q, session, fh):
    while True:
        p = await q.get()
        if p is None:
            return
        system = pick_system(p)
        messages = ([{"role": "system", "content": system}] if system else []) + p["messages"]
        body = {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens_think if p["think"] else args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": p["think"]},
            "return_token_ids": True,
        }
        try:
            async with session.post(args.url, json=body) as r:
                j = await r.json()
            c = j["choices"][0]
            rec = {
                "id": p["id"], "src": p["src"], "think": p["think"], "split": p["split"],
                "prompt_ids": j["prompt_token_ids"], "out_ids": c["token_ids"],
                "finish": c["finish_reason"], "sys": system,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            stats["n"] += 1
            stats["tok"] += len(c["token_ids"])
            if stats["n"] % 100 == 0:
                dt = time.time() - stats["t0"]
                print(f"{stats['n']} done, {stats['tok']} out tok, {stats['tok']/dt:.0f} tok/s, err {stats['err']}", flush=True)
        except Exception as e:
            stats["err"] += 1
            print(f"ERR {p['id']}: {type(e).__name__} {str(e)[:200]} {str(locals().get('j', ''))[:300]}", flush=True)


async def main():
    q = asyncio.Queue()
    for p in todo:
        q.put_nowait(p)
    for _ in range(args.concurrency):
        q.put_nowait(None)
    timeout = aiohttp.ClientTimeout(total=3600)
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        with open(args.out, "a") as fh:
            await asyncio.gather(*(worker(q, session, fh) for _ in range(args.concurrency)))
    dt = time.time() - stats["t0"]
    print(f"FINISHED {stats['n']} in {dt:.0f}s, {stats['tok']} tok, {stats['tok']/dt:.0f} tok/s, err {stats['err']}", flush=True)


asyncio.run(main())
