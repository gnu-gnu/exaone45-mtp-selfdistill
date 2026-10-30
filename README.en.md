# EXAONE 4.5 MTP self-distillation

English | [한국어](README.md)

> **Disclaimer.** This is a personal research project. It is not an official derivative of EXAONE and is not affiliated with LG AI Research or with the author's institution. It contains no EXAONE weights, no generated data and no AI Hub data.

The MTP (multi-token prediction) head of EXAONE-4.5-33B reaches around 30% acceptance in thinking mode, but drops below 1% when thinking is turned off. With MTP speculative decoding enabled, non-thinking requests therefore get slower than without it. This repository reproduces the fix: keep the 33B target frozen and retrain only the MTP head on the model's own outputs.

The analysis is written up (in Korean) in [EXAONE 4.5 MTP 수용률을 뜯어보았다](https://blog.gnu.or.kr/posts/exaone45-mtp-non-thinking/).

## Results

vLLM v0.31.0, EXAONE-4.5-33B-FP8, 3 MTP draft tokens, 5 natural prompts × 2 runs, up to 400 output tokens.

| RTX PRO 6000 | No MTP | Original MTP | Retrained MTP |
| --- | --- | --- | --- |
| Non-thinking tok/s (speedup) | 36.7 | 31.0 (0.85×) | 61.3 (1.67×) |
| Non-thinking acceptance | - | 0.8% | 33.7% |
| Thinking tok/s (speedup) | 36.7 | 55.4 (1.51×) | 54.5 (1.49×) |
| Thinking acceptance | - | 28.3% | 27.2% |

Non-thinking requests became about 2× faster than with the original head and 1.67× faster than without MTP. Thinking requests stay close to the original head, with acceptance 1.1 points lower. A same-day re-measurement of all three configurations on a GB10 showed the same pattern (non-thinking 0.85× → 1.64×, thinking acceptance 29.7% → 29.2%; see [verification record](docs/verification.md)).

Raw measurements are in [results/](results/). GB10 numbers and offline metrics are in the [reproduction guide](docs/reproduce.md).

## What it does

1. Selects 14,737 prompts with fixed rules from public chat and instruction datasets and AI Hub dataset 271.
2. Has the EXAONE target (served by vLLM) answer 4,464 of them and stores the token ids. About 20% are in thinking mode, and part of them get a system prompt.
3. Trains only the MTP module (one decoder layer plus its input projection, about 536M parameters) with the 33B target frozen. The loss is the KL divergence between the target's next-token distribution and the MTP prediction. Training unrolls three recursive steps, the same way vLLM drafts three tokens.
4. Builds a serving directory from the original checkpoint with only the `mtp.*` tensors replaced. vLLM itself is not modified.
5. Benchmarks no MTP / original MTP / retrained MTP on the same prompts.

## Quick start

```bash
git clone https://github.com/gnu-gnu/exaone45-mtp-selfdistill && cd exaone45-mtp-selfdistill
cp .env.example .env && chmod 600 .env      # paths, AI Hub key, GPU and server connection
scripts/00_check.sh
scripts/01_download.sh                       # source data at pinned revisions, checksum-verified
scripts/02_prompts.sh                        # prompt files, checked byte-for-byte against the published run
scripts/03_generate.sh all                   # self-generated responses
scripts/04_train.sh stage1 && scripts/04_train.sh stage2
scripts/05_overlay.sh stage2
scripts/serve.sh up mtp work/serve/stage2 && scripts/bench.sh stage2 all
```

The detailed guides are in Korean:

- [Configuration](docs/configuration.md): everything that goes into `.env`, including the AI Hub application and key, GPU selection, connecting to a remote vLLM server, and GB10 memory settings.
- [Reproduction](docs/reproduce.md): time per stage and the expected result of each stage.
- [Verification record](docs/verification.md): what was checked from a clean clone before publishing.

### Data access

No data is redistributed. `scripts/01_download.sh` fetches it:

- Hugging Face datasets at pinned revisions: allenai/WildChat-1M (ODC-BY), maywell/koVast (MIT), nlpai-lab/kullm-v2 (Apache-2.0), m-a-p/CodeFeedback-Filtered-Instruction (Apache-2.0), beomi/KoAlpaca-v1.1a (no license stated). A Hugging Face token is optional.
- AI Hub dataset 271, [Korean continuous dialogue dataset with emotion labels](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=271). You need an aihub.or.kr account, an approved download application for this dataset (Korean nationals only), and an API key. Set it with `AIHUB_API_KEY`, `AIHUB_API_KEY_FILE` or `AIHUB_API_KEY_CMD` in `.env`, or export it for a single run.
- Model: `hf download LGAI-EXAONE/EXAONE-4.5-33B-FP8 --revision 69570ad5d9a5159816ef267f2f4d37e36447f1be`.

### Requirements

Linux, Docker with the NVIDIA Container Toolkit, `bash`, `curl` (7.55+), `jq`, `bc`, `sha256sum`. No Python on the host: data tools run in `docker/tools.Dockerfile`, training and serving in `vllm/vllm-openai:v0.31.0`. Measured on a GB10 (128 GB unified memory) and an RTX PRO 6000; on a GB10 set `TRAIN_MEM_LIMIT=96g`, `TRAIN_MIN_FREE_GB=16` and `EXCLUSIVE_GPU=1`.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/build_prompts.py`, `src/add_ko_prompts.py`, `src/emodial_to_jsonl.py` | Prompt selection |
| `src/gen_client.py` | Collects response token ids through the vLLM OpenAI-compatible API (resumable) |
| `src/train_mtp.py` | MTP head training and evaluation, plus the `--regime` (thinking vs non-thinking), `--diag` and `--simulate` analyses |
| `src/build_overlay.py` | Serving directory with only the `mtp.*` tensors replaced |
| `scripts/` | One script per stage. All settings come from `.env`. |
| `bench/` | Benchmark prompts and input documents |
| `data/SHA256SUMS.*` | Expected checksums of the source data and prompt files |
| `results/` | Raw measurements used in the blog post (CSV) |

## License and terms of use

- The code in this repository is under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for third-party material.
- **The EXAONE model is not included.** Obtain it yourself under the [EXAONE AI Model License Agreement 1.2 - NC](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B-FP8/blob/main/LICENSE).
- MTP weights produced with this code are Derivatives under that license: research and educational use only, and any distribution must include a copy of the license and use a name that starts with "EXAONE".
- Generated responses are Output under that license and may not be used commercially or to develop or improve models that compete with LG's models (Section 3.1).
- Prompt sources keep their own licenses. AI Hub data cannot be redistributed; each user has to apply for it.

When citing results, please name the model used: EXAONE-4.5-33B-FP8, LG AI Research.

Generated by Geunwoo Shim (geunwoo.j.shim@gmail.com)
