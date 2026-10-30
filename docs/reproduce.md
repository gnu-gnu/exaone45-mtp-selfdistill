# 재현 절차

게시한 결과를 처음부터 다시 만드는 순서입니다. 시작하기 전에 [설정 가이드](configuration.md)대로 `.env`를 채우고 `scripts/00_check.sh`를 통과시킵니다. 소요 시간은 GB10(DGX Spark 계열, 통합 메모리 128GB) 한 대 기준 실측값입니다.

| 단계 | 명령 | GB10 소요 | 결과 | 재현 기준 |
| --- | --- | --- | --- | --- |
| 1 | `scripts/01_download.sh` | 수 분 | `work/raw/` | `data/SHA256SUMS.raw`와 바이트 단위로 같음 |
| 2 | `scripts/02_prompts.sh` | 수 분 | `work/data/prompts.jsonl`(13,157), `prompts_ko.jsonl`(1,580) | `data/SHA256SUMS.prompts`와 바이트 단위로 같음 |
| 3 | `scripts/03_generate.sh all` | 약 5.5시간 | `gen.jsonl`(2,884), `gen_ko.jsonl`(1,580) | 건수와 프롬프트 id가 같음. 답변 토큰이 같다는 보장은 없음 |
| 4 | `scripts/04_train.sh stage1` | 약 3.6시간 | `work/ckpt/stage1/` | 아래 오프라인 지표가 비슷함 |
| 5 | `scripts/04_train.sh stage2` | 약 2.7시간 | `work/ckpt/stage2/mtp.safetensors` | 아래 오프라인 지표가 비슷함 |
| 6 | `scripts/05_overlay.sh stage2` | 1분 | `work/serve/stage2/` | vLLM이 원본처럼 불러옴 |
| 7 | `scripts/serve.sh up mtp work/serve/stage2` 후 `scripts/bench.sh stage2 all` | 약 30분 | `work/bench/stage2/` | 아래 속도와 수용률이 비슷함 |

3~5단계는 길어서 `nohup scripts/03_generate.sh all > work/logs/gen.out 2>&1 &`처럼 백그라운드로 돌리기를 권합니다. 생성은 끊겨도 같은 명령을 다시 실행하면 이어서 돕니다. 학습은 처음부터 다시 돌아갑니다.

## 단계별 설명

### 1. 원본 데이터

받는 데이터셋과 고정 리비전은 다음과 같습니다. 데이터는 이 저장소에 포함하지 않으며, 각 데이터셋의 라이선스를 따릅니다.

| 데이터셋 | 리비전 | 받는 범위 | 라이선스 |
| --- | --- | --- | --- |
| [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | `7d6490e4` | 샤드 0~3 (14개 중) | ODC-BY |
| [maywell/koVast](https://huggingface.co/datasets/maywell/koVast) | `b0704da1` | 샤드 0 (3개 중) | MIT |
| [nlpai-lab/kullm-v2](https://huggingface.co/datasets/nlpai-lab/kullm-v2) | `cddcb73c` | 전체 | Apache-2.0 |
| [m-a-p/CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) | `a08c213a` | 전체 | Apache-2.0 |
| [beomi/KoAlpaca-v1.1a](https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a) | `03d3122c` | 전체 | 데이터 카드에 표기 없음 |
| [AI Hub 271](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=271) 한국어 감정 정보가 포함된 연속적 대화 데이터셋 | 파일 50327 | xlsx 1개 | AI Hub 이용약관 (재배포 불가) |

### 2. 프롬프트

- `build_prompts.py`: WildChat 영어 5,000개와 한국어 157개(목표는 2,500개지만 받은 샤드 4개의 한국어 대화가 170개이고 그중 157개가 조건을 통과), koVast 3,000, kullm-v2 2,000, CodeFeedback 3,000개, 합계 13,157개를 뽑습니다. 유해·비식별 처리된 WildChat 대화는 제외합니다. 중복과 3,000토큰 초과 프롬프트를 거르고, 20%에 추론 모드를 지정하고, 앞 2%를 평가용으로 나눕니다(seed 0).
- `add_ko_prompts.py`: AI Hub 271 대화의 첫 발화 중 질문 형태인 것 840개와 KoAlpaca 질문 740개를 뽑습니다. 각각 40개가 평가용입니다.
- 결과가 게시한 실행과 바이트 단위로 같은지 체크섬으로 확인합니다. 다르면 원본 데이터 체크섬, `MODEL_DIR`의 `tokenizer.json`, 도구 이미지 버전(`docker/tools.Dockerfile`)을 먼저 확인합니다.

### 3. 자기 생성 답변

EXAONE 본체가 프롬프트마다 답한 토큰 id를 그대로 저장합니다. 이것이 MTP 헤드의 학습 목표 분포를 만드는 입력입니다. 샘플링은 모델 기본값(`generation_config.json`)이고, `max_tokens`는 비추론 1,024, 추론 2,048입니다. 게시한 실행과 같은 세 묶음으로 돌립니다.

| 묶음 | 명령 | 건수 | 시스템 프롬프트 |
| --- | --- | --- | --- |
| pilot | `03_generate.sh pilot` | 384 (평가 263 + 학습 121) | 없음 |
| main | `03_generate.sh main` | 2,500 | 학습 프롬프트의 50%에 붙임(간결화 25%, 일반 25%) |
| ko | `03_generate.sh ko` | 1,580 | 위와 같음 |

`03_generate.sh smoke 8`은 8건만 만들어 서버와 클라이언트가 동작하는지 확인합니다.

### 4~5. MTP 헤드 학습

`train.sh`가 `src/train_mtp.py`를 `VLLM_IMAGE` 안에서 돌립니다. 본체 33B는 고정하고 MTP 모듈(약 5.4억 파라미터)만 학습합니다.

- stage1: 원본 MTP 가중치에서 시작합니다. K=3 재귀 학습(TTT), 단계별 KL 손실 가중치 0.6^(k-1), 2에폭, lr 5e-5, warmup 10. 268 스텝입니다.
- stage2: stage1 결과에서 이어서 한국어 묶음까지 합친 데이터로 1에폭, lr 2e-5. 191 스텝입니다.

`ckpt/NAME/log.jsonl`의 `eval` 이벤트에서 `acc`는 E[p_target(argmax p_mtp)], 즉 greedy 드래프트의 기대 수용률입니다. `acc2`, `acc3`은 두 번째와 세 번째 드래프트 위치의 값입니다. 게시한 실행의 최종 평가값은 다음과 같습니다. 다시 생성한 답변이 게시한 실행과 같다는 보장이 없으므로, 재현한 지표는 이 값과 조금 다를 수 있습니다.

| 체크포인트 | 비추론 acc / acc2 / acc3 | 추론 acc / acc2 / acc3 |
| --- | --- | --- |
| 원본 MTP (`--eval-only`) | 0.037 / - / - | 0.612 / - / - |
| stage1 (`--eval-max 263`) | 0.569 / 0.500 / 0.464 | 0.606 / 0.532 / 0.492 |

stage2는 출처별로 기록합니다(`--eval-by-src`). 게시한 실행의 최종값 일부: 코드 비추론 0.744, WildChat 영어 비추론 0.527, AI Hub 271 비추론 0.403, KoAlpaca 비추론 0.489.

### 6~7. 서빙과 측정

`05_overlay.sh`는 `MODEL_DIR`에서 `mtp.*` 텐서가 든 샤드(`model-00009-of-00009.safetensors`)만 새로 쓰고 나머지 파일은 하드링크로 연결합니다. vLLM 코드는 바꾸지 않습니다. MTP 텐서 중 원본이 FP8인 것은 같은 방식(채널별 scale)으로 다시 양자화합니다.

비교하려면 같은 설정으로 세 서버를 차례로 측정합니다.

```bash
scripts/serve.sh up nospec               && scripts/bench.sh nospec off && scripts/serve.sh down
scripts/serve.sh up mtp                  && scripts/bench.sh orig off   && scripts/serve.sh down   # 원본 MTP
scripts/serve.sh up mtp work/serve/stage2 && scripts/bench.sh stage2 all && scripts/serve.sh down
```

서버를 띄운 뒤 첫 요청은 워밍업 때문에 2~3초 느립니다. `off`와 `on`은 기본 2회 측정하고, 로그 끝에 평균 속도와 전체 수용률을 적습니다. 게시한 측정값(비추론, 프롬프트 5개 × 2회 평균 tok/s, 괄호는 드래프트 전체를 합친 수용률)은 다음과 같습니다.

| 장비 | MTP 없음 | 원래 MTP | 새 MTP |
| --- | --- | --- | --- |
| RTX PRO 6000 (`MAX_NUM_SEQS=8`) | 36.7 | 31.0 (수용률 0.8%) | 61.3 (33.7%) |
| GB10 (`GPU_MEM_UTIL=0.55`, `MAX_NUM_SEQS=1`, `CUDAGRAPH_MAX_SIZE=24`) | 5.8 (1회) | 5.1 (1회, 0.9%) | 9.7 (31.9%) |

프롬프트별 값은 [results/natural-prompts.csv](../results/natural-prompts.csv)에 있습니다.

## 오프라인 분석

```bash
scripts/analyze.sh regime          # 원본 MTP: 같은 답변 토큰을 실제 추론 뒤 / 빈 think 블록 뒤에 놓고 비교
scripts/analyze.sh regime stage2   # 재학습한 헤드로 같은 비교
scripts/analyze.sh eval stage2     # stage2 평가셋(343건)에서 출처·모드별 기대 수용률
```

`regime`은 pilot 묶음(384건)의 추론 모드 기록 중 답변이 64토큰 이상인 것으로 쌍을 만듭니다. 게시한 실행에서는 54쌍이었고, 원본 MTP의 기대 수용률이 실제 추론 뒤 58.0%, 빈 think 블록 뒤 2.5%였습니다. 같은 로그에 층별 hidden state 코사인 유사도와 두 모드를 가르는 방향(1차원 probe) 결과도 남습니다.
