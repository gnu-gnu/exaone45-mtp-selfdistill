# 설정 가이드

이 저장소의 스크립트는 사람마다 다른 값(경로, 인증 정보, GPU와 서버 연결 정보)을 모두 `.env` 한 파일에서 읽습니다. 스크립트 안에는 호스트 이름, 경로, 키가 들어 있지 않습니다.

```bash
cp .env.example .env
chmod 600 .env
$EDITOR .env
scripts/00_check.sh     # 설정 점검. 비밀 값은 출력하지 않고 "set / not set"만 보여 줍니다.
```

`.env`는 `.gitignore`에 들어 있어 커밋되지 않습니다. 다른 파일을 쓰려면 `ENV_FILE=/path/to/file scripts/...`로 지정합니다. `.env`는 bash가 `source`로 읽으므로 공백이 있는 값은 따옴표로 감쌉니다. 셸에서 이미 export한 변수는 `.env`보다 우선합니다.

## 1. 경로

| 변수 | 필수 | 설명 |
| --- | --- | --- |
| `MODEL_DIR` | 예 | EXAONE-4.5-33B-FP8 체크포인트 디렉터리. 아래 명령으로 받은 위치입니다. |
| `WORK_DIR` | 아니요 (`./work`) | 원본 데이터, 생성 데이터, 체크포인트, 서빙 디렉터리, 로그, 벤치마크 결과가 모두 여기에 생깁니다. |
| `CACHE_DIR` | 아니요 (`$WORK_DIR/cache`) | vLLM 컴파일 캐시 |

vLLM 서버 컨테이너는 root로 돌기 때문에 `CACHE_DIR` 아래 파일은 root 소유가 됩니다. 일반 사용자로는 지워지지 않으므로 컨테이너로 지웁니다.

```bash
docker run --rm -v "$PWD/work:/w" --entrypoint rm vllm/vllm-openai:v0.31.0 -rf /w/cache
```

나머지 `WORK_DIR` 파일은 실행한 사용자 소유입니다. `work/serve/` 아래 하드링크는 지워도 `MODEL_DIR`의 원본에는 영향이 없습니다.

```bash
hf download LGAI-EXAONE/EXAONE-4.5-33B-FP8 --revision 69570ad5d9a5159816ef267f2f4d37e36447f1be --local-dir /path/to/EXAONE-4.5-33B-FP8
```

모델은 [EXAONE AI Model License 1.2 - NC](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B-FP8/blob/main/LICENSE)를 따릅니다. 받기 전에 라이선스를 읽어 주세요.

`WORK_DIR`과 `MODEL_DIR`을 같은 파일시스템에 두면 `05_overlay.sh`가 바뀌지 않은 샤드를 하드링크로 연결해 추가 디스크가 약 2GB만 듭니다. 파일시스템이 다르면 체크포인트 전체(약 33GB)를 복사합니다.

## 2. 인증 정보

### AI Hub API 키 (한국어 대화 데이터, 필수)

AI Hub 데이터셋 271 [한국어 감정 정보가 포함된 연속적 대화 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=271)은 재배포할 수 없어 각자 받아야 합니다.

1. aihub.or.kr에 가입하고 로그인합니다.
2. 데이터셋 271 페이지에서 "다운로드"를 눌러 이용 신청을 합니다. 내국인만 신청할 수 있습니다. API 키가 있어도 신청이 승인되지 않은 데이터셋은 받을 수 없습니다.
3. 마이페이지에서 API 키를 발급받습니다.
4. 아래 세 방법 중 하나로 `.env`에 넣습니다. 앞에 있는 방법이 우선합니다.

| 변수 | 쓰는 경우 |
| --- | --- |
| `AIHUB_API_KEY=...` | 키를 `.env`에 직접 적습니다. `.env`를 `chmod 600`으로 둡니다. |
| `AIHUB_API_KEY_FILE=~/.config/aihub/key` | 키만 담긴 파일을 따로 둡니다. 파일도 `chmod 600`으로 둡니다. |
| `AIHUB_API_KEY_CMD='op read "op://Vault/AI Hub/API-KEY"'` | 비밀번호 관리자 등 키를 출력하는 명령을 씁니다. 실행할 때마다 명령을 호출하고, 키는 디스크에 남지 않습니다. |

`.env`에 적지 않고 실행할 때만 환경 변수로 넘겨도 됩니다.

```bash
AIHUB_API_KEY="$(op read 'op://Vault/AI Hub/API-KEY')" scripts/01_download.sh
```

`01_download.sh`는 키를 curl 명령줄이 아니라 헤더 파일 디스크립터로 넘기므로 `ps`에 보이지 않습니다. 실패하면 AI Hub가 돌려준 메시지만 출력합니다. AI Hub 없이 나머지만 확인하려면 `SKIP_AIHUB=1`을 붙입니다. 다만 한국어 프롬프트를 만드는 `02_prompts.sh`부터는 이 데이터가 필요합니다.

### Hugging Face 토큰 (선택)

`HF_TOKEN`. 데이터셋과 모델은 모두 공개라 없어도 받을 수 있습니다. 토큰이 있으면 속도 제한이 완화됩니다.

### vLLM 서버 키 (선택)

`VLLM_API_KEY`. `SERVER_MODE=local`이면 스크립트가 이 키로 서버를 띄웁니다. `external`이면 연결할 서버의 키와 같아야 합니다. 키는 docker의 `-e`와 curl 헤더 파일로만 전달되고 명령줄에는 나타나지 않습니다.

## 3. GPU와 서버 연결

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `GPU_DEVICES` | `all` | `docker run --gpus`에 넘길 값입니다. 예: `all`, `device=0`, `device=0,1`. |
| `VLLM_IMAGE` | `vllm/vllm-openai:v0.31.0` | 결과를 낸 버전입니다. 학습 컨테이너도 이 이미지(PyTorch, transformers 포함)를 씁니다. |
| `SERVER_MODE` | `local` | `local`: 스크립트가 이 호스트에서 vLLM 컨테이너를 띄우고 내립니다. `external`: 직접 띄운 서버를 `VLLM_BASE_URL`로 씁니다. |
| `VLLM_PORT` | `18001` | `local`일 때 서버를 열 포트 |
| `VLLM_BASE_URL` | `http://localhost:$VLLM_PORT` | 클라이언트(생성, 벤치마크)가 붙을 주소입니다. `/v1`은 붙이지 않습니다. |
| `SERVED_MODEL_NAME` | `exaone45-33b-fp8` | 요청의 `model` 필드 |
| `EXCLUSIVE_GPU` | `1` | 학습과 vLLM 서버를 동시에 띄우지 않습니다. GB10처럼 메모리를 공유하는 장비에서는 꼭 `1`로 둡니다. |

### 실행 형태별 설정

**GPU 서버에서 직접 실행 (기본).** 저장소를 GPU 서버에 클론하고 `SERVER_MODE=local`로 둡니다. 모든 단계가 같은 호스트에서 돌아갑니다.

**서버는 GPU 호스트, 클라이언트는 다른 곳.** 생성(`03_generate.sh`)과 벤치마크(`bench.sh`)는 HTTP 클라이언트라 원격 서버를 쓸 수 있습니다. GPU 호스트에서 `scripts/serve.sh up gen` 또는 `up mtp ...`로 서버를 띄우고, 클라이언트 쪽 `.env`를 이렇게 둡니다.

```bash
SERVER_MODE=external
VLLM_BASE_URL=http://gpu-host:18001     # 또는 ssh -L 18001:localhost:18001 gpu-host 로 터널을 열고 http://localhost:18001
VLLM_API_KEY=...                        # 서버를 키와 함께 띄웠다면
```

학습(`train.sh`, `04_train.sh`, `analyze.sh`)과 서빙 디렉터리 생성(`05_overlay.sh`)은 GPU와 체크포인트 파일에 직접 접근해야 하므로 GPU 호스트에서 실행합니다.

**이미 운영 중인 vLLM 서버로 측정.** 원래 MTP와 비교하려면 서버를 `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`로 띄우고 `SERVER_MODE=external`로 `bench.sh`만 돌립니다. 측정 중에는 다른 요청이 없어야 수용률 지표(`/metrics` 차이)가 정확합니다.

### 서버 프로필 값

`serve.sh up nospec|mtp`(벤치마크)와 `serve.sh up gen`(데이터 생성)이 쓰는 값입니다. 장비 메모리에 맞게 조정합니다.

| 변수 | 기본값 | 게시한 측정 |
| --- | --- | --- |
| `GPU_MEM_UTIL` | `0.80` | RTX PRO 6000 0.80, GB10 0.55 |
| `MAX_MODEL_LEN` | `131072` | 두 장비 모두 131072 |
| `MAX_NUM_SEQS` | `8` | RTX PRO 6000 8, GB10 1 |
| `CUDAGRAPH_MAX_SIZE` | `32` | RTX PRO 6000 32, GB10 24 |
| `NUM_SPEC_TOKENS` | `3` | 3 |
| `VLLM_EXTRA_ARGS` | 없음 | 그대로 vLLM 인자 뒤에 붙습니다. |
| `GEN_GPU_MEM_UTIL` / `GEN_MAX_MODEL_LEN` / `GEN_MAX_NUM_SEQS` / `GEN_CONCURRENCY` | `0.80` / `8192` / `128` / `96` | GB10에서 같은 값 |

### 학습 메모리

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `TRAIN_MEM_FRAC` | 비움(0.55) | PyTorch 프로세스 메모리 상한 비율. GB10은 0.55, 80GB 이상 단일 GPU는 0.9 정도로 둡니다. FP8 본체만 35.5GB를 씁니다. |
| `TRAIN_MEM_LIMIT` | 비움 | 학습 컨테이너의 `docker --memory`. GB10에서는 `96g`로 썼습니다. |
| `TRAIN_MIN_FREE_GB` | `0` | 호스트의 `MemAvailable`이 이 값(GiB)보다 작아지면 학습을 멈춥니다. GB10에서는 `16`으로 썼습니다. |

GB10(통합 메모리 128GB)에서 33B 모델을 제한 없이 올리면 호스트 전체가 멈출 수 있습니다. GB10에서는 `TRAIN_MEM_LIMIT=96g`, `TRAIN_MIN_FREE_GB=16`, `EXCLUSIVE_GPU=1`을 권장합니다.

## 4. 호스트 요구 사항

- Linux, Docker, NVIDIA Container Toolkit
- `bash`, `curl`(7.55 이상), `jq`, `bc`, `sha256sum`
- 디스크: 체크포인트 33GB, 원본 데이터 약 3GB, 생성 데이터 약 1GB, 학습 체크포인트 단계당 약 1GB × 저장 횟수
- Python은 호스트에 필요 없습니다. 데이터 준비와 생성 클라이언트는 `docker/tools.Dockerfile` 이미지(처음 실행할 때 자동 빌드)에서, 학습과 서빙은 `VLLM_IMAGE`에서 돌아갑니다.
