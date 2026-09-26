# Setup Guide

> Docker Compose 실행, 로컬 개발 환경, HuggingFace Spaces 배포 가이드

## 사전 요구사항

| 도구 | 최소 버전 | 용도 |
|------|----------|------|
| Python | 3.10+ | 런타임 |
| Docker | 24.0+ | 컨테이너 실행 |
| Docker Compose | v2.0+ | 멀티 컨테이너 관리 |
| Git | 2.30+ | 소스 관리 |
| DVC | 3.0+ | 데이터 버전 관리 |
| Node.js | 18+ (선택) | 프론트엔드 (있는 경우) |

## Docker Compose로 전체 스택 실행

### 1단계: 리포지토리 클론

```bash
git clone https://github.com/minjungsung/visionRAG.git
cd visionRAG
```

### 2단계: 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# 필수 변수 편집
vi .env
```

`.env` 파일 예시:

```env
# API 설정
API_HOST=0.0.0.0
API_PORT=8000

# 벡터 DB 설정
VECTOR_DB_TYPE=milvus
MILVUS_HOST=milvus-standalone
MILVUS_PORT=19530

# LLM 설정
OPENAI_API_KEY=sk-your-key-here

# 모니터링
GRAFANA_ADMIN_PASSWORD=admin123
```

### 3단계: Docker Compose 실행

```bash
# 전체 서비스 시작 (백그라운드)
docker compose up -d

# 로그 확인
docker compose logs -f fastapi-server

# 서비스 상태 확인
docker compose ps
```

### 4단계: 서비스 접속

| 서비스 | URL | 비고 |
|--------|-----|------|
| FastAPI 서버 | http://localhost:8000 | API 문서: `/docs` |
| Grafana | http://localhost:3000 | admin / admin123 |
| Prometheus | http://localhost:9090 | 메트릭 조회 |
| Milvus | localhost:19530 | gRPC 포트 |
| MinIO Console | http://localhost:9001 | 오브젝트 스토리지 |

### 서비스 종료

```bash
# 전체 서비스 종료
docker compose down

# 볼륨 포함 종료 (데이터 삭제)
docker compose down -v
```

## 로컬 개발 환경

Docker 없이 로컬에서 직접 실행하려면:

### Python 환경 설정

```bash
# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 개발 도구
```

### FAISS 로컬 모드로 실행

```bash
# .env에서 벡터 DB를 FAISS로 설정
export VECTOR_DB_TYPE=faiss

# API 서버 실행
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### DVC 데이터 준비

```bash
# DVC 데이터 가져오기
dvc pull

# 또는 파이프라인 처음부터 실행
dvc repro
```

### 테스트 실행

```bash
# 전체 테스트
pytest

# 커버리지 포함
pytest --cov=src --cov-report=html

# 특정 테스트만
pytest tests/test_api.py -v
```

## HuggingFace Spaces 배포

visionRAG는 `spaces/` 디렉토리를 통해 HuggingFace Spaces에 데모를 배포할 수 있습니다.

### Spaces 구조

```
spaces/
├── app.py              # Gradio/Streamlit 앱 진입점
├── requirements.txt    # Spaces용 의존성
└── README.md           # HuggingFace Space 카드
```

### 배포 방법

```bash
# HuggingFace CLI 설치 및 로그인
pip install huggingface_hub
huggingface-cli login

# Space 생성 (처음 한 번)
huggingface-cli repo create visionRAG --type space --space_sdk gradio

# spaces/ 디렉토리를 Space에 푸시
cd spaces
git init
git remote add space https://huggingface.co/spaces/your-username/visionRAG
git add .
git commit -m "initial space deployment"
git push space main
```

### Spaces 환경 변수

HuggingFace Spaces 설정에서 다음 Secrets을 추가하세요:

| 변수명 | 설명 |
|--------|------|
| `OPENAI_API_KEY` | LLM API 키 |
| `VECTOR_INDEX_PATH` | 사전 빌드된 인덱스 경로 |

## 문제 해결

| 문제 | 해결 방법 |
|------|----------|
| Milvus 연결 실패 | `docker compose ps`로 상태 확인, etcd/minio가 healthy인지 확인 |
| DVC pull 실패 | AWS 자격증명 설정 확인: `aws configure` |
| GPU 메모리 부족 | `EMBEDDING_BATCH_SIZE` 줄이기 또는 CPU 모드 사용 |
| 포트 충돌 | `.env`에서 포트 변경 또는 기존 프로세스 종료 |