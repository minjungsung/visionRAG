# visionRAG Wiki

> Multimodal RAG (Retrieval-Augmented Generation) 시스템 — 이미지와 텍스트를 통합하는 검색 증강 생성 파이프라인

## 프로젝트 개요

visionRAG는 **멀티모달 RAG 파이프라인**을 구현하는 프로젝트입니다. 텍스트뿐만 아니라 이미지 데이터를 함께 처리하여, 사용자 질의에 대해 관련 문서와 이미지를 검색하고 LLM을 통해 답변을 생성합니다.

### 핵심 기능
- **멀티모달 임베딩**: 텍스트와 이미지를 통합 벡터 공간에 임베딩
- **벡터 검색**: FAISS 및 Milvus 기반 유사도 검색
- **RAG 파이프라인**: Ingest → Chunk → Embed → Index → Query → Generate
- **데이터 버전 관리**: DVC를 통한 데이터셋 및 모델 버전 관리
- **모니터링**: Prometheus/Grafana 기반 시스템 메트릭 대시보드
- **API 서비스**: FastAPI 기반 REST API 서버

## 기술 스택

| 카테고리 | 기술 |
|---------|------|
| 언어 | Python 3.10+ |
| API 서버 | FastAPI, Uvicorn |
| 벡터 DB | FAISS, Milvus |
| 데이터 관리 | DVC (Data Version Control) |
| 컨테이너 | Docker, Docker Compose |
| 모니터링 | Prometheus, Grafana |
| 배포 | HuggingFace Spaces |
| 테스트 | pytest |

## 프로젝트 구조

```
visionRAG/
├── src/
│   ├── api/           # FastAPI 서버 엔드포인트
│   ├── ingestion/     # 데이터 수집 및 처리 파이프라인
│   ├── config/        # 설정 파일
│   └── ...
├── scripts/           # 유틸리티 스크립트
├── spaces/            # HuggingFace Spaces 배포 파일
├── docs/              # 프로젝트 문서 (ARCHITECTURE.md, COMPONENTS.md 등)
├── docker-compose.yml
├── dvc.yaml           # DVC 파이프라인 정의
└── requirements.txt
```

## Wiki 목차

| 페이지 | 설명 |
|--------|------|
| [Architecture](Architecture) | RAG 파이프라인 아키텍처, 컴포넌트 다이어그램, Docker Compose 구성 |
| [Services](Services) | FastAPI 서버, Ingestion Pipeline, Vector Store, Evaluation |
| [Data Pipeline](Data-Pipeline) | DVC 데이터 관리, 파이프라인 정의, 데이터 폴더 구조 |
| [Setup Guide](Setup-Guide) | Docker Compose 실행, 로컬 개발, HuggingFace Spaces 배포 |
| [Monitoring](Monitoring) | Prometheus/Grafana 대시보드, 메트릭 설명 |

## 빠른 시작

```bash
# 리포지토리 클론
git clone https://github.com/minjungsung/visionRAG.git
cd visionRAG

# Docker Compose로 전체 스택 실행
docker compose up -d

# 또는 로컬 개발 환경
pip install -r requirements.txt
python -m src.api.main
```

자세한 설정 방법은 [Setup Guide](Setup-Guide)를 참조하세요.
