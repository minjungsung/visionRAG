# VisionRAG Makefile
# 사용법: make <명령어>
# 전체 목록: make help

.PHONY: help setup start stop ui api ingest test lint clean reset

help: ## 도움말
	@echo "사용법: make <명령어>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# === 처음 시작 ===

setup: ## 🚀 첫 실행 (인프라 + 초기화 + 데이터 + UI)
	docker compose up -d milvus minio redis
	@sleep 3
	PYTHONPATH=. python scripts/init_milvus.py
	PYTHONPATH=. python scripts/ingest_raw.py
	@echo ""
	@echo "✅ 셋업 완료! 'make ui' 로 UI 실행"

# === 실행 ===

start: ## 📦 Docker 인프라만 띄우기
	docker compose up -d milvus minio redis

stop: ## 🛑 Docker 인프라 내리기 (데이터 유지)
	docker compose down

ui: ## 🌐 Web UI 실행 (http://localhost:7860)
	PYTHONPATH=. python app.py

api: ## 📡 API 서버 실행 (http://localhost:8080)
	uvicorn src.api.main:app --reload --port 8080

# === 데이터 ===

ingest: ## 📄 data/raw 폴더 전체 인제스천
	PYTHONPATH=. python scripts/ingest_raw.py

ingest-file: ## 📄 파일 하나 인제스천 (FILE=경로)
	@test -n "$(FILE)" || (echo "사용법: make ingest-file FILE=data/raw/xxx.md" && exit 1)
	PYTHONPATH=. python -c "from src.ingestion.pipeline import IngestionPipeline; from pathlib import Path; p=IngestionPipeline(); print(p.ingest_file(Path('$(FILE)').name, Path('$(FILE)').read_bytes()))"

# === 검색 테스트 ===

search: ## 🔍 검색 테스트 (Q="질문")
	@test -n "$(Q)" || (echo "사용법: make search Q=\"렘은 누구야?\"" && exit 1)
	PYTHONPATH=. python -c "from src.retrieval.pipeline import RetrievalPipeline; r=RetrievalPipeline(); [print(f'{x[\"score\"]:.3f} | {x[\"text\"][:80]}') for x in r.search('$(Q)', top_k=5)]"

# === 개발 ===

test: ## 🧪 테스트 실행
	python -m pytest tests/ -v --tb=short

lint: ## 🔍 린트 + 포맷 체크
	ruff check src/ tests/ scripts/ workers/
	ruff format --check src/ tests/ scripts/ workers/

format: ## ✨ 자동 포맷
	ruff check --fix src/ tests/ scripts/ workers/
	ruff format src/ tests/ scripts/ workers/

# === 평가 ===

eval: ## 📊 검색 품질 평가
	PYTHONPATH=. python scripts/evaluate_retrieval.py

# === 관리 ===

reset: ## ⚠️  Milvus 데이터 초기화 (컬렉션 재생성)
	docker compose down milvus
	docker volume rm visionrag_milvus_data || true
	docker compose up -d milvus
	@sleep 5
	PYTHONPATH=. python scripts/init_milvus.py
	@echo "✅ 초기화 완료. 'make ingest'로 데이터 재투입"

clean: ## 🧹 캐시 정리
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache

logs: ## 📋 Docker 로그 보기
	docker compose logs -f --tail=50
