# VisionRAG Makefile
# 사용법: make <명령어>
# 전체 목록: make help

.PHONY: help setup ui ingest search test lint format clean

help: ## 도움말
	@echo "사용법: make <명령어>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# === 처음 시작 ===

setup: ## 🚀 첫 실행 (데이터 인제스천 + UI 실행)
	PYTHONPATH=. python scripts/ingest_raw.py
	@echo ""
	@echo "✅ 준비 완료! 'make ui' 로 UI 실행"

# === 실행 ===

ui: ## 🌐 Web UI (http://localhost:7860)
	PYTHONPATH=. python app.py

api: ## 📡 API 서버 (http://localhost:8080)
	uvicorn src.api.main:app --reload --port 8080

# === 데이터 ===

ingest: ## 📄 data/raw 전체 인제스천
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

lint: ## 🔍 린트 체크
	ruff check src/ tests/ scripts/ workers/
	ruff format --check src/ tests/ scripts/ workers/

format: ## ✨ 자동 포맷
	ruff check --fix src/ tests/ scripts/ workers/ app.py
	ruff format src/ tests/ scripts/ workers/ app.py

# === 관리 ===

clean: ## 🧹 인덱스 + 캐시 초기화
	rm -rf data/index/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ 초기화 완료. 'make ingest'로 재빌드"
