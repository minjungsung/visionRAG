#!/bin/bash
# VisionRAG 원클릭 실행 스크립트
# 사용법: ./run.sh

set -e
cd "$(dirname "$0")"

echo "🚀 VisionRAG 시작"
echo ""

# 1. Docker 인프라
echo "📦 Docker 인프라 확인..."
docker compose up -d milvus minio redis
sleep 3

# 2. 가상환경 활성화
source .venv/bin/activate

# 3. Milvus 컬렉션 초기화
echo ""
echo "🗄️  Milvus 컬렉션 초기화..."
PYTHONPATH=. python scripts/init_milvus.py

# 4. 데이터 인제스천 (data/raw에 파일 있으면)
echo ""
echo "📄 데이터 인제스천..."
PYTHONPATH=. python scripts/ingest_raw.py

# 5. Web UI 실행
echo ""
echo "✅ 준비 완료!"
echo ""
echo "🌐 Web UI: http://localhost:7860"
echo "📡 API:    http://localhost:8080"
echo ""
echo "UI 실행중... (Ctrl+C로 종료)"
PYTHONPATH=. python app.py
