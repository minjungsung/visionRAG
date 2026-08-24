"""VisionRAG Web UI — Gradio 기반 질문/검색/평가 인터페이스.

실행:
    PYTHONPATH=. python app.py

브라우저에서:
    http://localhost:7860
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from src.config.settings import settings

# Lazy init (모델 로딩 시간 때문)
_rag = None
_retriever = None


def get_rag():
    global _rag
    if _rag is None:
        from src.retrieval.rag import RAGPipeline

        _rag = RAGPipeline()
    return _rag


def get_retriever():
    global _retriever
    if _retriever is None:
        from src.retrieval.pipeline import RetrievalPipeline

        _retriever = RetrievalPipeline()
    return _retriever


# --- Tab 1: RAG 질의 ---


def ask_question(question, prompt_type, top_k):
    """질문에 대한 RAG 답변 생성."""
    if not question.strip():
        return "질문을 입력하세요.", ""

    rag = get_rag()
    result = rag.answer(
        question,
        top_k=int(top_k),
        query_type=prompt_type if prompt_type != "default" else None,
    )

    answer = result["answer"]
    sources_text = ""
    for i, src in enumerate(result["sources"], 1):
        score = src.get("score", 0)
        text = src.get("text", "")[:200]
        sources_text += f"**[{i}]** (score: {score:.4f})\n{text}\n\n"

    return answer, sources_text


# --- Tab 2: 검색만 ---


def search_only(query, top_k):
    """검색만 수행 (답변 생성 없이)."""
    if not query.strip():
        return ""

    retriever = get_retriever()
    results = retriever.search(query, top_k=int(top_k))

    output = ""
    for i, r in enumerate(results, 1):
        output += f"### [{i}] score: {r['score']:.4f}\n"
        output += f"doc_id: `{r.get('doc_id', 'N/A')}` | page: {r.get('page_num', 'N/A')}\n\n"
        output += f"{r['text'][:300]}\n\n---\n\n"

    return output if output else "결과 없음"


# --- Tab 3: 파일 업로드 ---


def upload_file(file):
    """파일을 인제스천 파이프라인에 넣기."""
    if file is None:
        return "파일을 선택하세요."

    from pathlib import Path

    from src.ingestion.pipeline import IngestionPipeline

    try:
        pipeline = IngestionPipeline()
        file_path = Path(file.name)
        file_bytes = file_path.read_bytes()
        doc_id = pipeline.ingest_file(file_path.name, file_bytes)
        return f"✅ 업로드 완료!\n\n**파일**: {file_path.name}\n**문서 ID**: `{doc_id}`\n\n이제 검색/질의에서 이 문서를 찾을 수 있습니다."
    except Exception as e:
        return f"❌ 업로드 실패: {e}"


# --- Tab 4: 시스템 상태 ---


def get_system_status():
    """시스템 구성 정보 표시."""
    status = f"""## 시스템 설정

| 항목 | 값 |
|------|------|
| Milvus | `{settings.milvus_host}:{settings.milvus_port}` |
| Triton | `{settings.triton_url}` (사용: {settings.use_triton}) |
| OpenAI Model | `{settings.openai_model}` |
| OpenAI Base URL | `{settings.openai_base_url[:50]}...` |
| Rewrite Strategy | `{settings.rewrite_strategy}` |
| LangSmith | {"✅ 활성" if settings.langsmith_tracing else "❌ 비활성"} |

## 사용법

### 질문하기 (RAG)
1. "RAG 질의" 탭에서 질문 입력
2. 프롬프트 타입 선택 (optional)
3. "질문하기" 클릭 → 답변 + 출처 표시

### 검색만
1. "검색" 탭에서 쿼리 입력
2. 검색 결과만 확인 (답변 생성 X, 더 빠름)

### 품질 평가
터미널에서:
```bash
# 검색 품질 비교 (rewrite 전략)
PYTHONPATH=. python scripts/experiment_rewrite.py

# End-to-end RAG 평가 (LLM-as-judge)
PYTHONPATH=. python scripts/evaluate_rag_simple.py

# DVC 파이프라인 (retrieval 메트릭)
dvc repro evaluate_retrieval
```
"""
    return status


# --- Gradio UI ---

with gr.Blocks(title="VisionRAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 VisionRAG — 멀티모달 지식 관리 시스템")
    gr.Markdown("텍스트 + 이미지 통합 검색 및 AI 답변 생성")

    with gr.Tab("💬 RAG 질의"):
        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="질문",
                    placeholder="예: VisionRAG에서 이미지 검색은 어떤 모델을 사용하나요?",
                    lines=2,
                )
            with gr.Column(scale=1):
                prompt_type = gr.Dropdown(
                    choices=["default", "factual", "analytical", "comparative", "how_to"],
                    value="default",
                    label="프롬프트 타입",
                )
                top_k_rag = gr.Slider(1, 20, value=5, step=1, label="Top-K")

        ask_btn = gr.Button("🚀 질문하기", variant="primary")

        answer_output = gr.Markdown(label="답변")
        sources_output = gr.Markdown(label="출처 문서")

        ask_btn.click(
            ask_question,
            inputs=[question_input, prompt_type, top_k_rag],
            outputs=[answer_output, sources_output],
        )
        question_input.submit(
            ask_question,
            inputs=[question_input, prompt_type, top_k_rag],
            outputs=[answer_output, sources_output],
        )

    with gr.Tab("🔎 검색"):
        search_input = gr.Textbox(
            label="검색 쿼리",
            placeholder="예: Milvus HNSW 인덱스 설정",
            lines=1,
        )
        top_k_search = gr.Slider(1, 20, value=10, step=1, label="Top-K")
        search_btn = gr.Button("검색", variant="primary")
        search_output = gr.Markdown(label="검색 결과")

        search_btn.click(search_only, inputs=[search_input, top_k_search], outputs=[search_output])
        search_input.submit(
            search_only, inputs=[search_input, top_k_search], outputs=[search_output]
        )

    with gr.Tab("📁 파일 업로드"):
        gr.Markdown("문서를 업로드하면 자동으로 임베딩 생성 후 검색 가능해집니다.")
        upload_input = gr.File(
            label="파일 선택",
            file_types=[".txt", ".md", ".pdf"],
        )
        upload_btn = gr.Button("📤 업로드 & 인제스천", variant="primary")
        upload_output = gr.Markdown(label="결과")

        upload_btn.click(upload_file, inputs=[upload_input], outputs=[upload_output])

    with gr.Tab("ℹ️ 시스템 정보"):
        status_output = gr.Markdown(value=get_system_status())
        refresh_btn = gr.Button("새로고침")
        refresh_btn.click(get_system_status, outputs=[status_output])


if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
