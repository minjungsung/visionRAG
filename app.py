"""VisionRAG Web UI — Gradio 기반 질문/검색/평가 인터페이스.

실행:
    PYTHONPATH=. python app.py

브라우저에서:
    http://localhost:7860
"""

import sys
import time
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


def ask_question(question, prompt_type, top_k, progress=gr.Progress()):
    """질문에 대한 RAG 답변 생성."""
    if not question.strip():
        return "❌ 질문을 입력하세요.", ""

    progress(0.1, desc="🔍 관련 문서 검색 중...")
    t0 = time.time()

    rag = get_rag()

    progress(0.4, desc="📄 문서에서 답변 생성 중...")
    result = rag.answer(
        question,
        top_k=int(top_k),
        query_type=prompt_type if prompt_type != "default" else None,
    )

    elapsed = time.time() - t0
    progress(1.0, desc="✅ 완료!")

    answer = (
        f"{result['answer']}\n\n---\n⏱️ {elapsed:.1f}초 소요 | {len(result['sources'])}개 문서 참조"
    )

    sources_text = ""
    for i, src in enumerate(result["sources"], 1):
        score = src.get("score", 0)
        text = src.get("text", "")[:200]
        sources_text += f"**[{i}]** (유사도: {score:.4f})\n{text}\n\n"

    return answer, sources_text if sources_text else "참조 문서 없음"


# --- Tab 2: 검색만 ---


def search_only(query, top_k, progress=gr.Progress()):
    """검색만 수행 (답변 생성 없이)."""
    if not query.strip():
        return "❌ 검색어를 입력하세요."

    progress(0.2, desc="🔍 벡터 DB 검색 중...")
    t0 = time.time()

    retriever = get_retriever()
    results = retriever.search(query, top_k=int(top_k))

    elapsed = time.time() - t0
    progress(1.0, desc="✅ 완료!")

    if not results:
        return f"검색 결과 없음 (⏱️ {elapsed:.1f}초)"

    output = f"**{len(results)}개 결과** (⏱️ {elapsed:.1f}초)\n\n---\n\n"
    for i, r in enumerate(results, 1):
        score_bar = "🟢" if r["score"] > 0.5 else "🟡" if r["score"] > 0.3 else "🔴"
        output += f"### {score_bar} [{i}] 유사도: {r['score']:.4f}\n"
        output += f"{r['text'][:300]}\n\n---\n\n"

    return output


# --- Tab 3: 파일 업로드 ---


def upload_file(file, progress=gr.Progress()):
    """파일을 인제스천 파이프라인에 넣기."""
    if file is None:
        return "❌ 파일을 선택하세요."

    from src.ingestion.pipeline import IngestionPipeline

    progress(0.2, desc="📄 파일 파싱 중...")

    try:
        pipeline = IngestionPipeline()
        file_path = Path(file.name)
        file_bytes = file_path.read_bytes()

        progress(0.5, desc="🧮 임베딩 생성 중...")
        doc_id = pipeline.ingest_file(file_path.name, file_bytes)

        progress(1.0, desc="✅ 완료!")
        return (
            f"✅ **업로드 완료!**\n\n"
            f"| 항목 | 값 |\n|------|------|\n"
            f"| 파일 | `{file_path.name}` |\n"
            f"| 크기 | {len(file_bytes):,} bytes |\n"
            f"| 문서 ID | `{doc_id}` |\n\n"
            f"이제 **검색 탭**에서 이 문서를 찾을 수 있습니다."
        )
    except Exception as e:
        return f"❌ **업로드 실패**\n\n```\n{e}\n```"


# --- Tab 4: 시스템 상태 ---


def get_system_status():
    """시스템 구성 정보 표시."""
    status = f"""## ⚙️ 시스템 설정

| 항목 | 값 | 상태 |
|------|------|------|
| Milvus | `{settings.milvus_host}:{settings.milvus_port}` | 🟢 |
| Triton | `{settings.triton_url}` | {"🟢 사용중" if settings.use_triton else "⚪ 미사용 (로컬 모델)"} |
| 임베딩 모델 | BGE-M3 (1024차원) | 🟢 |
| LangSmith | {settings.langsmith_project} | {"🟢 활성" if settings.langsmith_tracing else "⚪ 비활성"} |

## 📖 사용법

| 탭 | 기능 | 설명 |
|----|------|------|
| 💬 RAG 질의 | 질문 → AI 답변 | 문서 검색 + 답변 생성 (OpenAI 필요) |
| 🔎 검색 | 쿼리 → 관련 문서 | 벡터 유사도 검색만 (빠름) |
| 📁 파일 업로드 | 파일 → DB 저장 | txt/md/pdf 지원 |

**팁**: 검색 탭에서 엔터 치면 바로 검색됩니다.
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
                    placeholder="예: 렘은 누구야? / 고조 사토루 능력 알려줘",
                    lines=2,
                )
            with gr.Column(scale=1):
                prompt_type = gr.Dropdown(
                    choices=["default", "factual", "analytical", "comparative", "how_to"],
                    value="default",
                    label="프롬프트 타입",
                )
                top_k_rag = gr.Slider(1, 20, value=5, step=1, label="Top-K")

        ask_btn = gr.Button("🚀 질문하기 (Enter로도 가능)", variant="primary")

        answer_output = gr.Markdown(
            label="답변", value="*질문을 입력하고 Enter 또는 버튼을 클릭하세요*"
        )
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
            label="검색 쿼리 (Enter로 검색)",
            placeholder="예: 스쿠나 능력 / 에반게리온 세계관 / 이세계 추천",
            lines=1,
        )
        top_k_search = gr.Slider(1, 20, value=5, step=1, label="Top-K")
        search_btn = gr.Button("🔍 검색", variant="primary")
        search_output = gr.Markdown(label="검색 결과", value="*검색어를 입력하고 Enter를 누르세요*")

        search_btn.click(search_only, inputs=[search_input, top_k_search], outputs=[search_output])
        search_input.submit(
            search_only, inputs=[search_input, top_k_search], outputs=[search_output]
        )

    with gr.Tab("📁 파일 업로드"):
        gr.Markdown("문서를 업로드하면 자동으로 텍스트 추출 → 임베딩 생성 → DB 저장됩니다.")
        upload_input = gr.File(
            label="파일 선택 (.txt, .md, .pdf)",
            file_types=[".txt", ".md", ".pdf"],
        )
        upload_btn = gr.Button("📤 업로드 & 인제스천", variant="primary")
        upload_output = gr.Markdown(label="결과")

        upload_btn.click(upload_file, inputs=[upload_input], outputs=[upload_output])

    with gr.Tab("ℹ️ 시스템 정보"):
        status_output = gr.Markdown(value=get_system_status())
        refresh_btn = gr.Button("🔄 새로고침")
        refresh_btn.click(get_system_status, outputs=[status_output])


if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
