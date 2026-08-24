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


# --- Tab 1: 질문하기 ---


def ask_question(question, answer_style, num_results, progress=gr.Progress()):
    """질문에 대한 AI 답변 생성."""
    if not question.strip():
        return "💡 질문을 입력하고 Enter를 눌러주세요!", ""

    progress(0.1, desc="🔍 관련 정보 찾는 중...")
    t0 = time.time()

    rag = get_rag()

    # 답변 스타일 → 내부 프롬프트 타입 변환
    style_map = {
        "기본": None,
        "사실 위주로 정확하게": "factual",
        "분석적으로 자세하게": "analytical",
        "비교해서 설명": "comparative",
        "방법/순서 알려줘": "how_to",
    }

    progress(0.4, desc="💭 답변 만드는 중...")
    result = rag.answer(
        question,
        top_k=int(num_results),
        query_type=style_map.get(answer_style),
    )

    elapsed = time.time() - t0
    progress(1.0, desc="✅ 완료!")

    answer = result["answer"]
    if elapsed > 0:
        answer += f"\n\n---\n_⏱️ {elapsed:.1f}초 걸림 · {len(result['sources'])}개 문서 참고_"

    sources_text = ""
    if result["sources"]:
        sources_text = "### 📚 참고한 문서들\n\n"
        for i, src in enumerate(result["sources"], 1):
            score = src.get("score", 0)
            text = src.get("text", "")[:150]
            relevance = "🟢 높음" if score > 0.5 else "🟡 보통" if score > 0.3 else "🔴 낮음"
            sources_text += f"**{i}.** ({relevance})\n> {text}...\n\n"

    return answer, sources_text if sources_text else "_참고할 문서를 찾지 못했어요_"


# --- Tab 2: 검색 ---


def search_only(query, num_results, progress=gr.Progress()):
    """문서 검색만 수행."""
    if not query.strip():
        return "💡 검색어를 입력하고 Enter를 눌러주세요!"

    progress(0.2, desc="🔍 찾는 중...")
    t0 = time.time()

    retriever = get_retriever()
    results = retriever.search(query, top_k=int(num_results))

    elapsed = time.time() - t0
    progress(1.0, desc="✅ 완료!")

    if not results:
        return f"😅 '{query}'에 대한 결과를 찾지 못했어요. 다른 키워드로 시도해보세요!"

    output = f'### 🔎 "{query}" 검색 결과 ({len(results)}개, {elapsed:.1f}초)\n\n'
    for i, r in enumerate(results, 1):
        score = r["score"]
        relevance = "🟢" if score > 0.5 else "🟡" if score > 0.3 else "🔴"
        text = r["text"][:250]
        output += f"---\n\n**{relevance} {i}번째 결과** (관련도 {score:.0%})\n\n{text}\n\n"

    return output


# --- Tab 3: 파일 업로드 ---


def upload_file(file, progress=gr.Progress()):
    """파일 업로드 → 검색 가능하게 만들기."""
    if file is None:
        return "💡 파일을 선택해주세요!"

    from src.ingestion.pipeline import IngestionPipeline

    progress(0.2, desc="📄 파일 읽는 중...")

    try:
        pipeline = IngestionPipeline()
        file_path = Path(file.name)
        file_bytes = file_path.read_bytes()
        size_kb = len(file_bytes) / 1024

        progress(0.5, desc="🧮 AI가 내용 분석 중...")
        doc_id = pipeline.ingest_file(file_path.name, file_bytes)

        progress(1.0, desc="✅ 완료!")
        return (
            f"### ✅ 업로드 성공!\n\n"
            f"**{file_path.name}** ({size_kb:.1f} KB)가 등록됐어요.\n\n"
            f"이제 **검색** 탭에서 이 파일의 내용을 찾을 수 있습니다! 🎉"
        )
    except Exception as e:
        return f"### ❌ 실패\n\n뭔가 잘못됐어요: `{e}`"


# --- Tab 4: 도움말 ---


def get_help():
    return """## 🎯 사용법

### 💬 질문하기
AI가 등록된 문서에서 답을 찾아 답변해줘요.
- "렘은 누구야?"
- "주술회전 영역전개 설명해줘"
- "이세계물 추천해줘"

### 🔎 검색
관련 문서만 빠르게 찾을 때. AI 답변 없이 검색만 해요.
- "진격의 거인 세계관"
- "호흡법 종류"

### 📁 파일 업로드
새로운 문서를 추가할 수 있어요.
- .txt, .md, .pdf 파일 지원
- 올린 파일은 바로 검색/질문에 반영됨

### 💡 팁
- **결과 개수**: 많을수록 더 다양한 답, 적을수록 더 정확한 답
- **답변 스타일**: "사실 위주로"는 팩트 체크, "비교해서"는 여러 작품 비교할 때
- 검색이 안 되면 다른 키워드로 바꿔보세요 (예: "SAO" → "소드 아트 온라인")
"""


# --- Gradio UI ---

with gr.Blocks(title="VisionRAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 VisionRAG")
    gr.Markdown("궁금한 거 물어보세요! 등록된 문서에서 답을 찾아드려요.")

    with gr.Tab("💬 질문하기"):
        question_input = gr.Textbox(
            label="질문 (Enter로 전송)",
            placeholder="예: 렘은 누구야? / 주술회전 최강 캐릭터는?",
            lines=2,
        )
        with gr.Row():
            answer_style = gr.Dropdown(
                choices=[
                    "기본",
                    "사실 위주로 정확하게",
                    "분석적으로 자세하게",
                    "비교해서 설명",
                    "방법/순서 알려줘",
                ],
                value="기본",
                label="답변 스타일",
            )
            num_results_rag = gr.Slider(
                1, 10, value=5, step=1, label="참고할 문서 수 (많을수록 넓게 검색)"
            )

        ask_btn = gr.Button("🚀 질문하기", variant="primary", size="lg")

        answer_output = gr.Markdown(value="*질문을 입력하고 Enter를 눌러보세요! ↵*")
        sources_output = gr.Markdown()

        ask_btn.click(
            ask_question,
            inputs=[question_input, answer_style, num_results_rag],
            outputs=[answer_output, sources_output],
        )
        question_input.submit(
            ask_question,
            inputs=[question_input, answer_style, num_results_rag],
            outputs=[answer_output, sources_output],
        )

    with gr.Tab("🔎 검색"):
        search_input = gr.Textbox(
            label="검색어 (Enter로 검색)",
            placeholder="예: 에반게리온 세계관 / 이세계 작품 / 스탠드 능력",
            lines=1,
        )
        num_results_search = gr.Slider(1, 20, value=5, step=1, label="결과 개수")
        search_btn = gr.Button("🔍 검색", variant="primary")
        search_output = gr.Markdown(value="*검색어를 입력하고 Enter! ↵*")

        search_btn.click(
            search_only, inputs=[search_input, num_results_search], outputs=[search_output]
        )
        search_input.submit(
            search_only, inputs=[search_input, num_results_search], outputs=[search_output]
        )

    with gr.Tab("📁 파일 업로드"):
        gr.Markdown("새 문서를 올리면 AI가 내용을 분석해서 검색 가능하게 만들어요.")
        upload_input = gr.File(
            label="파일 선택 (.txt .md .pdf)",
            file_types=[".txt", ".md", ".pdf"],
        )
        upload_btn = gr.Button("📤 업로드", variant="primary")
        upload_output = gr.Markdown()

        upload_btn.click(upload_file, inputs=[upload_input], outputs=[upload_output])

    with gr.Tab("❓ 도움말"):
        gr.Markdown(value=get_help())


if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
