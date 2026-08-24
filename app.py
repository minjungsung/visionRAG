"""VisionRAG Web UI — FAISS 기반, 로컬 + HuggingFace Spaces 호환.

실행: python app.py
브라우저: http://localhost:7860
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

# Lazy init
_retriever = None
_rag = None


def get_retriever():
    global _retriever
    if _retriever is None:
        from src.retrieval.pipeline import RetrievalPipeline

        _retriever = RetrievalPipeline()
    return _retriever


def get_rag():
    global _rag
    if _rag is None:
        from src.retrieval.rag import RAGPipeline

        _rag = RAGPipeline()
    return _rag


# --- 질문하기 ---


def ask_ui(question, num_results, progress=gr.Progress()):
    if not question.strip():
        return "💡 질문을 입력하고 Enter!", ""

    progress(0.2, desc="🔍 관련 정보 찾는 중...")
    t0 = time.time()

    rag = get_rag()
    result = rag.answer(question, top_k=int(num_results))
    elapsed = time.time() - t0

    progress(1.0, desc="✅ 완료!")

    answer = result["answer"]
    answer += f"\n\n---\n_⏱️ {elapsed:.1f}초 · {len(result['sources'])}개 문서 참고_"

    sources = ""
    if result["sources"]:
        sources = "### 📚 참고 문서\n\n"
        for i, s in enumerate(result["sources"], 1):
            score = s.get("score", 0)
            icon = "🟢" if score > 0.5 else "🟡" if score > 0.3 else "🔴"
            sources += f"**{i}.** {icon} (관련도 {score:.0%})\n> {s['text'][:150]}...\n\n"

    return answer, sources


# --- 검색 ---


def search_ui(query, num_results, progress=gr.Progress()):
    if not query.strip():
        return "💡 검색어를 입력하고 Enter!"

    progress(0.2, desc="🔍 찾는 중...")
    t0 = time.time()

    retriever = get_retriever()
    results = retriever.search(query, top_k=int(num_results))
    elapsed = time.time() - t0

    progress(1.0, desc="✅ 완료!")

    if not results:
        return f"😅 '{query}'에 대한 결과를 찾지 못했어요."

    output = f'### 🔎 "{query}" ({len(results)}개, {elapsed:.1f}초)\n\n'
    for i, r in enumerate(results, 1):
        score = r["score"]
        icon = "🟢" if score > 0.5 else "🟡" if score > 0.3 else "🔴"
        output += f"---\n\n**{icon} {i}번째** (관련도 {score:.0%})\n\n{r['text'][:300]}\n\n"

    return output


# --- 파일 업로드 ---


def upload_ui(file, progress=gr.Progress()):
    if file is None:
        return "💡 파일을 선택해주세요!"

    from src.ingestion.pipeline import IngestionPipeline

    progress(0.3, desc="📄 분석 중...")
    try:
        pipeline = IngestionPipeline()
        file_path = Path(file.name)
        doc_id = pipeline.ingest_file(file_path.name, file_path.read_bytes())
        progress(1.0, desc="✅ 완료!")
        return f"### ✅ 업로드 성공!\n\n**{file_path.name}**이 등록됐어요.\n이제 검색에서 찾을 수 있습니다! 🎉"
    except Exception as e:
        return f"### ❌ 실패\n\n`{e}`"


# --- 도움말 ---

HELP_TEXT = """## 🎯 사용법

### 💬 질문하기
- "렘은 누구야?"
- "주술회전 영역전개 설명해줘"
- "이세계물 추천해줘"

### 🔎 검색
- "진격의 거인 세계관"
- "넨 시스템"
- "스탠드 능력"

### 📁 파일 업로드
.txt, .md, .pdf 파일을 올리면 바로 검색 가능!

### 💡 팁
- 결과 개수 늘리면 더 다양한 정보
- 캐릭터 이름, 능력명, 작품명 모두 검색 가능
"""

# --- UI ---

with gr.Blocks(title="VisionRAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 VisionRAG")
    gr.Markdown("궁금한 거 물어보세요! 100+ 애니/만화 작품에서 찾아드려요.")

    with gr.Tab("💬 질문하기"):
        q_input = gr.Textbox(label="질문 (Enter로 전송)", placeholder="예: 렘은 누구야?", lines=1)
        q_num = gr.Slider(1, 10, value=5, step=1, label="참고할 문서 수")
        q_btn = gr.Button("🚀 질문하기", variant="primary", size="lg")
        q_answer = gr.Markdown(value="*질문을 입력하고 Enter ↵*")
        q_sources = gr.Markdown()
        q_btn.click(ask_ui, [q_input, q_num], [q_answer, q_sources])
        q_input.submit(ask_ui, [q_input, q_num], [q_answer, q_sources])

    with gr.Tab("🔎 검색"):
        s_input = gr.Textbox(
            label="검색어 (Enter로 검색)", placeholder="예: 에반게리온 세계관", lines=1
        )
        s_num = gr.Slider(1, 20, value=5, step=1, label="결과 개수")
        s_btn = gr.Button("🔍 검색", variant="primary")
        s_output = gr.Markdown(value="*검색어를 입력하고 Enter ↵*")
        s_btn.click(search_ui, [s_input, s_num], [s_output])
        s_input.submit(search_ui, [s_input, s_num], [s_output])

    with gr.Tab("📁 파일 업로드"):
        gr.Markdown("파일 올리면 AI가 분석해서 검색 가능하게 만들어요.")
        u_file = gr.File(label="파일 (.txt .md .pdf)", file_types=[".txt", ".md", ".pdf"])
        u_btn = gr.Button("📤 업로드", variant="primary")
        u_output = gr.Markdown()
        u_btn.click(upload_ui, [u_file], [u_output])

    with gr.Tab("❓ 도움말"):
        gr.Markdown(HELP_TEXT)

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
