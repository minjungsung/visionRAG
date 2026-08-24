"""VisionRAG — HuggingFace Spaces 버전 (Milvus 불필요, 파일 기반 검색)

이 파일을 HuggingFace Spaces에 app.py로 올리면 됩니다.
임베딩을 미리 계산해서 .npz 파일로 저장, numpy cosine similarity로 검색.
"""

import json
import time
from pathlib import Path

import gradio as gr
import numpy as np

# --- 데이터 로드 ---

DATA_DIR = Path("data")
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npz"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"

_chunks = None
_embeddings = None
_model = None


def get_model():
    """임베딩 모델 로드 (최초 1회)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("BAAI/bge-m3")
    return _model


def load_data():
    """미리 계산된 임베딩 + 청크 로드."""
    global _chunks, _embeddings

    if _chunks is not None:
        return

    # 청크 로드
    _chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                _chunks.append(json.loads(line))

    # 임베딩 로드
    data = np.load(EMBEDDINGS_FILE)
    _embeddings = data["embeddings"].astype(np.float32)

    print(f"✅ 로드 완료: {len(_chunks)}개 청크, 임베딩 shape={_embeddings.shape}")


def search(query: str, top_k: int = 5) -> list[dict]:
    """Numpy cosine similarity 기반 검색."""
    load_data()
    model = get_model()

    # 쿼리 임베딩
    query_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)

    # 코사인 유사도
    scores = (_embeddings @ query_emb.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "text": _chunks[idx]["text"],
            "score": float(scores[idx]),
            "source": _chunks[idx].get("source", ""),
        })
    return results


# --- UI 함수 ---


def search_ui(query, num_results, progress=gr.Progress()):
    """검색 UI."""
    if not query.strip():
        return "💡 검색어를 입력하고 Enter를 눌러주세요!"

    progress(0.3, desc="🔍 찾는 중...")
    t0 = time.time()
    results = search(query, top_k=int(num_results))
    elapsed = time.time() - t0
    progress(1.0, desc="✅ 완료!")

    if not results:
        return f"😅 '{query}'에 대한 결과를 찾지 못했어요."

    output = f"### 🔎 \"{query}\" 검색 결과 ({len(results)}개, {elapsed:.1f}초)\n\n"
    for i, r in enumerate(results, 1):
        score = r["score"]
        relevance = "🟢" if score > 0.5 else "🟡" if score > 0.3 else "🔴"
        text = r["text"][:300]
        output += f"---\n\n**{relevance} {i}번째** (관련도 {score:.0%})\n\n{text}\n\n"

    return output


def ask_ui(question, answer_style, num_results, progress=gr.Progress()):
    """질문 UI — 검색 결과 기반 답변 (LLM 없이 검색만)."""
    if not question.strip():
        return "💡 질문을 입력하고 Enter를 눌러주세요!", ""

    progress(0.3, desc="🔍 관련 정보 찾는 중...")
    t0 = time.time()
    results = search(question, top_k=int(num_results))
    elapsed = time.time() - t0
    progress(1.0, desc="✅ 완료!")

    if not results:
        return "😅 관련 정보를 찾지 못했어요.", ""

    # LLM 없이 검색 결과를 답변으로 표시
    answer = f"### 💬 \"{question}\"에 대한 관련 정보\n\n"
    for i, r in enumerate(results[:3], 1):
        answer += f"**{i}.** {r['text'][:200]}\n\n"
    answer += f"\n_⏱️ {elapsed:.1f}초 · {len(results)}개 문서 참고_"

    sources = ""
    for i, r in enumerate(results, 1):
        relevance = "🟢" if r["score"] > 0.5 else "🟡" if r["score"] > 0.3 else "🔴"
        sources += f"**{i}.** {relevance} (관련도 {r['score']:.0%})\n> {r['text'][:100]}...\n\n"

    return answer, sources


def get_help():
    return """## 🎯 사용법

### 💬 질문하기
등록된 애니/만화 정보에서 관련 내용을 찾아줘요.
- "렘은 누구야?"
- "주술회전 영역전개 설명해줘"
- "이세계물 추천해줘"

### 🔎 검색
키워드로 빠르게 찾을 때.
- "진격의 거인 세계관"
- "호흡법 종류"
- "스탠드 능력"

### 💡 팁
- 결과 개수를 늘리면 더 다양한 정보가 나와요
- 일본어 제목으로도 검색 가능 (예: "鬼滅の刃")
- 캐릭터 이름, 능력명, 작품명 모두 검색 가능
"""


# --- Gradio UI ---

with gr.Blocks(title="VisionRAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 VisionRAG — 애니/만화 지식 검색")
    gr.Markdown("궁금한 거 물어보세요! 100+ 작품 정보에서 찾아드려요.")

    with gr.Tab("💬 질문하기"):
        question_input = gr.Textbox(
            label="질문 (Enter로 전송)",
            placeholder="예: 렘은 누구야? / 고조 사토루 능력은?",
            lines=1,
        )
        with gr.Row():
            answer_style = gr.Dropdown(
                choices=["기본", "사실 위주로", "자세하게", "비교해서"],
                value="기본",
                label="답변 스타일",
            )
            num_results_q = gr.Slider(1, 10, value=5, step=1, label="참고할 문서 수")

        ask_btn = gr.Button("🚀 질문하기", variant="primary", size="lg")
        answer_output = gr.Markdown(value="*질문을 입력하고 Enter ↵*")
        sources_output = gr.Markdown()

        ask_btn.click(ask_ui, inputs=[question_input, answer_style, num_results_q], outputs=[answer_output, sources_output])
        question_input.submit(ask_ui, inputs=[question_input, answer_style, num_results_q], outputs=[answer_output, sources_output])

    with gr.Tab("🔎 검색"):
        search_input = gr.Textbox(
            label="검색어 (Enter로 검색)",
            placeholder="예: 에반게리온 세계관 / 넨 시스템 / 악마의 열매",
            lines=1,
        )
        num_results_s = gr.Slider(1, 20, value=5, step=1, label="결과 개수")
        search_btn = gr.Button("🔍 검색", variant="primary")
        search_output = gr.Markdown(value="*검색어를 입력하고 Enter ↵*")

        search_btn.click(search_ui, inputs=[search_input, num_results_s], outputs=[search_output])
        search_input.submit(search_ui, inputs=[search_input, num_results_s], outputs=[search_output])

    with gr.Tab("❓ 도움말"):
        gr.Markdown(value=get_help())


if __name__ == "__main__":
    demo.launch()
