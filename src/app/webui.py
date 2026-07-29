import time

import streamlit as st

from app import store
from app.config import LLM_DEFAULT_TIER, LLM_TIERS, RUNTIME_VARIANT, resolve_model
from app.generation import REFUSAL
from app.pipeline import Answer, answer

# One-word tier descriptions, shown as the radio options' captions.
TIER_HINT = {"turbo": "daha hızlı", "large": "reasoning"}

# on_stage keys (pipeline/generation) -> the Turkish progress labels.
STAGE_LABELS = {
    "retrieval": "Belgeler aranıyor (retrieval)",
    "rerank": "Pasajlar yeniden sıralanıyor (cross-encoder)",
    "generate": "Cevap üretiliyor (LLM)",
    "thinking": "Model düşünüyor",
    "writing": "Cevap yazılıyor",
}

st.set_page_config(page_title="Doküman Bilgi Erişimi", page_icon="📄")

# Streamlit ships no localization: the uploader's "Drag and drop files here" /
# "Browse files" / "Limit 200MB per file" strings are baked in English. Replace
# them with CSS (fragile across Streamlit versions, written for 1.60) and hide
# the English toolbar/menu chrome.
st.markdown("""
<style>
[data-testid="stToolbar"], #MainMenu, footer {visibility: hidden;}
/* The reopen arrow of a collapsed sidebar renders INSIDE stToolbar (1.60);
   visibility (unlike display) can be turned back on for a child. */
[data-testid="stExpandSidebarButton"] {visibility: visible;}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span {display: none;}
[data-testid="stFileUploaderDropzoneInstructions"] > div::before {
    content: "Dosyaları buraya sürükleyin"; font-size: 0.875rem;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > small {display: none;}
[data-testid="stFileUploaderDropzoneInstructions"] > div::after {
    content: "Dosya başına en fazla 200 MB"; display: block;
    font-size: 0.75rem; opacity: 0.6;
}
[data-testid="stFileUploaderDropzone"] button {font-size: 0; line-height: 0;}
[data-testid="stFileUploaderDropzone"] button span {display: none;}
[data-testid="stFileUploaderDropzone"] button::after {
    content: "Dosya seç"; font-size: 0.875rem; line-height: normal;
}
/* Source chips and score badges: "kaynak etiketi" look, monospace on top of the
   native badge's own background and rounding. */
[data-testid="stMarkdownBadge"] {
    font-family: "Source Code Pro", monospace;
    border-radius: 0.5rem;
}
/* Center the app title + subtitle (markdown/caption have no align parameter);
   .st-key-* is the official CSS hook for keyed containers. */
.st-key-app_header h3,
.st-key-app_header [data-testid="stCaptionContainer"] {text-align: center;}
/* Passage score bars in the diagnostics expander: slim, tucked under the header. */
[data-testid="stExpanderDetails"] [data-testid="stProgress"] {margin-top: -0.6rem;}
[data-testid="stExpanderDetails"] [data-testid="stProgress"] div[role="progressbar"],
[data-testid="stExpanderDetails"] [data-testid="stProgress"] div[role="progressbar"] > div {
    height: 0.3rem;
}
/* The default user avatar takes its circle from primaryColor. Recolor
   it to the navy, so red stays an accent, not the person icon (1.60 test-id;
   inline style, so !important). */
[data-testid="stChatMessageAvatarUser"] {background-color: #263685 !important;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def warm_models() -> bool:
    """Load the embedder and reranker once per server process (cold start ~15 s)."""
    from app.models import rerank_scores
    from app.retrieval import embed_query

    embed_query("ısınma sorgusu")
    rerank_scores("ısınma", ["ısınma pasajı"])
    return True


def render_answer(a: Answer, tier: str, sure: float) -> None:
    if a.reddedildi:
        # A refusal is deliberate behavior, not an error: neutral warning tone,
        # and no source chips / retrieved passages: nothing was used as a source.
        st.warning(f"**YANITLANAMADI** — {a.gerekce}", icon=":material/search_off:")
        st.caption(f"{tier} · {LLM_TIERS[tier]} · {sure:.1f} s")
        return
    st.write(a.cevap)
    if a.kaynaklar:
        st.markdown(" ".join(
            f":blue-badge[:material/description: {k['dosya']} · s.{k['sayfa']}]"
            for k in a.kaynaklar
        ))
    st.caption(f"{tier} · {LLM_TIERS[tier]} · {sure:.1f} s")
    if a.hits:
        with st.expander("Getirilen pasajlar (tanılama)", icon=":material/search:"):
            cited = {(k["dosya"], k["sayfa"]) for k in a.kaynaklar}
            for i, h in enumerate(a.hits, start=1):
                pages = f"s.{h.sayfa_baslangic}"
                if h.sayfa_bitis != h.sayfa_baslangic:
                    pages += f"-{h.sayfa_bitis}"
                badge = (" :green-badge[:material/check: kaynak]"
                         if (h.dosya, h.sayfa_baslangic) in cited else "")
                st.markdown(
                    f"**[{i}] {h.dosya} {pages}**{badge} :gray-badge[{h.skor:.3f}]"
                )
                # Score as a slim visual bar so 0.591 vs 0.520 is visible at a
                # glance (rerank scores are already in [0,1]).
                st.progress(min(max(h.skor, 0.0), 1.0))
                st.caption(h.metin[:300] + ("…" if len(h.metin) > 300 else ""))
                if len(h.metin) > 300:
                    with st.popover("Pasajın tamamı"):
                        st.markdown(f"**[{i}] {h.dosya} {pages}**")
                        st.write(h.metin)


def render_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.write(turn["soru"])
    with st.chat_message("assistant"):
        render_answer(turn["cevap"], turn["tier"], turn["sure"])


def ask(question: str, tier: str) -> tuple[Answer, float]:
    """Run pipeline.answer with live stage updates and a streaming answer box."""
    status = st.status("Başlıyor…", expanded=True)
    answer_box = st.empty()
    progress = {"label": None, "t": None}
    acc: list[str] = []

    def finish_stage() -> None:
        if progress["label"] is not None:
            status.write(f"✅ {progress['label']} — {time.perf_counter() - progress['t']:.1f} s")

    def on_stage(key: str) -> None:
        finish_stage()
        label = STAGE_LABELS.get(key, key)
        progress.update(label=label, t=time.perf_counter())
        status.update(label=f"⏳ {label}…")

    def on_token(delta: str) -> None:
        acc.append(delta)
        text = "".join(acc).lstrip()
        # Hold rendering while the text can still turn out to be the refusal
        # sentinel, a refusal is shown as the red card at the end, not streamed.
        if REFUSAL.startswith(text[:len(REFUSAL)]):
            return
        answer_box.markdown(text + " ▌")

    t0 = time.perf_counter()
    a = answer(
        question,
        model=resolve_model(tier),
        variant=RUNTIME_VARIANT,
        on_stage=on_stage,
        on_token=on_token,
    )
    sure = time.perf_counter() - t0
    finish_stage()
    status.update(label=f"Tamamlandı · {sure:.1f} s", state="complete", expanded=False)
    answer_box.empty()
    return a, sure


with st.spinner("Modeller yükleniyor (ilk açılışta ~15-20 s)…"):
    warm_models()

with st.sidebar:
    st.subheader("Model")
    tiers = list(LLM_TIERS)
    tier = st.radio(
        "Katman", tiers, index=tiers.index(LLM_DEFAULT_TIER),
        captions=[TIER_HINT[t] for t in tiers],
        label_visibility="collapsed",
    )
    st.caption(f"Aktif model: `{LLM_TIERS[tier]}`")

    st.divider()
    st.subheader("Belgeler")
    uploads = st.file_uploader(
        "PDF / PNG / JPG", type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    if st.button("İndeksle", disabled=not uploads, type="primary"):
        with st.status("İndeksleniyor…", expanded=True) as status:
            for f in uploads:
                path = store.save_upload(f.name, f.getvalue())
                try:
                    info = store.add_document(path)
                except ValueError as err:
                    st.write(f"⚠️ {err}")
                    continue
                tag = " *(güncellendi)*" if info["guncellendi"] else ""
                st.write(
                    f"✅ **{info['dosya']}**{tag}: {info['sayfa']} sayfa "
                    f"({info['ocr_sayfa']} OCR) → {info['chunk']} chunk, "
                    f"{info['sure']:.1f} s"
                )
            status.update(label="İndeksleme tamam", state="complete")

    docs = store.list_documents()
    if docs:
        st.caption("İndeksteki belgeler")
        for d in docs:
            st.markdown(
                f":material/description: **{d['dosya']}** "
                f":gray-badge[{d['chunk']} chunk]"
            )
        with st.popover("İndeksi temizle", icon=":material/delete:"):
            st.caption("Tüm yüklenen belgeler ve indeks silinecek.")
            if st.button("Evet, temizle", type="primary"):
                store.clear_index()
                st.rerun()

# Theme is fixed to dark in .streamlit/config.toml (theme.base).
with st.container(key="app_header"):
    st.markdown("### Doküman Bilgi Erişimi")
    st.caption("Cevaplar yüklediğiniz belgelere dayanır ve kaynak gösterir.")
st.divider()

if "gecmis" not in st.session_state:
    st.session_state.gecmis = []

for turn in st.session_state.gecmis:
    render_turn(turn)

ready = store.has_index()
if not ready:
    st.info(
        "Soru sormak için önce soldan belge yükleyip **İndeksle**'ye basın.",
        icon=":material/upload_file:",
    )

question = st.chat_input("Sorunuzu yazın…", disabled=not ready)
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        a, sure = ask(question, tier)
        render_answer(a, tier, sure)
    st.session_state.gecmis.append(
        {"soru": question, "cevap": a, "tier": tier, "sure": sure}
    )
