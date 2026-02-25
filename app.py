import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from RAG import answer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClearPath HR Assistance",
    page_icon="🧭",
    layout="centered",
)

# ── Minimal custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { max-width: 780px; padding-top: 2rem; }
    .badge-ok   { background:#d1fae5; color:#065f46; padding:2px 10px;
                  border-radius:99px; font-size:0.78rem; font-weight:600; }
    .badge-warn { background:#fef3c7; color:#92400e; padding:2px 10px;
                  border-radius:99px; font-size:0.78rem; font-weight:600; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helper: debug panel ────────────────────────────────────────────────────────
def render_debug_panel(meta: dict):
    """Collapsible panel showing model, tokens, latency, evaluator result."""
    flagged      = meta.get("flagged", False)
    flag_reasons = meta.get("flag_reasons", [])

    badge = (
        f'<span class="badge-warn">⚠ {", ".join(flag_reasons)}</span>'
        if flagged else
        '<span class="badge-ok">✓ Passed</span>'
    )

    with st.expander("🔍 Query details", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Model used**")
            st.code(meta.get("model", "—"), language=None)
            st.markdown("**Routing reason**")
            st.caption(meta.get("routing_reason", "—"))

        with col2:
            st.markdown("**Tokens**")
            tin  = meta.get("tokens_input", 0)
            tout = meta.get("tokens_output", 0)
            st.markdown(f"`{tin}` prompt &nbsp;·&nbsp; `{tout}` completion")
            st.markdown("**Latency**")
            st.caption(f"{meta.get('latency_ms', 0):.0f} ms")

        st.markdown("**Evaluator**")
        st.markdown(badge, unsafe_allow_html=True)
        if flag_reasons:
            st.caption(f"Reasons: {', '.join(flag_reasons)}")


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 🧭 ClearPath HR Assistance")
st.caption("Ask me anything about HR policies, benefits, or company guidelines.")
st.divider()

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # each: {"role", "content", "meta"?}

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            render_debug_panel(msg["meta"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about HR policies…"):

    # Show user bubble right away
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call RAG pipeline, render assistant bubble
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = answer(prompt)
                answer_text = result["answer"]
            except Exception as e:
                answer_text = f"⚠️ Error: {e}"
                result = {}

        st.write(answer_text)

        if result:
            render_debug_panel(result)

    # Persist to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer_text,
        "meta":    result,
    })