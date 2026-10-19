"""Streamlit interface for the AgroChat retrieval application."""
import html
import os
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from jinja2 import Environment, FileSystemLoader, select_autoescape

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="AgroChat | Agriculture Q&A", page_icon="🌱")

try:
    custom_css = (BASE_DIR / "static" / "style.css").read_text(encoding="utf-8")
except OSError as exc:
    st.error(f"Could not load the AgroChat stylesheet: {exc}")
    st.stop()

st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
template_environment = Environment(
    loader=FileSystemLoader(BASE_DIR / "templates"),
    autoescape=select_autoescape(("html", "xml")),
)
branding_html = template_environment.get_template("_branding.html").render()

for setting in ("EMBEDDER_PATH", "SIMILARITY_THRESHOLD"):
    try:
        value = st.secrets.get(setting)
    except StreamlitSecretNotFoundError:
        break
    if value:
        os.environ[setting] = str(value)

from app import EMBEDDER_TAG, retrieve_answer

st.markdown(branding_html, unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.container(border=True):
    st.markdown(
        """
        <div class="chat-heading">
            <div class="avatar">🌿</div>
            <div>
                <h2>Chat with AgroChat</h2>
                <p>Answers from an agriculture knowledge base</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    messages_html = []
    if not st.session_state.chat_history:
        messages_html.append(
            """
            <div class="message bot"><div class="bubble">
                Hello! I'm AgroChat. Ask me an agriculture question, and
                I'll find the closest answer in my knowledge base.
            </div></div>
            """
        )

    for message in st.session_state.chat_history:
        question = html.escape(message["question"])
        answer = html.escape(message["answer"])
        if message["accepted"]:
            meta = (
                f"Topic: {html.escape(str(message['topic'] or 'agriculture'))}"
                f" · Similarity: {message['similarity']:.4f}"
            )
        else:
            meta = "No sufficiently relevant answer found."
        messages_html.append(
            f'<div class="message user"><div class="bubble">{question}</div></div>'
            f'<div class="message bot"><div class="bubble">{answer}'
            f'<span class="meta">{meta}</span></div></div>'
        )

    st.markdown(
        f'<div class="messages" aria-live="polite">{"".join(messages_html)}</div>',
        unsafe_allow_html=True,
    )

    suggestion_columns = st.columns(3)
    suggestions = (
        ("Soil fertility", "How can I improve soil fertility?"),
        ("Crop health", "Why are tomato leaves turning yellow?"),
        ("Water management", "How can I conserve water on a farm?"),
    )
    selected_question = None
    for column, (label, question) in zip(suggestion_columns, suggestions):
        with column:
            if st.button(label, key=f"suggestion_{label}", use_container_width=True):
                selected_question = question

    with st.form("chat_form", clear_on_submit=True):
        input_column, submit_column = st.columns([5, 1], vertical_alignment="center")
        with input_column:
            question_input = st.text_input(
                "Your agriculture question",
                max_chars=500,
                placeholder="Ask your farming question...",
                label_visibility="collapsed",
            )
        with submit_column:
            submitted = st.form_submit_button("Send ↑", use_container_width=True)

    st.caption(
        "Educational information only. Confirm important crop treatments "
        "with a qualified local agricultural adviser."
    )

if selected_question or submitted:
    question = (selected_question or question_input).strip()
    if not question:
        st.warning("Please enter a question.")
    elif len(question) > 500:
        st.warning("Please keep questions under 500 characters.")
    else:
        with st.spinner("Searching the agriculture knowledge base..."):
            result = retrieve_answer(question)
        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": result["answer"],
                "topic": result["topic"],
                "similarity": result["similarity"],
                "accepted": result["accepted"],
            }
        )
        st.rerun()

st.markdown(
    f'<footer>AgroChat · Agriculture Q&amp;A · {html.escape(EMBEDDER_TAG)}</footer>',
    unsafe_allow_html=True,
)
