"""Streamlit interface for the AgroChat retrieval application."""
import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

st.set_page_config(page_title="AgroChat", page_icon="🌱")
st.title("AgroChat")
st.caption("Ask an agriculture-related question.")

for setting in ("EMBEDDER_PATH", "SIMILARITY_THRESHOLD"):
    try:
        value = st.secrets.get(setting)
    except StreamlitSecretNotFoundError:
        break
    if value:
        os.environ[setting] = str(value)

from app import EMBEDDER_TAG, retrieve_answer


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(message["question"])
    with st.chat_message("assistant"):
        st.markdown(message["answer"])
        st.caption(
            f"Similarity: {message['similarity']:.3f} · "
            f"{'Matched' if message['accepted'] else 'No confident match'}"
        )

with st.form("question_form", clear_on_submit=True):
    question = st.text_input("Your question", max_chars=500)
    submitted = st.form_submit_button("Ask")

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        result = retrieve_answer(question.strip())
        st.session_state.chat_history.append(
            {
                "question": question.strip(),
                "answer": result["answer"],
                "similarity": result["similarity"],
                "accepted": result["accepted"],
            }
        )
        st.rerun()

st.caption(f"Embedder: {EMBEDDER_TAG}")
