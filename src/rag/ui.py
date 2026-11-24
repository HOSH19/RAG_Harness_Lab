from __future__ import annotations

import json
import os

import httpx
import streamlit as st

API_BASE = os.getenv("RAG_API_BASE_URL", "http://localhost:8080")


st.set_page_config(page_title="vLLM RAG Practice", page_icon=":books:", layout="wide")
st.title("vLLM RAG Practice")
st.caption(f"API: {API_BASE}")

with st.sidebar:
    st.header("Options")
    retrieval_only = st.toggle("Retrieval only", value=False)
    top_k = st.slider("top_k", min_value=1, max_value=10, value=5)
    if st.button("Refresh health"):
        st.session_state.pop("health", None)

    try:
        health = httpx.get(f"{API_BASE}/health", timeout=10.0).json()
        st.session_state["health"] = health
    except Exception as exc:  # noqa: BLE001
        health = {"ok": False, "error": str(exc)}
        st.session_state["health"] = health

    st.json(st.session_state.get("health", {}))

query = st.text_input("Ask AcmeOps", placeholder="What does AO-2002 mean?")
ask = st.button("Ask", type="primary")

if ask and query.strip():
    if retrieval_only:
        with st.spinner("Searching..."):
            resp = httpx.post(
                f"{API_BASE}/search",
                json={"query": query.strip(), "top_k": top_k},
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
        st.subheader("Retrieved passages")
        for item in data.get("results", []):
            with st.expander(f"{item.get('doc_id')}#{item.get('chunk_index')}  score={item.get('score')}"):
                st.write(item.get("text", ""))
    else:
        answer_box = st.empty()
        sources_box = st.container()
        collected = []
        with st.spinner("Generating..."):
            with httpx.stream(
                "POST",
                f"{API_BASE}/chat/stream",
                json={"query": query.strip(), "top_k": top_k},
                timeout=120.0,
            ) as stream:
                for line in stream.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = json.loads(line[6:])
                    if payload.get("type") == "sources":
                        with sources_box:
                            st.subheader("Sources")
                            for item in payload.get("sources", []):
                                with st.expander(
                                    f"[{item.get('doc_id')}#{item.get('chunk_index')}]"
                                ):
                                    st.write(item.get("text", ""))
                    elif payload.get("type") == "token":
                        collected.append(payload.get("text", ""))
                        answer_box.markdown("".join(collected))
                    elif payload.get("type") == "error":
                        st.error(payload.get("error"))
                        break
        if collected:
            st.subheader("Answer")
            st.markdown("".join(collected))