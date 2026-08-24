from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidencepilot.app import create_runtime  # noqa: E402
from evidencepilot.config import Settings  # noqa: E402

load_dotenv()
st.set_page_config(page_title="EvidencePilot", page_icon="🔎", layout="wide")
st.title("🔎 EvidencePilot")
st.caption("可验证的深度研究 Agent：规划、搜索、证据提取、缺口判断、带引用报告与引用核验")

if "runtime" not in st.session_state:
    try:
        st.session_state.runtime = create_runtime(Settings.from_env())
    except ValueError as exc:
        st.error(f"Provider 配置无效：{exc}")
        st.info("离线演示必须显式设置 LLM_PROVIDER=mock 和 SEARCH_PROVIDER=mock。")
        st.stop()
runtime = st.session_state.runtime

with st.sidebar:
    st.subheader("运行状态")
    st.write(f"模型：{runtime.llm_mode}")
    st.write(f"搜索：{runtime.search_mode}")
    if not runtime.runnable:
        st.warning("尚未配置 OPENAI_API_KEY。请复制 .env.example 为 .env 并填入兼容 API 的密钥，然后重启。测试与开发可使用 FakeLLMProvider。")
    st.subheader("历史任务")
    history = runtime.workflow.store.list_tasks() if runtime.workflow.store else []
    labels = {f"{x['question'][:35]} · {x['created_at']}": x["task_id"] for x in history}
    selected = st.selectbox("选择历史报告", [""] + list(labels))
    if selected:
        old = runtime.workflow.store.load_state(labels[selected])
        if old:
            st.session_state.result = old

question = st.text_area("研究问题", placeholder="例如：如何设计一个可靠、可验证的深度研究 Agent？", height=100)
max_rounds = st.select_slider("最大研究轮数", options=[1, 2, 3], value=2)
start = st.button("开始研究", type="primary", disabled=not runtime.runnable or not question.strip())

if start:
    status = st.status("研究进行中…", expanded=True)
    current = st.empty()
    state = runtime.workflow.initial_state(question, max_rounds)

    async def stream_run():
        latest = state
        async for event in runtime.workflow.graph.astream(state, {"recursion_limit": 30}, stream_mode="updates"):
            for node, update in event.items():
                current.info(f"当前执行节点：{node}")
                status.write(f"✓ {node}")
                latest.update(update)
        return latest

    try:
        st.session_state.result = asyncio.run(stream_run())
        status.update(label="研究完成", state="complete", expanded=False)
    except Exception as exc:
        status.update(label="研究失败", state="error")
        st.error(f"执行失败：{type(exc).__name__}: {exc}")

result = st.session_state.get("result")
if result:
    tab_report, tab_process, tab_audit = st.tabs(["研究报告", "过程与来源", "引用核验"])
    with tab_report:
        st.markdown(result.get("report") or "尚无报告")
    with tab_process:
        st.subheader("研究计划")
        st.json(result.get("research_plan", {}))
        st.subheader("已执行查询")
        st.write(result.get("completed_queries", []))
        st.subheader("来源")
        for source in result.get("sources", []):
            st.markdown(f"**[{source['source_id']}] [{source['title']}]({source['url']})**")
        if result.get("errors"):
            st.warning("\n".join(result["errors"]))
    with tab_audit:
        audit = result.get("citation_audit", {})
        st.metric("引用覆盖率", f"{audit.get('coverage', 0):.0%}")
        st.dataframe(audit.get("checks", []), use_container_width=True)
        metrics = result.get("metrics", {})
        cols = st.columns(5)
        cols[0].metric("Token", metrics.get("total_tokens", 0))
        cols[1].metric("耗时", f"{metrics.get('elapsed_seconds', 0):.2f}s")
        cols[2].metric("搜索次数", metrics.get("search_count", 0))
        cols[3].metric("抓取次数", metrics.get("fetch_count", 0))
        cols[4].metric("模型调用", metrics.get("model_calls", 0))
        with st.expander("模型与节点指标"):
            st.json({
                "model": metrics.get("model"),
                "prompt_tokens": metrics.get("prompt_tokens", 0),
                "completion_tokens": metrics.get("completion_tokens", 0),
                "reasoning_tokens": metrics.get("reasoning_tokens", 0),
                "cached_tokens": metrics.get("cached_tokens", 0),
                "retry_count": metrics.get("retry_count", 0),
                "node_metrics": metrics.get("node_metrics", {}),
                "calls": metrics.get("call_metrics", []),
            })
