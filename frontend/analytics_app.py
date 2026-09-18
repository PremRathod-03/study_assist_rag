import json
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="Usage Analytics", layout="wide")
st.title("Study Assistant - Usage Analytics")

# One consistent color palette used across every chart on this page
PALETTE = ["#6366F1", "#22C55E", "#F59E0B", "#EC4899", "#14B8A6", "#8B5CF6", "#EF4444"]


def load_usage_log():
    records = []
    try:
        with open("data/usage_log.jsonl") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    except FileNotFoundError:
        pass
    return pd.DataFrame(records)


df = load_usage_log()

if df.empty:
    st.info("No usage data yet. Ask a few questions or generate some content first.")
else:
    total_calls = len(df)
    total_tokens = int(df["total_tokens"].sum())
    total_prompt = int(df["prompt_tokens"].sum())
    total_completion = int(df["completion_tokens"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total LLM Calls", total_calls)
    col2.metric("Total Tokens", f"{total_tokens:,}")
    col3.metric("Prompt Tokens", f"{total_prompt:,}")
    col4.metric("Completion Tokens", f"{total_completion:,}")

    st.subheader("Tokens by Subject")
    by_subject = df.groupby("subject", as_index=False)["total_tokens"].sum()
    chart1 = (
        alt.Chart(by_subject)
        .mark_bar()
        .encode(
            x=alt.X("subject:N", title="Subject", sort="-y", axis=alt.Axis(labelLimit=200, labelAngle=-30)),
            y=alt.Y("total_tokens:Q", title="Total Tokens"),
            color=alt.Color("subject:N", scale=alt.Scale(range=PALETTE), legend=None),
            tooltip=["subject", "total_tokens"],
        )
        .properties(height=350)
    )
    st.altair_chart(chart1, use_container_width=True)

    st.subheader("Tokens by Operation Type")
    by_operation = df.groupby("operation", as_index=False)["total_tokens"].sum()
    chart2 = (
        alt.Chart(by_operation)
        .mark_bar()
        .encode(
            x=alt.X("operation:N", title="Operation", sort="-y", axis=alt.Axis(labelLimit=200, labelAngle=-30)),
            y=alt.Y("total_tokens:Q", title="Total Tokens"),
            color=alt.Color("operation:N", scale=alt.Scale(range=PALETTE), legend=None),
            tooltip=["operation", "total_tokens"],
        )
        .properties(height=350)
    )
    st.altair_chart(chart2, use_container_width=True)

    st.subheader("Usage Over Time")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    chart3 = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(color=PALETTE[0], size=60), color=PALETTE[0])
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("total_tokens:Q", title="Tokens per Call"),
            tooltip=["timestamp", "subject", "operation", "total_tokens"],
        )
        .properties(height=350)
    )
    st.altair_chart(chart3, use_container_width=True)

    st.subheader("Raw Log")
    st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
