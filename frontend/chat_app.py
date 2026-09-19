import json
import requests
import streamlit as st

API = "http://127.0.0.1:8000"
ANALYTICS_PORT = 8501
MAX_FILES = 5
NEW_SUBJECT_OPTION = "+ Create new subject..."

st.set_page_config(page_title="Study Assistant", layout="wide")


def get_subjects():
    response = requests.get(f"{API}/subjects")
    return response.json()["subjects"]


def ask_question(subject, question):
    response = requests.get(f"{API}/ask", params={"subject": subject, "question": question})
    return response.json()["answer"]


def detect_subject_for_file(uploaded_file):
    response = requests.post(f"{API}/detect-subject", files={"file": uploaded_file})
    return response.json()


def confirm_and_ingest_streaming(temp_path, subject, status_box):
    with requests.post(
        f"{API}/confirm-ingest",
        data={"temp_path": temp_path, "subject": subject},
        stream=True,
    ) as response:
        for line in response.iter_lines():
            if not line:
                continue
            update = json.loads(line)
            if update["status"] == "loading":
                status_box.update(label=update["message"])
            elif update["status"] == "progress":
                status_box.write(f"Enriching chunk {update['current']}/{update['total']}")
            elif update["status"] in ("skipped", "done"):
                status_box.update(label=update["message"], state="complete")
                return update["message"]


with st.sidebar:
    st.markdown("### Subjects")
    subjects = get_subjects()
    if "active_subject" not in st.session_state:
        st.session_state.active_subject = subjects[0] if subjects else None

    for subject in subjects:
        if st.button(subject, use_container_width=True, key=f"subj_{subject}"):
            st.session_state.active_subject = subject

    if st.button("Refresh Subjects", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.caption("**To add a new subject:** attach a file (or up to 5) using the + icon below. New subjects are detected automatically.")

active = st.session_state.active_subject
st.markdown(f"## {active or 'No subject selected'}")

btn_col1, btn_col2, btn_col3 = st.columns(3)
with btn_col1:
    gen_questions_clicked = st.button("Questions", use_container_width=True)
with btn_col2:
    gen_paper_clicked = st.button("Paper", use_container_width=True)
with btn_col3:
    analytics_link = f"http://localhost:{ANALYTICS_PORT}/?subject={active}" if active else f"http://localhost:{ANALYTICS_PORT}"
    st.link_button("Analytics", analytics_link, use_container_width=True)

if gen_questions_clicked:
    st.session_state.show_question_form = True
if st.session_state.get("show_question_form"):
    with st.expander("Generate Practice Questions", expanded=True):
        topic = st.text_input("Topic", key="qgen_topic")
        marks = st.number_input("Marks per question", value=5, key="qgen_marks")
        count = st.number_input("Number of questions", value=3, key="qgen_count")
        if st.button("Generate", key="qgen_submit"):
            with st.spinner("Generating..."):
                response = requests.get(
                    f"{API}/generate-questions",
                    params={"subject": active, "topic": topic, "marks": marks, "count": count},
                )
                st.markdown(response.json()["questions"])

if gen_paper_clicked:
    st.session_state.show_paper_form = True
if st.session_state.get("show_paper_form"):
    with st.expander("Generate Sample Paper", expanded=True):
        exam_type = st.selectbox("Exam type", ["internals", "end_sem"], key="paper_exam_type")
        if st.button("Generate", key="paper_submit"):
            with st.spinner("Generating full paper (this can take a minute)..."):
                response = requests.get(
                    f"{API}/generate-paper",
                    params={"subject": active, "exam_type": exam_type},
                )
                st.markdown(response.json()["paper"])

st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if "pending_uploads" not in st.session_state:
    st.session_state.pending_uploads = []

for i, pending in enumerate(list(st.session_state.pending_uploads)):
    with st.chat_message("assistant"):
        st.markdown(f"**{pending['filename']}** - detected subject: **{pending['guessed_subject']}**")

        options = subjects + [NEW_SUBJECT_OPTION]
        default_index = options.index(pending["guessed_subject"]) if pending["guessed_subject"] in subjects else len(subjects)

        chosen = st.selectbox(
            "Confirm the subject:",
            options,
            index=default_index,
            key=f"confirm_subject_select_{i}",
        )

        if chosen == NEW_SUBJECT_OPTION:
            final_subject = st.text_input(
                "New subject name:",
                value=pending["guessed_subject"],
                key=f"confirm_subject_input_{i}",
            )
        else:
            final_subject = chosen

        if st.button("Confirm & Add to Knowledge Base", key=f"confirm_ingest_btn_{i}"):
            with st.status(f"Adding {pending['filename']} to {final_subject}...", expanded=True) as status_box:
                result = confirm_and_ingest_streaming(pending["temp_path"], final_subject, status_box)
            st.session_state.messages.append({"role": "assistant", "content": result})
            st.session_state.pending_uploads.pop(i)
            st.rerun()

prompt = st.chat_input(
    "Ask a question, or attach up to 5 files to add/grow a subject...",
    accept_file="multiple",
    file_type=["pdf", "pptx", "docx"],
)

if prompt:
    if prompt.files:
        files_to_process = prompt.files[:MAX_FILES]
        if len(prompt.files) > MAX_FILES:
            st.warning(f"Only the first {MAX_FILES} files were processed (limit reached).")

        for uploaded_file in files_to_process:
            st.session_state.messages.append({"role": "user", "content": f"Uploaded: {uploaded_file.name}"})
            detection = detect_subject_for_file(uploaded_file)
            st.session_state.pending_uploads.append({
                "filename": uploaded_file.name,
                "guessed_subject": detection["guessed_subject"],
                "temp_path": detection["temp_path"],
            })
        st.rerun()

    elif prompt.text:
        st.session_state.messages.append({"role": "user", "content": prompt.text})
        with st.chat_message("user"):
            st.markdown(prompt.text)

        with st.chat_message("assistant"):
            if not active:
                answer = "Pick a subject from the sidebar first."
            else:
                with st.spinner("Thinking..."):
                    answer = ask_question(active, prompt.text)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})