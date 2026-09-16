import json
import requests
import gradio as gr

API = "http://127.0.0.1:8000"


def get_subjects():
    response = requests.get(f"{API}/subjects")
    subjects = response.json()["subjects"]
    return subjects if subjects else ["reinforcement_learning"]


def respond(message, history, subject):
    response = requests.get(
        f"{API}/ask",
        params={"subject": subject, "question": message},
    )
    return response.json()["answer"]


def handle_upload(file):
    if file is None:
        return "No file selected.", "", ""

    with open(file, "rb") as f:
        response = requests.post(f"{API}/detect-subject", files={"file": f})
    data = response.json()
    guess = data["guessed_subject"]
    temp_path = data["temp_path"]

    message = f"Detected subject: **{guess}**\n\nEdit below if wrong, then click Confirm & Add."
    return message, guess, temp_path


def handle_confirm(subject, temp_path):
    if not temp_path:
        yield "Upload a file first."
        return

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
                yield update["message"]
            elif update["status"] == "progress":
                yield f"Ingesting... {update['current']}/{update['total']} chunks ({update['message']})"
            elif update["status"] == "skipped":
                yield f"already ingested: {update['message']}"
            elif update["status"] == "done":
                yield f"done: {update['message']} Go to the Chat tab and click Refresh Subjects."


with gr.Blocks(title="Study Assistant") as demo:
    gr.Markdown("# Study Assistant")

    with gr.Tab("Chat"):
        subject_dropdown = gr.Dropdown(
            choices=get_subjects(),
            value=get_subjects()[0],
            label="Subject",
        )
        refresh_button = gr.Button("Refresh Subjects")
        refresh_button.click(fn=get_subjects, outputs=[subject_dropdown])

        gr.ChatInterface(fn=respond, additional_inputs=[subject_dropdown])

    with gr.Tab("Upload Notes"):
        gr.Markdown("Upload a PDF, PPTX, or DOCX. We will guess the subject - confirm or correct it before it is added.")

        file_input = gr.File(label="Upload file", file_types=[".pdf", ".pptx", ".docx"])
        detect_button = gr.Button("Detect Subject")
        detect_output = gr.Markdown()

        confirm_subject = gr.Textbox(label="Subject (edit if wrong)", visible=True)
        temp_path_state = gr.State("")
        confirm_button = gr.Button("Confirm and Add")
        confirm_output = gr.Markdown()

        detect_button.click(
            fn=handle_upload,
            inputs=[file_input],
            outputs=[detect_output, confirm_subject, temp_path_state],
        )
        confirm_button.click(
            fn=handle_confirm,
            inputs=[confirm_subject, temp_path_state],
            outputs=[confirm_output],
        )

if __name__ == "__main__":
    demo.launch()
