import requests
import gradio as gr

API = "http://127.0.0.1:8000"

# For now, subjects are hardcoded here - once we build dynamic subject
# detection, this list will instead be fetched live from the backend.
SUBJECTS = ["reinforcement_learning"]


def respond(message, history, subject):
    """
    Called every time the user sends a message in the chat.
    Sends the question to our FastAPI /ask endpoint and returns the answer.
    'history' is provided automatically by Gradio's ChatInterface but we
    don't use it yet - each question is answered independently for now.
    """
    response = requests.get(
        f"{API}/ask",
        params={"subject": subject, "question": message},
    )
    data = response.json()
    return data["answer"]


with gr.Blocks(title="Study Assistant") as demo:
    gr.Markdown("# 📚 Study Assistant")

    subject_dropdown = gr.Dropdown(
        choices=SUBJECTS,
        value=SUBJECTS[0],
        label="Subject",
    )

    gr.ChatInterface(
        fn=respond,
        additional_inputs=[subject_dropdown],
    )

if __name__ == "__main__":
    demo.launch()
