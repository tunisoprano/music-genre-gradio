import gradio as gr
from src.inference import load_checkpoint, predict_file

MODEL, GENRES = load_checkpoint()

def predict(audio_path):
    if audio_path is None:
        return "No audio provided.", {}
    probs, top = predict_file(MODEL, GENRES, audio_path)
    top_text = "Top-3: " + ", ".join([f"{g} ({p*100:.2f}%)" for g, p in top])
    top1 = top[0][0] if top else "?"
    return f"Top-1: {top1}\n{top_text}", probs

demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload a WAV file"),
    outputs=[gr.Textbox(label="Prediction"), gr.Label(label="Probabilities")],
    title="Music Genre Classification",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
