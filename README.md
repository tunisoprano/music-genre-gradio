# 🎵 Music Genre Classification with CNN & Gradio

This project implements a **music genre classification system** using **deep learning**.
Audio signals are transformed into **mel-spectrograms** and classified with a
**Convolutional Neural Network (CNN)**.
A **Gradio-based web interface** is provided for interactive inference.

## 📌 Project Overview

Automatic music genre classification is a fundamental problem in audio signal
processing and music information retrieval. Such systems are commonly used in:

- Music recommendation systems
- Audio content organization
- Streaming platforms

In this project, a CNN model is trained on mel-spectrogram representations of
audio signals to classify music into predefined genres.


## 📂 Dataset (Not Included)

The model was trained using a **GTZAN-style dataset**.
The dataset is **not included in this repository** due to size constraints.

Expected directory structure:

music-genre-gradio/
└── data/
└── genres/
├── blues/
├── classical/
├── country/
├── disco/
├── hiphop/
├── jazz/
├── metal/
├── pop/
├── reggae/
└── rock/
└── *.wav

Folder names must match the `GENRES` list defined in `src/config.py`.

## ⚙️ Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


🧠 Model Training

Train the CNN model:

python -m src.train

The trained model is saved as:

artifacts/best_model.pt


📊 Model Evaluation

Evaluate the trained model:

python -m src.evaluate

This prints classification metrics such as accuracy, precision, recall, F1-score,
and the confusion matrix.

🌐 Gradio Web Interface

Launch the Gradio application for interactive inference:

python app.py

Then open your browser at:

http://127.0.0.1:7860

You can upload an audio file (.wav, .mp3, etc.) and obtain genre predictions.

🚀 HuggingFace Spaces Deployment

To deploy this project on HuggingFace Spaces:
	1.	Create a new Gradio Space
	2.	Push this repository to the Space
	3.	Ensure artifacts/best_model.pt is included in the repository

The dataset is not required for deployment, as inference is performed using
the trained model weights.



🛠 Technologies Used
	•	Python
	•	PyTorch
	•	Torchaudio
	•	SoundFile
	•	Gradio

⸻

This project was developed as part of a Deep Learning class project.
