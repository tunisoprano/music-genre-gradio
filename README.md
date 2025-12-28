# Music Genre Classification (GTZAN-style folders) + Gradio

## Dataset (NOT included)
Place your dataset like this:

```
music-genre-gradio/
└── data/
    └── genres/
        ├── blues/
        ├── classical/
        ├── ...
        └── rock/
            ├── rock.00000.wav
            └── ...
```

Folder names must match `GENRES` in `src/config.py`.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
python -m src.train
```

This saves `artifacts/best_model.pt`.

## Evaluate
```bash
python -m src.evaluate
```

## Run Gradio UI
```bash
python app.py
```

## HuggingFace Spaces
- Create a new Space (Gradio)
- Push this repository
- Make sure `artifacts/best_model.pt` exists in the repo (train locally first and commit it)
