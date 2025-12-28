import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from .config import GENRES, CHECKPOINT_PATH
from .dataset import FolderGenreDataset
from .model import GenreCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu")
    genres = ckpt.get("genres", GENRES)

    model = GenreCNN(num_classes=len(genres)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = FolderGenreDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(y.tolist())

    print(classification_report(y_true, y_pred, target_names=genres, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
