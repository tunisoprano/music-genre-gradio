
import random
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from .config import GENRES, ARTIFACTS_DIR, CHECKPOINT_PATH
from .dataset import FolderGenreDataset
from .model import GenreCNN

SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
LR = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = FolderGenreDataset()  # reads from data/genres
    idxs = list(range(len(dataset)))
    labels = [dataset.samples[i][1] for i in idxs]

    train_idx, val_idx = train_test_split(
        idxs, test_size=0.2, random_state=SEED, stratify=labels
    )

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)

    model = GenreCNN(num_classes=len(GENRES)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                pred = logits.argmax(dim=1)
                v_correct += (pred == y).sum().item()
                v_total += x.size(0)

        val_acc = v_correct / max(v_total, 1)
        print(f"Epoch {epoch:02d}/{EPOCHS} | loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model_state": model.state_dict(), "genres": GENRES}, str(CHECKPOINT_PATH))

    print("Done. Best val_acc:", best_val)
    print(f"Saved: {CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()
