
from src.data.pt_loader import load_books_dataset

data = load_books_dataset()

print("\nFeature shape:", data.x.shape)
print("Edge shape:", data.edge_index.shape)
print("Label shape:", data.y.shape)
