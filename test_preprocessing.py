from src.data.pt_loader import load_books_dataset
from src.data.graph_preprocessing import preprocess_graph

data = load_books_dataset()

x, edge_index, y = preprocess_graph(data)

print("\nFinal tensor shapes:")
print("X:", x.shape)
print("Edge Index:", edge_index.shape)
print("Y:", y.shape)
