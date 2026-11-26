
from src.data.pt_loader import load_books_dataset
from src.data.graph_preprocessing import preprocess_graph
from src.data.structural_features import extract_structural_features

data = load_books_dataset()

x, edge_index, y = preprocess_graph(data)

structural_features = extract_structural_features(
    edge_index,
    num_nodes=x.shape[0]
)

print("\nFirst 5 nodes:")
print(structural_features[:5])
