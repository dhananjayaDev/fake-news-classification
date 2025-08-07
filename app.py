# app.py
import streamlit as st
import torch
import torch.nn.functional as F
import os

# Import your model class
from model.gnn_model import GNN

# Stub: Convert input text to graph-compatible heterogeneous tensors
def preprocess_text_to_graph(text):
    # Define node counts
    num_articles = 5
    num_subjects = 5
    num_words = 10

    # Dummy node features matching training dimensions
    x_dict = {
        "article": torch.randn(num_articles, 256),
        "subject": torch.randn(num_subjects, 8),
        "word": torch.randn(num_words, 4999),
    }

    # Edge indices must reference valid node indices
    edge_index_dict = {
        ("article", "has_subject", "subject"): torch.randint(0, min(num_articles, num_subjects), (2, 8)),
        ("article", "contains", "word"): torch.randint(0, min(num_articles, num_words), (2, 12)),
        ("article", "similar", "article"): torch.randint(0, num_articles, (2, 6)),
    }

    return x_dict, edge_index_dict

# Load model from assets folder
@st.cache_resource
def load_model():
    model_path = os.path.join("assets", "gnn_model.pth")
    if not os.path.exists(model_path):
        st.error("Model file not found in 'assets/'. Please check the path.")
        st.stop()

    # Metadata used during training
    metadata = (
        ['article', 'subject', 'word'],  # node types
        [
            ('article', 'has_subject', 'subject'),
            ('article', 'contains', 'word'),
            ('article', 'similar', 'article')
        ]  # edge types
    )

    # Dummy data to infer input dimensions
    class DummyData:
        def __getitem__(self, key):
            dims = {
                "article": 256,
                "subject": 8,
                "word": 4999
            }
            return type('Node', (), {'x': torch.randn(10, dims[key])})()

    dummy_data = DummyData()

    model = GNN(metadata, hidden_channels=128, out_channels=2, data=dummy_data)

    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
    return model

# Streamlit UI
st.set_page_config(page_title="Fake News Classifier", layout="centered")
st.title("📰 Fake News Classifier")
st.write("Enter a news headline or article snippet to check if it's likely fake.")

user_input = st.text_area("Input Text", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            x_dict, edge_index_dict = preprocess_text_to_graph(user_input)
            model = load_model()
            output_dict = model(x_dict, edge_index_dict)

            # Ensure 'article' output exists and has valid shape
            if 'article' not in output_dict or output_dict['article'].ndim != 2:
                st.error("Model output is invalid or missing 'article' predictions.")
                st.stop()

            output = output_dict['article']  # shape: [num_articles, 2]
            probs = F.softmax(output, dim=1)  # shape: [num_articles, 2]

            # Use first article node for prediction
            pred = torch.argmax(probs[0]).item()
            label = "🟢 Real News" if pred == 0 else "🔴 Fake News"
            confidence = probs[0][pred].item()

            st.success(f"Prediction: {label}")
            st.write(f"Confidence: {confidence:.2f}")