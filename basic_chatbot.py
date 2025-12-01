import os
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

nltk.download('punkt')

model = SentenceTransformer('all-MiniLM-L6-v2') 

def load_scraped_data(file_path):
    """Load and split content by URL blocks."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found!")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    pages = content.split("--- ")
    url_text_pairs = []

    for page in pages:
        if not page.strip():
            continue
        try:
            url, text = page.strip().split(" ---\n", 1)
            url_text_pairs.append((url.strip(), text.strip()))
        except ValueError:
            continue

    return url_text_pairs

def compute_embeddings(texts):
    """Convert list of texts to embeddings using sentence-transformer."""
    return model.encode(texts, convert_to_tensor=True)

def find_best_response(user_input, pages, text_embeddings):
    """Find best-matching page using semantic similarity."""
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = cosine_similarity(
        [user_embedding.cpu().numpy()],
        text_embeddings.cpu().numpy()
    )[0]
    best_match_idx = int(np.argmax(similarities))
    best_url, best_text = pages[best_match_idx]

    short_snippet = best_text[:500].strip().replace('\n', ' ') + "..."
    return short_snippet, best_url

def main():
    print("Loading scraped data...")
    data = load_scraped_data("scraped_data.txt")
    print(f"Loaded {len(data)} pages.")

    pages = [(url, text) for url, text in data]
    texts = [text for _, text in pages]

    print("🔄 Computing embeddings (this might take a few seconds)...")
    text_embeddings = compute_embeddings(texts)

    print("\n Chatbot is ready! Type your question (or 'exit' to quit):\n")
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        response, source_url = find_best_response(question, pages, text_embeddings)
        print(f"\n Answer (from {source_url}):\n{response}\n")

if __name__ == "__main__":
    main()
