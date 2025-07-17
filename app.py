import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# === API Key Setup ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("⚠️ WARNING: No GOOGLE_API_KEY environment variable found.")
    print("Please set your API key using one of these methods:")
    print("1. Set environment variable: export GOOGLE_API_KEY='your_api_key_here'")
    print("2. Or temporarily hardcode it in the line below for testing")
    
    GOOGLE_API_KEY = "AIzaSyAZPplt6I4m9tHHKcyEsOzJb50TRbAgKto"
    
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        # Test the API key by making a simple request
        model = genai.GenerativeModel('gemini-1.5-flash')
        test_response = model.generate_content("Hello")
        print("✅ Google API key is valid and working")
    except Exception as e:
        print(f"❌ API key validation failed: {e}")
        GOOGLE_API_KEY = None

# === Flask App ===
app = Flask(__name__)

# === Sentence Transformer Model ===
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Sentence transformer model loaded successfully")
except Exception as e:
    print(f"❌ Error loading sentence transformer: {e}")
    embedder = None

# === Data Store ===
scraped_data_pages = []
scraped_data_texts = []
scraped_data_embeddings = None

# === Load Scraped Data ===
def load_scraped_data(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ File {file_path} not found.")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        pages = content.split("--- ")
        url_text_pairs = []

        for page in pages:
            if not page.strip():
                continue
            try:
                url, text = page.strip().split(" ---\n", 1)
                cleaned_text = " ".join(text.strip().split())
                url_text_pairs.append((url.strip(), cleaned_text))
            except ValueError:
                continue
        return url_text_pairs
    except Exception as e:
        print(f"❌ Error loading scraped data: {e}")
        return []

# === Semantic Retrieval ===
def retrieve_relevant_chunks(user_input, top_k=3, similarity_threshold=0.1):
    if scraped_data_embeddings is None or not scraped_data_embeddings.any():
        print("⚠️ No embeddings available for retrieval")
        return []

    if not embedder:
        print("❌ Sentence transformer not available")
        return []

    try:
        query_embedding = embedder.encode(user_input, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, scraped_data_embeddings)[0]
        
        # Debug: Print similarity scores
        print(f"🔍 Query: {user_input}")
        print(f"📊 Top similarity scores: {similarities.topk(min(5, len(similarities))).values.tolist()}")
        
        # Get top k results above threshold
        top_k_indices = similarities.argsort(descending=True)[:top_k]
        results = []
        
        for i in top_k_indices:
            similarity_score = similarities[i].item()
            if similarity_score >= similarity_threshold:
                results.append(scraped_data_pages[i])
                print(f"✅ Selected chunk {i} with similarity {similarity_score:.3f}")
            else:
                print(f"⚠️ Chunk {i} similarity {similarity_score:.3f} below threshold {similarity_threshold}")
        
        print(f"📋 Retrieved {len(results)} relevant chunks out of {len(scraped_data_pages)} total")
        return results
        
    except Exception as e:
        print(f"❌ Error in semantic retrieval: {e}")
        return []

# === Gemini Response ===
def generate_gemini_response(user_input, relevant_chunks):
    if not GOOGLE_API_KEY:
        return "❌ Google API key not configured. Please set your GOOGLE_API_KEY environment variable.", []

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if relevant_chunks:
            context_str = "\n\n--- Context ---\n"
            source_urls = set()
            for i, (url, text) in enumerate(relevant_chunks):
                context_str += f"Source {i+1} ({url}):\n{text[:1000]}...\n\n"  # Limit text length
                source_urls.add(url)
            context_str += "--- End Context ---"
            print(f"📝 Context length: {len(context_str)} characters")
        else:
            context_str = "No relevant content found."
            source_urls = set()
            print("⚠️ No relevant chunks found for context")

        # More flexible prompt
        prompt = f"""You are a helpful assistant. Answer the question based on the context provided below.
If you can find relevant information in the context, provide a comprehensive answer.
If the exact information isn't available but related information exists, mention what you found and indicate the limitation.
Only say "Sorry, I couldn't find that information" if there's truly no relevant information at all.

{context_str}

Question: {user_input}
Answer:"""

        response = model.generate_content(prompt)
        answer = response.text.strip()
        print(f"🤖 Generated response: {answer[:100]}...")
        return answer, list(source_urls)
        
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg:
            return "❌ Invalid API key. Please check your Google API key configuration.", []
        elif "PERMISSION_DENIED" in error_msg:
            return "❌ Permission denied. Please check your API key permissions.", []
        else:
            return f"❌ Error generating response: {error_msg}", []

# === Load Once ===
@app.before_request
def load_data_once():
    global scraped_data_pages, scraped_data_texts, scraped_data_embeddings
    if not scraped_data_pages and embedder:
        print("📄 Loading scraped data...")
        data = load_scraped_data("scraped_data.txt")
        if data:
            scraped_data_pages = data
            scraped_data_texts = [text for _, text in data]
            try:
                scraped_data_embeddings = embedder.encode(scraped_data_texts, convert_to_tensor=True)
                print(f"✅ Loaded {len(scraped_data_pages)} pages with embeddings.")
            except Exception as e:
                print(f"❌ Error creating embeddings: {e}")
                scraped_data_embeddings = np.array([])
        else:
            print("⚠️ No data loaded.")
            scraped_data_embeddings = np.array([])

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        question = request.form.get('question', '').strip()
        if not question:
            return jsonify({"answer": "Please enter a question.", "source_urls": []})

        if not GOOGLE_API_KEY:
            return jsonify({
                "answer": "❌ Google API key not configured. Please set your GOOGLE_API_KEY environment variable.",
                "source_urls": []
            })

        if not embedder:
            return jsonify({
                "answer": "❌ Sentence transformer model not available. Please check your installation.",
                "source_urls": []
            })

        relevant_chunks = retrieve_relevant_chunks(question)
        answer, urls = generate_gemini_response(question, relevant_chunks)

        return jsonify({"answer": answer, "source_urls": urls})

    except Exception as e:
        print(f"❌ Server error: {e}")
        return jsonify({"answer": f"⚠️ Internal server error: {e}", "source_urls": []})

# === Health Check Route ===
@app.route('/health')
def health_check():
    status = {
        "api_key_configured": bool(GOOGLE_API_KEY),
        "embedder_loaded": bool(embedder),
        "data_loaded": len(scraped_data_pages) > 0,
        "embeddings_created": scraped_data_embeddings is not None and scraped_data_embeddings.any()
    }
    return jsonify(status)

# === New Debug Route ===
@app.route('/debug')
def debug_info():
    return jsonify({
        "total_pages": len(scraped_data_pages),
        "sample_pages": [{"url": url, "text_preview": text[:200]} for url, text in scraped_data_pages[:3]],
        "embeddings_shape": scraped_data_embeddings.shape if scraped_data_embeddings is not None else None,
        "api_key_configured": bool(GOOGLE_API_KEY),
        "embedder_loaded": bool(embedder)
    })

# === Test Query Route ===
@app.route('/test_query', methods=['POST'])
def test_query():
    try:
        question = request.form.get('question', '').strip()
        if not question:
            return jsonify({"error": "Please enter a question."})

        # Test semantic retrieval
        relevant_chunks = retrieve_relevant_chunks(question, top_k=5, similarity_threshold=0.05)
        
        return jsonify({
            "question": question,
            "relevant_chunks_count": len(relevant_chunks),
            "relevant_chunks": [{"url": url, "text_preview": text[:300]} for url, text in relevant_chunks],
            "total_pages_available": len(scraped_data_pages)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})

# === Run App ===
if __name__ == '__main__':
    print("\n=== RAG Flask App Status ===")
    print(f"API Key Configured: {'✅' if GOOGLE_API_KEY else '❌'}")
    print(f"Embedder Loaded: {'✅' if embedder else '❌'}")
    print("================================\n")
    
    if not GOOGLE_API_KEY:
        print("⚠️ WARNING: App will run but won't be able to generate responses without API key")
    
    app.run(debug=True)