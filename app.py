import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from dotenv import load_dotenv
import threading
import time

# Import scraper functionality
from deep_scraper import crawl_website

load_dotenv()

# === API Key Setup ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Google API key loaded from .env file")
    print(f"🔑 API Key (first 10 chars): {GOOGLE_API_KEY[:10]}...")
    
    # Test the API key to get project info
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        # Make a minimal test request
        response = model.generate_content("Hi")
        print("✅ API key test successful - should be Tier 1")
    except Exception as e:
        print(f"⚠️ API key test failed: {e}")
        if "quota" in str(e).lower():
            print("🔍 This suggests the key is still in free tier mode")
else:
    print("❌ GOOGLE_API_KEY not found in .env file")

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
scraping_in_progress = False
scraping_complete = False

# === Scraping Function ===
def run_scraper():
    global scraped_data_pages, scraped_data_texts, scraped_data_embeddings, scraping_in_progress, scraping_complete
    
    scraping_in_progress = True
    print("🕷️ Starting web scraping...")
    
    try:
        # Run the scraper
        website = "https://inciem.com"
        print(f"🎯 Scraping website: {website}")
        
        general_scraped_data, structured_scraped_data = crawl_website(website, max_pages=50)
        
        # Save general scraped data
        output_general_file = "scraped_data.txt"
        with open(output_general_file, "w", encoding="utf-8") as f:
            for url, content in general_scraped_data.items():
                f.write(f"--- {url} ---\n")
                f.write(content + "\n\n")
        print(f"💾 General scraped data saved to {output_general_file}")
        
        # Save structured data
        output_structured_file = "structured_data.txt"
        with open(output_structured_file, "w", encoding="utf-8") as f:
            f.write("--- Contact Information ---\n")
            for entry in structured_scraped_data['contact_info']:
                f.write(f"URL: {entry['url']}\n")
                for key, value in entry['data'].items():
                    f.write(f"  {key.replace('_', ' ').title()}: {', '.join(value)}\n")
                f.write("\n")

            f.write("\n--- Job Opportunities ---\n")
            for entry in structured_scraped_data['job_opportunities']:
                f.write(f"URL: {entry['url']}\n")
                f.write("  Jobs:\n")
                for job in entry['data']:
                    f.write(f"    - {job}\n")
                f.write("\n")
        print(f"💾 Structured data saved to {output_structured_file}")
        
        # Load the newly scraped data
        load_scraped_data("scraped_data.txt")
        
        scraping_complete = True
        print("✅ Web scraping completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        scraping_complete = False
    finally:
        scraping_in_progress = False

# === Load Scraped Data ===
def load_scraped_data(file_path):
    global scraped_data_pages, scraped_data_texts, scraped_data_embeddings
    
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
        
        # Update global variables
        scraped_data_pages = url_text_pairs
        scraped_data_texts = [text for _, text in url_text_pairs]
        
        # Create embeddings if embedder is available
        if embedder and scraped_data_texts:
            try:
                scraped_data_embeddings = embedder.encode(scraped_data_texts, convert_to_tensor=True)
                print(f"✅ Created embeddings for {len(scraped_data_pages)} pages.")
            except Exception as e:
                print(f"❌ Error creating embeddings: {e}")
                scraped_data_embeddings = np.array([])
        
        return url_text_pairs
    except Exception as e:
        print(f"❌ Error loading scraped data: {e}")
        return []

# === Basic Chatbot Functionality ===
def basic_chatbot_response(user_input):
    """Basic chatbot response using semantic similarity (from basic_chatbot.py)"""
    if not scraped_data_pages or not embedder:
        return "No data available for basic chatbot.", []
    
    try:
        # Find best matching page using semantic similarity
        user_embedding = embedder.encode(user_input, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(user_embedding, scraped_data_embeddings)[0]
        
        best_match_idx = int(similarities.argmax())
        best_url, best_text = scraped_data_pages[best_match_idx]
        
        # Return a snippet of the best matching text
        short_snippet = best_text[:500].strip().replace('\n', ' ') + "..."
        return short_snippet, [best_url]
        
    except Exception as e:
        print(f"❌ Error in basic chatbot: {e}")
        return f"Error processing request: {e}", []

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

        # More natural prompt
        prompt = f"""You are a helpful and knowledgeable assistant. Answer the question naturally and conversationally using the information below.

When answering:
- Be direct and conversational
- Don't say "based on the context" or similar phrases
- If you find relevant information, share it naturally
- If the exact information isn't available, mention what you found and be honest about limitations
- Only say "I don't have that information" if there's truly nothing relevant

Information available:
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
        elif "429" in error_msg and "quota" in error_msg.lower():
            return "❌ API quota exceeded. You've hit the free tier limit (50 requests/day). Please upgrade your plan or wait until tomorrow.", []
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
        chatbot_type = request.form.get('chatbot_type', 'gemini')  # 'gemini' or 'basic'
        
        if not question:
            return jsonify({"answer": "Please enter a question.", "source_urls": []})

        if not embedder:
            return jsonify({
                "answer": "❌ Sentence transformer model not available. Please check your installation.",
                "source_urls": []
            })

        if chatbot_type == 'basic':
            # Use basic chatbot functionality
            answer, urls = basic_chatbot_response(question)
        else:
            # Use Gemini chatbot
            relevant_chunks = retrieve_relevant_chunks(question)
            answer, urls = generate_gemini_response(question, relevant_chunks)

        return jsonify({"answer": answer, "source_urls": urls})

    except Exception as e:
        print(f"❌ Server error: {e}")
        return jsonify({"answer": f"⚠️ Internal server error: {e}", "source_urls": []})

# === Scraping Status Route ===
@app.route('/scraping_status')
def get_scraping_status():
    return jsonify({
        "scraping_in_progress": scraping_in_progress,
        "scraping_complete": scraping_complete,
        "data_loaded": len(scraped_data_pages) > 0,
        "total_pages": len(scraped_data_pages)
    })

# === Manual Scraping Route ===
@app.route('/start_scraping', methods=['POST'])
def start_scraping():
    if scraping_in_progress:
        return jsonify({"status": "already_running", "message": "Scraping is already in progress"})
    
    # Start scraping in a separate thread
    thread = threading.Thread(target=run_scraper)
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started", "message": "Scraping started successfully"})

# === Health Check Route ===
@app.route('/health')
def health_check():
    status = {
        "api_key_configured": bool(GOOGLE_API_KEY),
        "api_key_preview": GOOGLE_API_KEY[:10] + "..." if GOOGLE_API_KEY else None,
        "embedder_loaded": bool(embedder),
        "data_loaded": len(scraped_data_pages) > 0,
        "embeddings_created": scraped_data_embeddings is not None and scraped_data_embeddings.any(),
        "scraping_status": {
            "in_progress": scraping_in_progress,
            "complete": scraping_complete
        }
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
        "embedder_loaded": bool(embedder),
        "scraping_status": {
            "in_progress": scraping_in_progress,
            "complete": scraping_complete
        }
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

# === API Key Test Route ===
@app.route('/test_api')
def test_api_key():
    if not GOOGLE_API_KEY:
        return jsonify({"error": "No API key configured"})
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello, this is a test.")
        return jsonify({
            "status": "success",
            "api_key_preview": GOOGLE_API_KEY[:10] + "...",
            "response": response.text,
            "message": "API key is working - should be Tier 1"
        })
    except Exception as e:
        error_msg = str(e)
        return jsonify({
            "status": "error",
            "api_key_preview": GOOGLE_API_KEY[:10] + "...",
            "error": error_msg,
            "is_quota_error": "quota" in error_msg.lower() or "429" in error_msg
        })

# === Project Info Route ===
@app.route('/project_info')
def get_project_info():
    if not GOOGLE_API_KEY:
        return jsonify({"error": "No API key configured"})
    
    try:
        # Try to get project info from the API key
        import google.auth
        from google.auth.transport.requests import Request
        
        # This might give us project info
        return jsonify({
            "api_key_preview": GOOGLE_API_KEY[:10] + "...",
            "note": "API key is working but showing free tier quota. This suggests the key is linked to a free tier project, not your Tier 1 project.",
            "action_required": "Check that you're using the API key from the correct Tier 1 project in Google AI Studio"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "api_key_preview": GOOGLE_API_KEY[:10] + "..."
        })

# === Run App ===
if __name__ == '__main__':
    print("\n=== Integrated Web Chatbot System ===")
    print(f"API Key Configured: {'✅' if GOOGLE_API_KEY else '❌'}")
    print(f"Embedder Loaded: {'✅' if embedder else '❌'}")
    print("================================\n")
    
    if not GOOGLE_API_KEY:
        print("⚠️ WARNING: App will run but won't be able to generate Gemini responses without API key")
        print("💡 Basic chatbot will still work with scraped data")
    else:
        print("💡 Note: Free tier has 50 requests/day limit. Upgrade plan for more requests.")
    
    # Start scraping automatically in background
    print("🚀 Starting automatic web scraping...")
    scraping_thread = threading.Thread(target=run_scraper)
    scraping_thread.daemon = True
    scraping_thread.start()
    
    print("🌐 Starting Flask web server...")
    app.run(debug=True)