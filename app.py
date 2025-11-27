import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from dotenv import load_dotenv
import threading
import time

# Try to import torch, fallback to numpy if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, using numpy fallback for tensor operations")

# Import scraper functionality
from deep_scraper import crawl_website

load_dotenv()

# === API Key Setup ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = None  # Will be set to a working model name

def find_working_model():
    """Find a working Gemini model by listing available models"""
    global GEMINI_MODEL_NAME
    
    if not GOOGLE_API_KEY:
        return None
    
    try:
        # List available models
        models = genai.list_models()
        available_models = []
        
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                # Extract model name (e.g., "models/gemini-pro" -> "gemini-pro")
                model_name = m.name.split('/')[-1] if '/' in m.name else m.name
                available_models.append(model_name)
        
        print(f"Found {len(available_models)} available models")
        if available_models:
            print(f"Available models: {', '.join(available_models[:10])}")
        
        # Try models in order of preference
        preferred_models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro', 'gemini-2.0-flash-exp']
        
        for model_name in preferred_models:
            if model_name in available_models:
                try:
                    # Test if this model actually works
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content("Hi")
                    GEMINI_MODEL_NAME = model_name
                    print(f"Using model: {model_name}")
                    return model_name
                except Exception as e:
                    print(f"Model {model_name} failed: {str(e)[:100]}")
                    continue
        
        # If preferred models don't work, try the first available one
        if available_models:
            for model_name in available_models:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content("Hi")
                    GEMINI_MODEL_NAME = model_name
                    print(f"Using first available model: {model_name}")
                    return model_name
                except Exception as e:
                    continue
        
        print("No working models found")
        return None
        
    except Exception as e:
        print(f"Error finding working model: {e}")
        return None

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Google API key loaded from .env file")
    print(f"🔑 API Key (first 10 chars): {GOOGLE_API_KEY[:10]}...")
    
    # Find a working model
    find_working_model()
    
    if GEMINI_MODEL_NAME:
        print(f"API key configured - using model: {GEMINI_MODEL_NAME}")
    else:
        print("Could not find a working model. Gemini features may not work.")
else:
    print("GOOGLE_API_KEY not found in .env file")

# === Flask App ===
app = Flask(__name__)

# === Sentence Transformer Model ===
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Sentence transformer model loaded successfully")
except Exception as e:
    print(f"Error loading sentence transformer: {e}")
    embedder = None

# === Data Store ===
scraped_data_pages = []
scraped_data_texts = []
scraped_data_embeddings = None
scraping_in_progress = False
scraping_complete = False
# target_website = "https://inciem.com/"  # Default website

# === Custom Data Store ===
custom_data_texts = []
custom_data_embeddings = None
custom_data_sources = []  # Store metadata about custom data entries

# === Scraping Function ===
def run_scraper():
    global scraped_data_pages, scraped_data_texts, scraped_data_embeddings, scraping_in_progress, scraping_complete
    
    scraping_in_progress = True
    print("Starting web scraping...")
    
    try:
        # Run the scraper using the target website
        website = target_website
        print(f"Scraping website: {website}")
        
        general_scraped_data, structured_scraped_data = crawl_website(website, max_pages=50)
        
        # Save general scraped data
        output_general_file = "scraped_data.txt"
        with open(output_general_file, "w", encoding="utf-8") as f:
            for url, content in general_scraped_data.items():
                f.write(f"--- {url} ---\n")
                f.write(content + "\n\n")
        print(f"General scraped data saved to {output_general_file}")
        
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
        print(f"Structured data saved to {output_structured_file}")
        
        # Load the newly scraped data
        load_scraped_data("scraped_data.txt")
        
        scraping_complete = True
        print("Web scraping completed successfully!")
        
    except Exception as e:
        print(f"Error during scraping: {e}")
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
                print(f"Created embeddings for {len(scraped_data_pages)} pages.")
            except Exception as e:
                print(f"Error creating embeddings: {e}")
                scraped_data_embeddings = np.array([])
        
        return url_text_pairs
    except Exception as e:
        print(f"Error loading scraped data: {e}")
        return []

# === Load Custom Data ===
def load_custom_data():
    """Load custom data from file if it exists"""
    global custom_data_texts, custom_data_embeddings, custom_data_sources
    
    custom_data_file = "custom_data.txt"
    if not os.path.exists(custom_data_file):
        return []
    
    try:
        with open(custom_data_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        if not content.strip():
            return []
        
        entries = content.split("--- CUSTOM ENTRY ---")
        custom_data_texts = []
        custom_data_sources = []
        
        for entry in entries:
            if not entry.strip():
                continue
            
            lines = entry.strip().split('\n')
            if len(lines) >= 3:
                title = lines[0].strip()
                category = lines[1].strip()
                text = '\n'.join(lines[2:]).strip()
                
                if text:
                    custom_data_texts.append(text)
                    custom_data_sources.append({
                        'title': title,
                        'category': category,
                        'type': 'custom'
                    })
        
        # Create embeddings for custom data if embedder is available
        if embedder and custom_data_texts:
            try:
                custom_data_embeddings = embedder.encode(custom_data_texts, convert_to_tensor=True)
                print(f"Created embeddings for {len(custom_data_texts)} custom data entries.")
            except Exception as e:
                print(f"Error creating custom data embeddings: {e}")
                custom_data_embeddings = np.array([])
        
        return custom_data_texts
    except Exception as e:
        print(f"Error loading custom data: {e}")
        return []

# === Save Custom Data ===
def save_custom_data():
    """Save custom data to file"""
    try:
        custom_data_file = "custom_data.txt"
        with open(custom_data_file, "w", encoding="utf-8") as f:
            for i, (text, source) in enumerate(zip(custom_data_texts, custom_data_sources)):
                f.write(f"{source['title']}\n")
                f.write(f"{source['category']}\n")
                f.write(f"{text}\n")
                if i < len(custom_data_texts) - 1:
                    f.write("--- CUSTOM ENTRY ---\n")
        
        print(f"Custom data saved to {custom_data_file}")
        return True
    except Exception as e:
        print(f"Error saving custom data: {e}")
        return False

# === Add Custom Data ===
def add_custom_data(title, category, content):
    """Add new custom data entry"""
    global custom_data_texts, custom_data_embeddings, custom_data_sources
    
    if not title.strip() or not category.strip() or not content.strip():
        return False, "All fields are required"
    
    # Add the new entry
    custom_data_texts.append(content.strip())
    custom_data_sources.append({
        'title': title.strip(),
        'category': category.strip(),
        'type': 'custom'
    })
    
    # Update embeddings if embedder is available
    if embedder:
        try:
            # Create new embedding for the added content
            new_embedding = embedder.encode([content.strip()], convert_to_tensor=True)
            
            if custom_data_embeddings is None:
                custom_data_embeddings = new_embedding
            else:
                # Concatenate with existing embeddings
                if TORCH_AVAILABLE:
                    custom_data_embeddings = torch.cat([custom_data_embeddings, new_embedding], dim=0)
                else:
                    # Fallback to numpy concatenation
                    custom_data_embeddings = np.concatenate([custom_data_embeddings, new_embedding], axis=0)
            
            print(f"Added custom data: {title} ({category})")
            return True, "Custom data added successfully"
        except Exception as e:
            print(f"Error creating embedding for new custom data: {e}")
            # Remove the added entry if embedding fails
            custom_data_texts.pop()
            custom_data_sources.pop()
            return False, f"Error creating embedding: {str(e)}"
    else:
        print(f"Added custom data (no embedding): {title} ({category})")
        return True, "Custom data added successfully (embeddings not available)"

# === Remove Custom Data ===
def remove_custom_data(index):
    """Remove custom data entry by index"""
    global custom_data_texts, custom_data_embeddings, custom_data_sources
    
    if index < 0 or index >= len(custom_data_texts):
        return False, "Invalid index"
    
    try:
        # Remove the entry
        removed_title = custom_data_sources[index]['title']
        custom_data_texts.pop(index)
        custom_data_sources.pop(index)
        
        # Recreate embeddings if embedder is available
        if embedder and custom_data_texts:
            try:
                custom_data_embeddings = embedder.encode(custom_data_texts, convert_to_tensor=True)
                print(f"✅ Recreated embeddings after removing custom data")
            except Exception as e:
                print(f"Error recreating embeddings: {e}")
                custom_data_embeddings = np.array([])
        elif not custom_data_texts:
            custom_data_embeddings = None
        
        print(f"Removed custom data: {removed_title}")
        return True, f"Removed: {removed_title}"
    except Exception as e:
        print(f"Error removing custom data: {e}")
        return False, f"Error removing data: {str(e)}"

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
        print(f"Error in basic chatbot: {e}")
        return f"Error processing request: {e}", []

# === Enhanced Semantic Retrieval ===
def retrieve_relevant_chunks(user_input, top_k=3, similarity_threshold=0.0):
    if not embedder:
        print("Sentence transformer not available")
        return []

    all_chunks = []
    all_embeddings = []
    all_sources = []
    
    # Add scraped data
    if scraped_data_embeddings is not None and scraped_data_embeddings.any():
        all_chunks.extend(scraped_data_pages)
        all_embeddings.append(scraped_data_embeddings)
        all_sources.extend(['scraped'] * len(scraped_data_pages))
    
    # Add custom data
    if custom_data_embeddings is not None and custom_data_embeddings.any():
        all_chunks.extend([(f"Custom: {source['title']}", text) for text, source in zip(custom_data_texts, custom_data_sources)])
        all_embeddings.append(custom_data_embeddings)
        all_sources.extend(['custom'] * len(custom_data_texts))
    
    if not all_chunks:
        print("⚠️ No data available for retrieval")
        return []

    try:
        # Concatenate all embeddings
        if len(all_embeddings) == 1:
            combined_embeddings = all_embeddings[0]
        else:
            if TORCH_AVAILABLE:
                combined_embeddings = torch.cat(all_embeddings, dim=0)
            else:
                # Fallback to numpy concatenation
                combined_embeddings = np.concatenate(all_embeddings, axis=0)
        
        query_embedding = embedder.encode(user_input, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, combined_embeddings)[0]
        
        # Debug: Print similarity scores
        print(f"Query: {user_input}")
        top_scores = similarities.topk(min(5, len(similarities))).values.tolist()
        print(f"Top similarity scores: {top_scores}")
        
        # Get top k results - always return top-k even if below threshold
        # This ensures we always have context to work with
        top_k_indices = similarities.argsort(descending=True)[:top_k]
        results = []
        
        for i in top_k_indices:
            similarity_score = similarities[i].item()
            results.append(all_chunks[i])
            source_type = all_sources[i]
            if similarity_score >= similarity_threshold:
                print(f"Selected {source_type} chunk {i} with similarity {similarity_score:.3f}")
            else:
                print(f"Selected {source_type} chunk {i} with similarity {similarity_score:.3f} (below threshold {similarity_threshold}, but included as top-k)")
        
        total_chunks = len(scraped_data_pages) + len(custom_data_texts)
        print(f"Retrieved {len(results)} relevant chunks out of {total_chunks} total")
        return results
        
    except Exception as e:
        print(f"Error in semantic retrieval: {e}")
        return []

# === Enhanced Gemini Response ===
def generate_gemini_response(user_input, relevant_chunks):
    if not GOOGLE_API_KEY:
        return "Google API key not configured. Please set your GOOGLE_API_KEY environment variable.", []
    
    if not GEMINI_MODEL_NAME:
        return "No working Gemini model found. Please check your API key configuration.", []

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        if relevant_chunks:
            context_str = "\n\n--- Context ---\n"
            source_urls = set()
            
            for i, (url, text) in enumerate(relevant_chunks):
                # Handle both scraped URLs and custom data titles
                if url.startswith('Custom:'):
                    context_str += f"Custom Data {i+1} ({url}):\n{text[:1000]}...\n\n"
                    source_urls.add(url)
                else:
                    context_str += f"Source {i+1} ({url}):\n{text[:1000]}...\n\n"
                    source_urls.add(url)
            
            context_str += "--- End Context ---"
            print(f"Context length: {len(context_str)} characters")
        else:
            context_str = "No relevant content found."
            source_urls = set()
            print("⚠️ No relevant chunks found for context")

        # Enhanced prompt that mentions custom data
        prompt = f"""You are a helpful and knowledgeable assistant for a company website. Answer the question naturally and conversationally using the information below.

When answering:
- Be direct and conversational
- Don't say "based on the context" or similar phrases
- If you find relevant information, share it naturally
- If the exact information isn't available, mention what you found and be honest about limitations
- Only say "I don't have that information" if there's truly nothing relevant
- If you're using custom company data, you can reference it naturally
- Provide comprehensive answers that combine information from both scraped website data and custom company data when relevant

Information available:
{context_str}

Question: {user_input}

Answer:"""

        response = model.generate_content(prompt)
        answer = response.text.strip()
        print(f"🤖 Generated response: {answer[:100]}...")
        return answer, list(source_urls)
        
    except Exception as e:
        print(f"Gemini error: {e}")
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg:
            return "Invalid API key. Please check your Google API key configuration.", []
        elif "PERMISSION_DENIED" in error_msg:
            return "Permission denied. Please check your API key permissions.", []
        elif "429" in error_msg and "quota" in error_msg.lower():
            return "API quota exceeded. You've hit the free tier limit (50 requests/day). Please upgrade your plan or wait until tomorrow.", []
        else:
            return f" Error generating response: {error_msg}", []

# === Load Once ===
@app.before_request
def load_data_once():
    global scraped_data_pages, scraped_data_texts, scraped_data_embeddings
    global custom_data_texts, custom_data_embeddings, custom_data_sources
    
    if not scraped_data_pages and embedder:
        print("📄 Loading scraped data...")
        data = load_scraped_data("scraped_data.txt")
        if data:
            scraped_data_pages = data
            scraped_data_texts = [text for _, text in data]
            try:
                scraped_data_embeddings = embedder.encode(scraped_data_texts, convert_to_tensor=True)
                print(f"Loaded {len(scraped_data_pages)} pages with embeddings.")
            except Exception as e:
                print(f"Error creating embeddings: {e}")
                scraped_data_embeddings = np.array([])
        else:
            print("No data loaded.")
            scraped_data_embeddings = np.array([])
    
    # Load custom data if not already loaded
    if not custom_data_texts and embedder:
        print("Loading custom data...")
        load_custom_data()

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
                "answer": "Sentence transformer model not available. Please check your installation.",
                "source_urls": []
            })

        if chatbot_type == 'basic':
            # Use basic chatbot functionality
            answer, urls = basic_chatbot_response(question)
        else:
            # Use Gemini chatbot
            relevant_chunks = retrieve_relevant_chunks(question, top_k=5, similarity_threshold=0.0)
            answer, urls = generate_gemini_response(question, relevant_chunks)

        return jsonify({"answer": answer, "source_urls": urls})

    except Exception as e:
        print(f"Server error: {e}")
        return jsonify({"answer": f"Internal server error: {e}", "source_urls": []})

# === Scraping Status Route ===
@app.route('/scraping_status')
def get_scraping_status():
    return jsonify({
        "scraping_in_progress": scraping_in_progress,
        "scraping_complete": scraping_complete,
        "data_loaded": len(scraped_data_pages) > 0,
        "total_pages": len(scraped_data_pages),
        "target_website": target_website
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

# === Set Website URL Route ===
@app.route('/set_website', methods=['POST'])
def set_website():
    global target_website
    try:
        data = request.get_json()
        new_website = data.get('website', '').strip()
        
        if not new_website:
            return jsonify({"status": "error", "message": "Website URL is required"})
        
        # Basic URL validation
        if not new_website.startswith(('http://', 'https://')):
            new_website = 'https://' + new_website
        
        target_website = new_website
        print(f"Target website updated to: {target_website}")
        
        return jsonify({
            "status": "success", 
            "message": f"Website updated to {target_website}",
            "website": target_website
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error setting website: {str(e)}"})

# === Get Current Website Route ===
@app.route('/get_website')
def get_website():
    return jsonify({
        "website": target_website,
        "status": "success"
    })

# === Reset Data Route ===
@app.route('/reset_data', methods=['POST'])
def reset_data():
    global scraped_data_pages, scraped_data_texts, scraped_data_embeddings, scraping_complete
    
    try:
        # Clear all scraped data
        scraped_data_pages = []
        scraped_data_texts = []
        scraped_data_embeddings = None
        scraping_complete = False
        
        print("🗑️ All scraped data has been reset")
        
        return jsonify({
            "status": "success",
            "message": "All scraped data has been reset"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Error resetting data: {str(e)}"
        })

# === Reset All Data Route ===
@app.route('/reset_all_data', methods=['POST'])
def reset_all_data():
    global scraped_data_pages, scraped_data_texts, scraped_data_embeddings, scraping_complete
    global custom_data_texts, custom_data_embeddings, custom_data_sources
    
    try:
        # Clear all data
        scraped_data_pages = []
        scraped_data_texts = []
        scraped_data_embeddings = None
        scraping_complete = False
        
        custom_data_texts = []
        custom_data_embeddings = None
        custom_data_sources = []
        
        # Remove custom data file
        if os.path.exists("custom_data.txt"):
            os.remove("custom_data.txt")
        
        print("🗑️ All data (scraped and custom) has been reset")
        
        return jsonify({
            "status": "success",
            "message": "All data has been reset"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Error resetting all data: {str(e)}"
        })

# === Custom Data Routes ===
@app.route('/add_custom_data', methods=['POST'])
def add_custom_data_route():
    try:
        data = request.get_json()
        title = data.get('title', '').strip()
        category = data.get('category', '').strip()
        content = data.get('content', '').strip()
        
        if not title or not category or not content:
            return jsonify({
                "status": "error",
                "message": "All fields (title, category, content) are required"
            })
        
        success, message = add_custom_data(title, category, content)
        
        if success:
            # Save to file
            save_custom_data()
            return jsonify({
                "status": "success",
                "message": message,
                "total_entries": len(custom_data_texts)
            })
        else:
            return jsonify({
                "status": "error",
                "message": message
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Server error: {str(e)}"
        })

@app.route('/get_custom_data')
def get_custom_data_route():
    try:
        custom_entries = []
        for i, (text, source) in enumerate(zip(custom_data_texts, custom_data_sources)):
            custom_entries.append({
                'index': i,
                'title': source['title'],
                'category': source['category'],
                'content': text[:200] + "..." if len(text) > 200 else text,
                'full_content': text
            })
        
        return jsonify({
            "status": "success",
            "entries": custom_entries,
            "total_count": len(custom_entries)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Server error: {str(e)}"
        })

@app.route('/remove_custom_data', methods=['POST'])
def remove_custom_data_route():
    try:
        data = request.get_json()
        index = data.get('index')
        
        if index is None:
            return jsonify({
                "status": "error",
                "message": "Index is required"
            })
        
        success, message = remove_custom_data(index)
        
        if success:
            # Save to file
            save_custom_data()
            return jsonify({
                "status": "success",
                "message": message,
                "total_entries": len(custom_data_texts)
            })
        else:
            return jsonify({
                "status": "error",
                "message": message
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Server error: {str(e)}"
        })

@app.route('/custom_data_stats')
def custom_data_stats():
    return jsonify({
        "total_custom_entries": len(custom_data_texts),
        "total_scraped_pages": len(scraped_data_pages),
        "custom_data_available": len(custom_data_texts) > 0,
        "scraped_data_available": len(scraped_data_pages) > 0
    })

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

# === Enhanced Debug Route ===
@app.route('/debug')
def debug_info():
    return jsonify({
        "total_scraped_pages": len(scraped_data_pages),
        "total_custom_entries": len(custom_data_texts),
        "sample_scraped_pages": [{"url": url, "text_preview": text[:200]} for url, text in scraped_data_pages[:3]],
        "sample_custom_entries": [{"title": source['title'], "category": source['category'], "text_preview": text[:200]} for text, source in zip(custom_data_texts[:3], custom_data_sources[:3])],
        "scraped_embeddings_shape": scraped_data_embeddings.shape if scraped_data_embeddings is not None else None,
        "custom_embeddings_shape": custom_data_embeddings.shape if custom_data_embeddings is not None else None,
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
    
    if not GEMINI_MODEL_NAME:
        return jsonify({"error": "No working model found"})
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
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
    
    # Load custom data on startup
    if embedder:
        load_custom_data()
    
    # Start scraping automatically in background
    print("Starting automatic web scraping...")
    scraping_thread = threading.Thread(target=run_scraper)
    scraping_thread.daemon = True
    scraping_thread.start()
    
    print("Starting Flask web server...")
    app.run(debug=True)