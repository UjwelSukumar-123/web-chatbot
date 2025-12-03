"""
Main Entry Point
Orchestrates all modules with threading support
"""

import os
import threading
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import uvicorn

# Import modules
from api import create_app
from data_manager import load_embeddings_data, load_structured_embeddings_data, embeddings_match_website
from custom_data import load_custom_data, get_custom_data, get_custom_data_embeddings
from scraper import run_scraper

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
TARGET_WEBSITE = os.getenv("TARGET_WEBSITE", "")

# Global state
app_state = {
    "scraped_data_pages": [],
    "scraped_data_texts": [],
    "scraped_data_embeddings": None,
    "structured_data_texts": [],
    "structured_data_embeddings": None,
    "structured_data_sources": [],
    "custom_data_texts": [],
    "custom_data_embeddings": None,
    "custom_data_sources": [],
    "scraping_in_progress": False,
    "scraping_complete": False,
    "target_website": TARGET_WEBSITE,
    "should_scrape": False,
    "user_name": ""  # Store user's name for personalized greetings
}

# State setters - Now updating app_state directly
def set_scraped_data(pages, texts, embeddings):
    app_state["scraped_data_pages"] = pages
    app_state["scraped_data_texts"] = texts
    app_state["scraped_data_embeddings"] = embeddings

def set_structured_data(texts, embeddings, sources):
    app_state["structured_data_texts"] = texts
    app_state["structured_data_embeddings"] = embeddings
    app_state["structured_data_sources"] = sources

def set_custom_data(texts, embeddings, sources):
    app_state["custom_data_texts"] = texts
    app_state["custom_data_embeddings"] = embeddings
    app_state["custom_data_sources"] = sources

def set_scraping_in_progress(value):
    app_state["scraping_in_progress"] = value

def set_scraping_complete(value):
    app_state["scraping_complete"] = value

def set_target_website(value):
    app_state["target_website"] = value

# Load embedder
print("Loading sentence transformer model...")
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Sentence transformer model loaded successfully")
except Exception as e:
    print(f"❌ Error loading sentence transformer: {e}")
    embedder = None

# Load data on startup
if embedder:
    print("📂 Loading scraped data from embeddings...")
    pages_data, embeddings = load_embeddings_data(embedder=embedder)
    
    # Check if embeddings exist and match the target website
    app_state["should_scrape"] = False
    if not pages_data or len(pages_data) == 0:
        print("⚠️ No embeddings found.")
        app_state["should_scrape"] = True
        set_scraping_complete(False)
    elif app_state["target_website"]:
        # Check if embeddings match the current target website
        if not embeddings_match_website(pages_data, app_state["target_website"]):
            print(f"⚠️ Existing embeddings are for a different website. Need to scrape {app_state['target_website']}.")
            app_state["should_scrape"] = True
            set_scraping_complete(False)
        else:
            set_scraped_data(pages_data, [text for _, text in pages_data], embeddings)
            set_scraping_complete(True)  # Data exists, so scraping was completed
            print(f"✅ Loaded {len(pages_data)} pages with embeddings for {app_state['target_website']}")
    else:
        # No target website set, but embeddings exist - use them
        set_scraped_data(pages_data, [text for _, text in pages_data], embeddings)
        set_scraping_complete(True)  # Data exists, so scraping was completed
        print(f"✅ Loaded {len(pages_data)} pages with embeddings")
    
    print("📋 Loading structured data from embeddings...")
    structured_texts, structured_sources, structured_emb = load_structured_embeddings_data(embedder=embedder)
    if structured_texts:
        set_structured_data(structured_texts, structured_emb, structured_sources)
        print(f"✅ Loaded {len(structured_texts)} structured entries")
    
    print("📝 Loading custom data...")
    load_custom_data(embedder=embedder)
    custom_texts, custom_sources = get_custom_data()
    set_custom_data(custom_texts, get_custom_data_embeddings(), custom_sources)
    if custom_texts:
        print(f"✅ Loaded {len(custom_texts)} custom entries")

# Scraping function with threading
def run_scraper_threaded():
    """Run scraper in a separate thread"""
    
    if app_state["scraping_in_progress"]:
        print("⚠️ Scraping already in progress")
        return
    
    # Get target website before starting
    target_website = app_state.get("target_website", "")
    if not target_website:
        print("No target website set. Cannot start scraping.")
        return
    
    set_scraping_in_progress(True)
    set_scraping_complete(False)
    print(f"Starting scraping for: {target_website}")
    
    try:
        max_pages = int(os.getenv("MAX_SCRAPE_PAGES", "200"))
        pages, embeddings, struct_texts, struct_emb, struct_sources = run_scraper(
            target_website, 
            embedder, 
            max_pages=max_pages
        )
        
        if pages and len(pages) > 0:
            set_scraped_data(pages, [text for _, text in pages], embeddings)
            set_structured_data(struct_texts, struct_emb, struct_sources)
            set_scraping_complete(True)
            print(f"Scraping completed successfully - {len(pages)} pages scraped")
        else:
            set_scraping_complete(False)
            print("⚠️ Scraping completed but no data was collected")
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        import traceback
        traceback.print_exc()
        set_scraping_complete(False)
    finally:
        set_scraping_in_progress(False)
        print(f"Scraping process finished. Status: in_progress=False, complete={app_state['scraping_complete']}")

# Create FastAPI app
app = create_app(
    embedder=embedder,
    openai_api_key=OPENAI_API_KEY,
    openai_model_name=OPENAI_MODEL_NAME,
    app_state=app_state
)

# Add route for starting scraping
@app.post('/start_scraping')
async def start_scraping():
    from fastapi.responses import JSONResponse
    
    if app_state["scraping_in_progress"]:
        return JSONResponse(content={"status": "already_running", "message": "Scraping is already in progress"})
    
    # Check if target website is set
    if not app_state.get("target_website"):
        return JSONResponse(content={
            "status": "error", 
            "message": "No target website set. Please set a website first using /set_website endpoint."
        })
    
    # Start scraping in a separate thread
    thread = threading.Thread(target=run_scraper_threaded)
    thread.daemon = True
    thread.start()
    
    return JSONResponse(content={
        "status": "started", 
        "message": f"Scraping started successfully for {app_state['target_website']}",
        "target_website": app_state["target_website"]
    })

# Startup event
@app.on_event("startup")
async def startup_event():
    print("\n=== Web Chatbot System Started ===")
    print(f"API Key Configured: {'✅' if OPENAI_API_KEY else '❌'}")
    print(f"Model: {OPENAI_MODEL_NAME if OPENAI_API_KEY else 'N/A'}")
    print(f"Embedder Loaded: {'✅' if embedder else '❌'}")
    print(f"Target Website: {app_state['target_website'] if app_state['target_website'] else 'Not set'}")
    print(f"Scraped Pages: {len(app_state['scraped_data_pages'])}")
    print(f"Structured Entries: {len(app_state['structured_data_texts'])}")
    print(f"Custom Entries: {len(app_state['custom_data_texts'])}")
    print("================================\n")
    
    # Auto-start scraping if needed
    if app_state["should_scrape"] and app_state["target_website"]:
        print(f"🚀 Starting automatic web scraping for {app_state['target_website']}...")
        thread = threading.Thread(target=run_scraper_threaded)
        thread.daemon = True
        thread.start()
    elif app_state["scraped_data_pages"] and len(app_state["scraped_data_pages"]) > 0:
        print(f"✅ Using existing scraped data ({len(app_state['scraped_data_pages'])} pages)")
    elif not app_state["target_website"]:
        print("⚠️ No target website set. Use /set_website endpoint to configure a website.")

if __name__ == '__main__':
    print("\n=== Starting Web Chatbot System ===")
    print(f"API Key Configured: {'✅' if OPENAI_API_KEY else '❌'}")
    print(f"Model: {OPENAI_MODEL_NAME if OPENAI_API_KEY else 'N/A'}")
    print(f"Embedder Loaded: {'✅' if embedder else '❌'}")
    print("================================\n")
    
    if not OPENAI_API_KEY:
        print("⚠️ WARNING: App will run but won't be able to generate OpenAI responses without API key")
        print("💡 Basic chatbot will still work with scraped data")
    else:
        print("💡 Note: Check your OpenAI account for usage limits and billing information.")
    
    print("Starting FastAPI web server...")
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")

