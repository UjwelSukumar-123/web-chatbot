"""
API Module
Contains all FastAPI routes and endpoints
"""

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
import openai

# Import modules
from chatbot import (
    basic_chatbot_response, 
    retrieve_relevant_chunks, 
    generate_openai_response
)
from custom_data import (
    add_custom_data, 
    remove_custom_data, 
    save_custom_data, 
    get_custom_data,
    get_custom_data_embeddings,
    load_custom_data
)
from data_manager import load_embeddings_data, load_structured_embeddings_data
from scraper import run_scraper as run_scraper_function

# Pydantic Models
class QuestionRequest(BaseModel):
    question: str
    chatbot_type: Optional[str] = "openai"

class WebsiteRequest(BaseModel):
    website: str
    auto_scrape: Optional[bool] = False

class CustomDataRequest(BaseModel):
    title: str
    category: str
    content: str

class RemoveCustomDataRequest(BaseModel):
    index: int


def create_app(
    embedder,
    openai_api_key,
    openai_model_name,
    app_state
):
    """Create and configure FastAPI app with all routes"""
    
    app = FastAPI(title="Web Chatbot System", version="1.0.0")
    
    # Helper to access state
    def get_state(key, default=None):
        return app_state.get(key, default)
    
    def set_state(key, value):
        app_state[key] = value
    
    @app.get('/', response_class=HTMLResponse)
    async def index():
        with open('templates/index.html', 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    
    @app.post('/ask')
    async def ask(question: str = Form(...), chatbot_type: str = Form("openai")):
        try:
            question = question.strip()
            chatbot_type = chatbot_type or 'openai'
            
            if not question:
                return JSONResponse(content={"answer": "Please enter a question.", "source_urls": []})
            
            if not embedder:
                return JSONResponse(content={
                    "answer": "Sentence transformer model not available. Please check your installation.",
                    "source_urls": []
                })
            
            if chatbot_type == 'basic':
                answer, urls = basic_chatbot_response(
                    question, 
                    embedder, 
                    get_state("scraped_data_pages"), 
                    get_state("scraped_data_embeddings")
                )
                return JSONResponse(content={"answer": answer, "source_urls": urls})
            else:
                # Detect query types
                address_keywords = ['address', 'location', 'where', 'office', 'contact', 'phone', 'email', 'located']
                product_keywords = ['product', 'service', 'offering', 'what do you', 'what does', 'what can', 'offer', 'provide', 'sell']
                about_keywords = ['tell me about', 'what is', 'describe', 'who is', 'about', 'information', 'overview', 
                                  'background', 'what does', 'details', 'about us', 'story', 'history', 'what are']
                is_address_query = any(keyword in question.lower() for keyword in address_keywords)
                is_product_query = any(keyword in question.lower() for keyword in product_keywords)
                is_about_query = any(keyword in question.lower() for keyword in about_keywords)
                is_generic_query = question.lower().strip() in ['hi', 'hello', 'hey', 'greetings', 'what can you do', 'help', '?']
                
                # Set top_k based on query type
                if is_about_query:
                    top_k = 15
                elif is_address_query or is_product_query:
                    top_k = 12
                elif is_generic_query:
                    top_k = 10
                else:
                    top_k = 8
                
                relevant_chunks = retrieve_relevant_chunks(
                    question,
                    embedder,
                    get_state("scraped_data_pages"),
                    get_state("scraped_data_embeddings"),
                    get_state("structured_data_texts"),
                    get_state("structured_data_embeddings"),
                    get_state("structured_data_sources"),
                    get_state("custom_data_texts"),
                    get_state("custom_data_embeddings"),
                    get_state("custom_data_sources"),
                    top_k=top_k,
                    similarity_threshold=0.0
                )
                
                answer, urls, token_usage = generate_openai_response(
                    question,
                    relevant_chunks,
                    openai_api_key,
                    openai_model_name,
                    is_about_query=is_about_query,
                    is_address_query=is_address_query,
                    is_product_query=is_product_query
                )
                
                return JSONResponse(content={"answer": answer, "source_urls": urls})
        
        except Exception as e:
            print(f"Server error: {e}")
            import traceback
            traceback.print_exc()
            return JSONResponse(content={"answer": f"Internal server error: {e}", "source_urls": []})
    
    @app.get('/scraping_status')
    async def get_scraping_status():
        scraping_in_progress = get_state("scraping_in_progress") or False
        scraping_complete = get_state("scraping_complete") or False
        target_website = get_state("target_website") or ""
        scraped_pages = get_state("scraped_data_pages") or []
        
        # If scraping is complete but flag is not set, check if we have data
        if not scraping_complete and not scraping_in_progress and len(scraped_pages) > 0:
            # Data exists but flag wasn't set - likely scraping completed but flag wasn't updated
            scraping_complete = True
        
        return JSONResponse(content={
            "scraping_in_progress": scraping_in_progress,
            "scraping_complete": scraping_complete,
            "data_loaded": len(scraped_pages) > 0,
            "total_pages": len(scraped_pages),
            "target_website": target_website
        })
    
    # Note: /start_scraping route is added in main.py to access the threaded function
    
    @app.post('/set_website')
    async def set_website_route(request_data: WebsiteRequest):
        try:
            new_website = request_data.website.strip()
            
            if not new_website:
                return JSONResponse(content={"status": "error", "message": "Website URL is required"})
            
            if not new_website.startswith(('http://', 'https://')):
                new_website = 'https://' + new_website
            
            # If website changed, clear old data
            if get_state("target_website") != new_website:
                print(f"⚠️ Website changed from {get_state('target_website')} to {new_website}. Clearing old data...")
                set_state("scraped_data_pages", [])
                set_state("scraped_data_texts", [])
                set_state("scraped_data_embeddings", None)
                set_state("structured_data_texts", [])
                set_state("structured_data_embeddings", None)
                set_state("structured_data_sources", [])
                set_state("scraping_complete", False)
                print("✅ Old data cleared")
            
            set_state("target_website", new_website)
            print(f"✅ Target website updated to: {new_website}")
            
            response_data = {
                "status": "success",
                "message": f"Website updated to {new_website}",
                "website": new_website
            }
            
            if request_data.auto_scrape:
                response_data["message"] = f"Website updated to {new_website}. Use /start_scraping to begin scraping."
                response_data["scraping_status"] = "ready"
            
            return JSONResponse(content=response_data)
        
        except Exception as e:
            return JSONResponse(content={"status": "error", "message": f"Error setting website: {str(e)}"})
    
    @app.get('/get_website')
    async def get_website():
        return JSONResponse(content={
            "website": get_state("target_website"),
            "status": "success"
        })
    
    @app.post('/reload_data')
    async def reload_data():
        try:
            if not embedder:
                return JSONResponse(content={
                    "status": "error",
                    "message": "Sentence transformer embedder not available"
                })
            
            print("🔄 Manually reloading scraped data from embeddings...")
            pages_data, embeddings = load_embeddings_data(embedder=embedder)
            
            if pages_data:
                set_state("scraped_data_pages", pages_data)
                set_state("scraped_data_texts", [text for _, text in pages_data])
                set_state("scraped_data_embeddings", embeddings)
                return JSONResponse(content={
                    "status": "success",
                    "message": f"Successfully reloaded {len(pages_data)} pages from embeddings",
                    "total_pages": len(pages_data),
                    "source": "embeddings"
                })
            else:
                return JSONResponse(content={
                    "status": "error",
                    "message": "No embeddings found. Please run scraping first."
                })
        
        except Exception as e:
            print(f"❌ Error reloading data: {e}")
            import traceback
            traceback.print_exc()
            return JSONResponse(content={
                "status": "error",
                "message": f"Error reloading data: {str(e)}"
            })
    
    @app.post('/reset_data')
    async def reset_data():
        try:
            set_state("scraped_data_pages", [])
            set_state("scraped_data_texts", [])
            set_state("scraped_data_embeddings", None)
            set_state("structured_data_texts", [])
            set_state("structured_data_embeddings", None)
            set_state("structured_data_sources", [])
            set_state("scraping_complete", False)
            
            print("✅ All scraped data has been reset")
            
            return JSONResponse(content={
                "status": "success",
                "message": "All scraped data has been reset"
            })
        
        except Exception as e:
            return JSONResponse(content={
                "status": "error",
                "message": f"Error resetting data: {str(e)}"
            })
    
    @app.post('/reset_all_data')
    async def reset_all_data():
        try:
            set_state("scraped_data_pages", [])
            set_state("scraped_data_texts", [])
            set_state("scraped_data_embeddings", None)
            set_state("structured_data_texts", [])
            set_state("structured_data_embeddings", None)
            set_state("structured_data_sources", [])
            set_state("custom_data_texts", [])
            set_state("custom_data_embeddings", None)
            set_state("custom_data_sources", [])
            set_state("scraping_complete", False)
            
            # Remove custom data file
            custom_data_file = os.path.join("custom_data", "custom_data.txt")
            if os.path.exists(custom_data_file):
                os.remove(custom_data_file)
            
            print("✅ All data (scraped and custom) has been reset")
            
            return JSONResponse(content={
                "status": "success",
                "message": "All data has been reset"
            })
        
        except Exception as e:
            return JSONResponse(content={
                "status": "error",
                "message": f"Error resetting all data: {str(e)}"
            })
    
    @app.post('/add_custom_data')
    async def add_custom_data_route(request_data: CustomDataRequest):
        try:
            title = request_data.title.strip() if request_data.title else ""
            category = request_data.category.strip() if request_data.category else ""
            content = request_data.content.strip() if request_data.content else ""
            
            if not title or not category or not content:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": "All fields (title, category, content) are required and cannot be empty"
                    }
                )
            
            print(f"📝 Adding custom data: title='{title}', category='{category}'")
            success, message = add_custom_data(title, category, content, embedder)
            
            if success:
                save_success = save_custom_data()
                if not save_success:
                    print(f"⚠️ Warning: Failed to save custom data to file, but data was added to memory")
                
                # Update global state
                custom_data_texts_new, custom_data_sources_new = get_custom_data()
                set_custom_data(custom_data_texts_new, get_custom_data_embeddings(), custom_data_sources_new)
                
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "success",
                        "message": message,
                        "total_entries": len(custom_data_texts_new),
                        "data": {
                            "title": title,
                            "category": category,
                            "content_preview": content[:100] + "..." if len(content) > 100 else content
                        }
                    }
                )
            else:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": message
                    }
                )
        
        except Exception as e:
            print(f"❌ Error in add_custom_data_route: {e}")
            import traceback
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Server error: {str(e)}"
                }
            )
    
    @app.get('/get_custom_data')
    async def get_custom_data_route():
        try:
            custom_entries = []
            for i, (text, source) in enumerate(zip(get_state("custom_data_texts"), get_state("custom_data_sources"))):
                custom_entries.append({
                    'index': i,
                    'title': source['title'],
                    'category': source['category'],
                    'content': text[:200] + "..." if len(text) > 200 else text,
                    'full_content': text
                })
            
            return JSONResponse(content={
                "status": "success",
                "entries": custom_entries,
                "total_count": len(custom_entries)
            })
        
        except Exception as e:
            return JSONResponse(content={
                "status": "error",
                "message": f"Server error: {str(e)}"
            })
    
    @app.post('/remove_custom_data')
    async def remove_custom_data_route(request_data: RemoveCustomDataRequest):
        try:
            index = request_data.index
            
            if index is None:
                return JSONResponse(content={
                    "status": "error",
                    "message": "Index is required"
                })
            
            success, message = remove_custom_data(index, embedder)
            
            if success:
                save_custom_data()
                # Update global state
                custom_data_texts_new, custom_data_sources_new = get_custom_data()
                set_state("custom_data_texts", custom_data_texts_new)
                set_state("custom_data_embeddings", get_custom_data_embeddings())
                set_state("custom_data_sources", custom_data_sources_new)
                
                return JSONResponse(content={
                    "status": "success",
                    "message": message,
                    "total_entries": len(custom_data_texts_new)
                })
            else:
                return JSONResponse(content={
                    "status": "error",
                    "message": message
                })
        
        except Exception as e:
            return JSONResponse(content={
                "status": "error",
                "message": f"Server error: {str(e)}"
            })
    
    @app.get('/custom_data_stats')
    async def custom_data_stats():
        return JSONResponse(content={
            "total_custom_entries": len(get_state("custom_data_texts")),
            "total_scraped_pages": len(get_state("scraped_data_pages")),
            "custom_data_available": len(get_state("custom_data_texts")) > 0,
            "scraped_data_available": len(get_state("scraped_data_pages")) > 0
        })
    
    @app.get('/health')
    async def health_check():
        status = {
            "api_key_configured": bool(openai_api_key),
            "api_key_preview": openai_api_key[:10] + "..." if openai_api_key else None,
            "model_name": openai_model_name,
            "embedder_loaded": bool(embedder),
            "data_loaded": len(get_state("scraped_data_pages")) > 0,
            "embeddings_created": get_state("scraped_data_embeddings") is not None and (
                (hasattr(get_state("scraped_data_embeddings"), 'shape') and get_state("scraped_data_embeddings").shape[0] > 0) if get_state("scraped_data_embeddings") is not None else False
            ),
            "scraping_status": {
                "in_progress": get_state("scraping_in_progress"),
                "complete": get_state("scraping_complete")
            }
        }
        return JSONResponse(content=status)
    
    @app.get('/debug')
    async def debug_info():
        return JSONResponse(content={
            "total_scraped_pages": len(get_state("scraped_data_pages")),
            "total_custom_entries": len(get_state("custom_data_texts")),
            "sample_scraped_pages": [{"url": url, "text_preview": text[:200]} for url, text in get_state("scraped_data_pages")[:3]],
            "sample_custom_entries": [{"title": source['title'], "category": source['category'], "text_preview": text[:200]} for text, source in zip(get_state("custom_data_texts")[:3], get_state("custom_data_sources")[:3])],
            "scraped_embeddings_shape": get_state("scraped_data_embeddings").shape if get_state("scraped_data_embeddings") is not None else None,
            "custom_embeddings_shape": get_state("custom_data_embeddings").shape if get_state("custom_data_embeddings") is not None else None,
            "api_key_configured": bool(openai_api_key),
            "embedder_loaded": bool(embedder),
            "scraping_status": {
                "in_progress": get_state("scraping_in_progress"),
                "complete": get_state("scraping_complete")
            }
        })
    
    @app.post('/test_query')
    async def test_query(request_data: QuestionRequest):
        try:
            question = request_data.question.strip()
            if not question:
                return JSONResponse(content={"error": "Please enter a question."})
            
            relevant_chunks = retrieve_relevant_chunks(
                question,
                embedder,
                get_state("scraped_data_pages"),
                get_state("scraped_data_embeddings"),
                get_state("structured_data_texts"),
                get_state("structured_data_embeddings"),
                get_state("structured_data_sources"),
                get_state("custom_data_texts"),
                get_state("custom_data_embeddings"),
                get_state("custom_data_sources"),
                top_k=5,
                similarity_threshold=0.05
            )
            
            return JSONResponse(content={
                "question": question,
                "relevant_chunks_count": len(relevant_chunks),
                "relevant_chunks": [{"url": url, "text_preview": text[:300]} for url, text in relevant_chunks],
                "total_pages_available": len(get_state("scraped_data_pages"))
            })
        
        except Exception as e:
            return JSONResponse(content={"error": str(e)})
    
    @app.get('/test_api')
    async def test_api_key():
        if not openai_api_key:
            return JSONResponse(content={"error": "No API key configured"})
        
        try:
            client = openai.OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model=openai_model_name,
                messages=[
                    {"role": "user", "content": "Hello, this is a test."}
                ],
                max_tokens=50
            )
            
            return JSONResponse(content={
                "status": "success",
                "api_key_preview": openai_api_key[:10] + "...",
                "model": openai_model_name,
                "response": response.choices[0].message.content,
                "message": "API key is working"
            })
        except Exception as e:
            error_msg = str(e)
            return JSONResponse(content={
                "status": "error",
                "api_key_preview": openai_api_key[:10] + "..." if openai_api_key else None,
                "error": error_msg,
                "is_quota_error": "quota" in error_msg.lower() or "429" in error_msg or "rate_limit" in error_msg.lower()
            })
    
    @app.get('/project_info')
    async def get_project_info():
        if not openai_api_key:
            return JSONResponse(content={"error": "No API key configured"})
        
        try:
            return JSONResponse(content={
                "api_key_preview": openai_api_key[:10] + "...",
                "model": openai_model_name,
                "note": "OpenAI API key is configured",
                "action_required": "Check your OpenAI account for usage limits and billing information"
            })
        except Exception as e:
            return JSONResponse(content={
                "error": str(e),
                "api_key_preview": openai_api_key[:10] + "..." if openai_api_key else None
            })
    
    return app

