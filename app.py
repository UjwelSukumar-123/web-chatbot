import os
import re
import numpy as np
import pickle
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import openai
from dotenv import load_dotenv
import threading
import time
from typing import Optional

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")  # Default to gpt-3.5-turbo, can be overridden

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    print("OpenAI API key loaded from .env file")
    print(f"API Key (first 10 chars): {OPENAI_API_KEY[:10]}...")
    print(f"Using model: {OPENAI_MODEL_NAME}")
else:
    print("OPENAI_API_KEY not found in .env file")

# === FastAPI App ===
app = FastAPI(title="Web Chatbot System", version="1.0.0")

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
# Initialize target_website from environment variable or use empty (will be set via API)
target_website = os.getenv("TARGET_WEBSITE", "")

# === Custom Data Store ===
custom_data_texts = []
custom_data_embeddings = None
custom_data_sources = []  # Store metadata about custom data entries

# === Structured Data Store ===
structured_data_texts = []
structured_data_embeddings = None
structured_data_sources = []  # Store metadata about structured data entries

# === Save Embeddings Function ===
def save_embeddings_data(pages_data, embeddings, metadata_file="embeddings_metadata.pkl", embeddings_file="embeddings.npy"):
    """Save scraped data as embeddings and metadata"""
    try:
        # Save metadata (URLs and texts) as pickle
        with open(metadata_file, "wb") as f:
            pickle.dump(pages_data, f)
        print(f"✅ Saved metadata to {metadata_file} ({len(pages_data)} pages)")
        
        # Save embeddings as numpy array
        if embeddings is not None:
            # Convert to numpy if it's a torch tensor
            if TORCH_AVAILABLE and hasattr(embeddings, 'cpu'):
                embeddings_np = embeddings.cpu().numpy()
            elif hasattr(embeddings, 'numpy'):
                embeddings_np = embeddings.numpy()
            else:
                embeddings_np = np.array(embeddings)
            
            np.save(embeddings_file, embeddings_np)
            print(f"✅ Saved embeddings to {embeddings_file} (shape: {embeddings_np.shape})")
        
        return True
    except Exception as e:
        print(f"❌ Error saving embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False

# === Load Embeddings Function ===
def load_embeddings_data(metadata_file="embeddings_metadata.pkl", embeddings_file="embeddings.npy"):
    """Load scraped data from embeddings and metadata files"""
    global scraped_data_pages, scraped_data_texts, scraped_data_embeddings
    
    try:
        # Load metadata
        if not os.path.exists(metadata_file):
            print(f"⚠️ Metadata file {metadata_file} not found")
            return []
        
        with open(metadata_file, "rb") as f:
            pages_data = pickle.load(f)
        
        if not pages_data:
            print(f"⚠️ No data in {metadata_file}")
            return []
        
        # Load embeddings
        embeddings_np = None
        if os.path.exists(embeddings_file):
            embeddings_np = np.load(embeddings_file)
            print(f"Loaded embeddings from {embeddings_file} (shape: {embeddings_np.shape})")
            
            # Convert to tensor if torch is available
            if TORCH_AVAILABLE:
                import torch
                scraped_data_embeddings = torch.from_numpy(embeddings_np)
            else:
                scraped_data_embeddings = embeddings_np
        else:
            print(f"⚠️ Embeddings file {embeddings_file} not found, will recreate embeddings")
            # Recreate embeddings if embedder is available
            if embedder:
                texts = [text for _, text in pages_data]
                scraped_data_embeddings = embedder.encode(texts, convert_to_tensor=True)
                # Save the recreated embeddings
                save_embeddings_data(pages_data, scraped_data_embeddings, metadata_file, embeddings_file)
        
        # Update global variables
        scraped_data_pages = pages_data
        scraped_data_texts = [text for _, text in pages_data]
        
        print(f"✅ Loaded {len(pages_data)} pages with embeddings")
        return pages_data
        
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        import traceback
        traceback.print_exc()
        return []

# === Save Structured Data Embeddings ===
def save_structured_embeddings_data(texts, sources, embeddings, metadata_file="structured_metadata.pkl", embeddings_file="structured_embeddings.npy"):
    """Save structured data as embeddings"""
    try:
        # Save metadata
        with open(metadata_file, "wb") as f:
            pickle.dump({'texts': texts, 'sources': sources}, f)
        print(f"✅ Saved structured metadata to {metadata_file}")
        
        # Save embeddings
        if embeddings is not None:
            if TORCH_AVAILABLE and hasattr(embeddings, 'cpu'):
                embeddings_np = embeddings.cpu().numpy()
            elif hasattr(embeddings, 'numpy'):
                embeddings_np = embeddings.numpy()
            else:
                embeddings_np = np.array(embeddings)
            
            np.save(embeddings_file, embeddings_np)
            print(f"✅ Saved structured embeddings to {embeddings_file} (shape: {embeddings_np.shape})")
        
        return True
    except Exception as e:
        print(f"❌ Error saving structured embeddings: {e}")
        return False

# === Load Structured Data Embeddings ===
def load_structured_embeddings_data(metadata_file="structured_metadata.pkl", embeddings_file="structured_embeddings.npy"):
    """Load structured data from embeddings"""
    global structured_data_texts, structured_data_embeddings, structured_data_sources
    
    try:
        if not os.path.exists(metadata_file):
            return []
        
        with open(metadata_file, "rb") as f:
            data = pickle.load(f)
        
        structured_data_texts = data.get('texts', [])
        structured_data_sources = data.get('sources', [])
        
        # Load embeddings
        if os.path.exists(embeddings_file):
            embeddings_np = np.load(embeddings_file)
            if TORCH_AVAILABLE:
                import torch
                structured_data_embeddings = torch.from_numpy(embeddings_np)
            else:
                structured_data_embeddings = embeddings_np
            print(f"✅ Loaded structured embeddings (shape: {embeddings_np.shape})")
        else:
            # Recreate embeddings
            if embedder and structured_data_texts:
                structured_data_embeddings = embedder.encode(structured_data_texts, convert_to_tensor=True)
                save_structured_embeddings_data(structured_data_texts, structured_data_sources, structured_data_embeddings)
        
        return structured_data_texts
    except Exception as e:
        print(f"❌ Error loading structured embeddings: {e}")
        return []

# === Scraping Function ===
def run_scraper():
    global scraped_data_pages, scraped_data_texts, scraped_data_embeddings, scraping_in_progress, scraping_complete
    global structured_data_texts, structured_data_embeddings, structured_data_sources
    
    scraping_in_progress = True
    print("Starting web scraping...")
    
    try:
        # Run the scraper using the target website
        website = target_website
        if not website:
            print("❌ No target website set. Please set a website using /set_website endpoint first.")
            scraping_complete = False
            scraping_in_progress = False
            return
        
        print(f"Scraping website: {website}")
        
        # Increased max_pages for deeper scraping - configurable via environment variable
        max_pages = int(os.getenv("MAX_SCRAPE_PAGES", "200"))
        print(f"Maximum pages to scrape: {max_pages}")
        
        general_scraped_data, structured_scraped_data = crawl_website(website, max_pages=max_pages)
        
        # Process general scraped data - create embeddings directly
        url_text_pairs = []
        for url, content in general_scraped_data.items():
            # Clean and validate the text
            cleaned_text = clean_text_content(content.strip())
            
            # Only add if content is valid
            if is_valid_content(cleaned_text, min_length=50):
                url_text_pairs.append((url.strip(), cleaned_text))
        
        if not url_text_pairs:
            print("⚠️ No valid pages scraped")
            scraping_complete = False
            scraping_in_progress = False
            return
        
        print(f"✅ Processed {len(url_text_pairs)} valid pages")
        
        # Create embeddings immediately
        if embedder:
            texts = [text for _, text in url_text_pairs]
            print(f"🔄 Creating embeddings for {len(texts)} pages...")
            embeddings = embedder.encode(texts, convert_to_tensor=True)
            
            # Save embeddings and metadata
            save_embeddings_data(url_text_pairs, embeddings)
            
            # Update global variables
            scraped_data_pages = url_text_pairs
            scraped_data_texts = texts
            scraped_data_embeddings = embeddings
            
            print(f"✅ Created and saved embeddings for {len(url_text_pairs)} pages")
        else:
            print("⚠️ Embedder not available, cannot create embeddings")
            scraping_complete = False
            scraping_in_progress = False
            return
        
        # Process structured data - create embeddings directly
        structured_data_texts = []
        structured_data_sources = []
        
        # Process contact information
        for entry in structured_scraped_data.get('contact_info', []):
            url = entry['url']
            contact_text_parts = []
            for key, value in entry['data'].items():
                if isinstance(value, list):
                    contact_text_parts.append(f"{key.replace('_', ' ').title()}: {', '.join(value)}")
                else:
                    contact_text_parts.append(f"{key.replace('_', ' ').title()}: {value}")
            
            if contact_text_parts:
                contact_text = f"Contact Information from {url}:\n" + "\n".join(contact_text_parts)
                structured_data_texts.append(contact_text)
                structured_data_sources.append({'url': url, 'type': 'contact_info'})
        
        # Process job opportunities
        for entry in structured_scraped_data.get('job_opportunities', []):
            url = entry['url']
            jobs = entry['data']
            if jobs:
                job_text = f"Job Opportunities from {url}:\n" + "\n".join([f"  - {job}" for job in jobs])
                structured_data_texts.append(job_text)
                structured_data_sources.append({'url': url, 'type': 'job_opportunities'})
        
        # Process services/products
        for entry in structured_scraped_data.get('services', []):
            url = entry['url']
            services = entry['data']
            if services:
                service_text_parts = [f"Services/Products from {url}:"]
                for service in services:
                    service_text_parts.append(f"  - {service.get('name', 'N/A')}")
                    if service.get('description'):
                        service_text_parts.append(f"    Description: {service['description']}")
                service_text = "\n".join(service_text_parts)
                structured_data_texts.append(service_text)
                structured_data_sources.append({'url': url, 'type': 'services'})
        
        # Process addresses
        for entry in structured_scraped_data.get('addresses', []):
            url = entry['url']
            addresses = entry['data']
            if addresses:
                address_text = f"Addresses from {url}:\n" + "\n".join([f"  - {addr}" for addr in addresses])
                structured_data_texts.append(address_text)
                structured_data_sources.append({'url': url, 'type': 'addresses'})
        
        # Process testimonials
        for entry in structured_scraped_data.get('testimonials', []):
            url = entry['url']
            testimonials = entry['data']
            if testimonials:
                testimonial_text_parts = [f"Testimonials from {url}:"]
                for testimonial in testimonials:
                    if testimonial.get('text'):
                        testimonial_text_parts.append(f"  - {testimonial['text'][:300]}")
                    if testimonial.get('author'):
                        testimonial_text_parts.append(f"    Author: {testimonial['author']}")
                testimonial_text = "\n".join(testimonial_text_parts)
                structured_data_texts.append(testimonial_text)
                structured_data_sources.append({'url': url, 'type': 'testimonials'})
        
        # Create embeddings for structured data
        if embedder and structured_data_texts:
            print(f"🔄 Creating embeddings for {len(structured_data_texts)} structured entries...")
            structured_data_embeddings = embedder.encode(structured_data_texts, convert_to_tensor=True)
            save_structured_embeddings_data(structured_data_texts, structured_data_sources, structured_data_embeddings)
            print(f"✅ Created and saved structured data embeddings")
        
        scraping_complete = True
        print("✅ Web scraping completed successfully! Data stored as embeddings.")
        
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        import traceback
        traceback.print_exc()
        scraping_complete = False
    finally:
        scraping_in_progress = False

# === Text Cleaning Functions ===
def clean_text_content(text):
    """Remove common error messages, HTML noise, and unwanted content from text."""
    if not text:
        return text
    
    # List of error messages and unwanted content to remove
    error_patterns = [
        "You need to enable JavaScript to run this app",
        "Please enable JavaScript",
        "JavaScript is disabled",
        "Enable JavaScript",
    ]
    
    # Remove error messages
    cleaned = text
    for pattern in error_patterns:
        # Case-insensitive removal
        cleaned = re.sub(re.escape(pattern), "", cleaned, flags=re.IGNORECASE)
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Multiple blank lines
    cleaned = re.sub(r' +', ' ', cleaned)  # Multiple spaces
    
    return cleaned.strip()

def is_valid_content(text, min_length=50):
    """Check if text is valid content (not just error messages or too short)."""
    if not text or len(text.strip()) < min_length:
        return False
    
    # Check if text is mostly error messages
    error_keywords = ["you need to enable javascript", "enable javascript", "javascript is disabled"]
    text_lower = text.lower()
    
    # Count error keywords
    error_count = sum(1 for keyword in error_keywords if keyword in text_lower)
    
    # If more than 30% of the text contains error messages, reject it
    words = text_lower.split()
    if len(words) < 10:
        # For very short texts, be more strict
        if error_count > 0:
            return False
    
    # Remove error messages and check remaining length
    cleaned = clean_text_content(text)
    if len(cleaned.strip()) < min_length:
        return False
    
    return True

# === Load Scraped Data ===
def load_scraped_data(file_path=None):
    """Load scraped data from embeddings (preferred) or fallback to text file"""
    global scraped_data_pages, scraped_data_texts, scraped_data_embeddings
    
    # Try loading from embeddings first
    pages_data = load_embeddings_data()
    if pages_data:
        return pages_data
    
    # Fallback to text file if embeddings don't exist (for backward compatibility)
    if file_path and os.path.exists(file_path):
        print(f"⚠️ Embeddings not found, loading from text file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            if not content.strip():
                print(f"⚠️ File {file_path} is empty.")
                return []

            pages = content.split("--- ")
            url_text_pairs = []
            parse_errors = 0
            filtered_count = 0

            for i, page in enumerate(pages):
                if not page.strip():
                    continue
                try:
                    if " ---\n" in page:
                        url, text = page.strip().split(" ---\n", 1)
                    elif " ---" in page:
                        parts = page.strip().split(" ---", 1)
                        if len(parts) == 2:
                            url, text = parts[0].strip(), parts[1].strip()
                        else:
                            parse_errors += 1
                            continue
                    else:
                        parse_errors += 1
                        continue
                    
                    cleaned_text = clean_text_content(text.strip())
                    
                    if is_valid_content(cleaned_text, min_length=50):
                        url_text_pairs.append((url.strip(), cleaned_text))
                    else:
                        filtered_count += 1
                except ValueError:
                    parse_errors += 1
                    continue
            
            if not url_text_pairs:
                return []
            
            # Create embeddings and save them
            if embedder and url_text_pairs:
                texts = [text for _, text in url_text_pairs]
                embeddings = embedder.encode(texts, convert_to_tensor=True)
                save_embeddings_data(url_text_pairs, embeddings)
                
                scraped_data_pages = url_text_pairs
                scraped_data_texts = texts
                scraped_data_embeddings = embeddings
                
                print(f"✅ Converted text file to embeddings format")
                return url_text_pairs
            
            return []
        except Exception as e:
            print(f"❌ Error loading from text file: {e}")
            return []
    
    return []

# === Load Structured Data ===
def load_structured_data(file_path=None):
    """Load structured data from embeddings (preferred) or fallback to text file"""
    global structured_data_texts, structured_data_embeddings, structured_data_sources
    
    # Try loading from embeddings first
    texts = load_structured_embeddings_data()
    if texts:
        return texts
    
    # Fallback to text file if embeddings don't exist (for backward compatibility)
    if file_path and os.path.exists(file_path):
        print(f"⚠️ Structured embeddings not found, loading from text file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            if not content.strip():
                return []
            
            structured_data_texts = []
            structured_data_sources = []
            
            # Parse contact information section
            if "--- Contact Information ---" in content:
                contact_section = content.split("--- Contact Information ---")[1]
                if "--- Job Opportunities ---" in contact_section:
                    contact_section = contact_section.split("--- Job Opportunities ---")[0]
                
                entries = contact_section.split("URL:")
                for entry in entries:
                    if not entry.strip():
                        continue
                    
                    lines = entry.strip().split('\n')
                    if len(lines) < 2:
                        continue
                    
                    url = lines[0].strip()
                    contact_text_parts = []
                    
                    for line in lines[1:]:
                        line = line.strip()
                        if line and not line.startswith('---'):
                            contact_text_parts.append(line)
                    
                    if contact_text_parts:
                        contact_text = f"Contact Information from {url}:\n" + "\n".join(contact_text_parts)
                        structured_data_texts.append(contact_text)
                        structured_data_sources.append({
                            'url': url,
                            'type': 'contact_info'
                        })
            
            # Parse job opportunities section
            if "--- Job Opportunities ---" in content:
                jobs_section = content.split("--- Job Opportunities ---")[1]
                entries = jobs_section.split("URL:")
                for entry in entries:
                    if not entry.strip():
                        continue
                    
                    lines = entry.strip().split('\n')
                    if len(lines) < 2:
                        continue
                    
                    url = lines[0].strip()
                    job_text_parts = []
                    
                    for line in lines[1:]:
                        line = line.strip()
                        if line and not line.startswith('---'):
                            job_text_parts.append(line)
                    
                    if job_text_parts:
                        job_text = f"Job Opportunities from {url}:\n" + "\n".join(job_text_parts)
                        structured_data_texts.append(job_text)
                        structured_data_sources.append({
                            'url': url,
                            'type': 'job_opportunities'
                        })
            
            # Create embeddings and save them
            if embedder and structured_data_texts:
                structured_data_embeddings = embedder.encode(structured_data_texts, convert_to_tensor=True)
                save_structured_embeddings_data(structured_data_texts, structured_data_sources, structured_data_embeddings)
                print(f"✅ Converted structured text file to embeddings format")
            
            return structured_data_texts
        except Exception as e:
            print(f"❌ Error loading structured data from text file: {e}")
            return []
    
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
    if not embedder:
        return "Sentence transformer model not available. Please check your installation.", []
    
    if not scraped_data_pages:
        return "No scraped data available. Please run scraping first.", []
    
    if scraped_data_embeddings is None:
        return "Embeddings not created. Please check if scraped data was loaded correctly.", []
    
    try:
        # Check if embeddings are valid
        if hasattr(scraped_data_embeddings, 'shape') and scraped_data_embeddings.shape[0] == 0:
            return "Embeddings are empty. Please reload the scraped data.", []
        
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
        import traceback
        traceback.print_exc()
        return f"Error processing request: {e}", []

# === Enhanced Semantic Retrieval ===
def retrieve_relevant_chunks(user_input, top_k=3, similarity_threshold=0.0):
    if not embedder:
        print("❌ Sentence transformer not available")
        return []

    all_chunks = []
    all_embeddings = []
    all_sources = []
    
    # Add scraped data
    has_scraped_embeddings = False
    if scraped_data_embeddings is not None:
        try:
            # Check if embeddings exist and are valid
            if hasattr(scraped_data_embeddings, 'shape') and len(scraped_data_embeddings.shape) > 0:
                if scraped_data_embeddings.shape[0] > 0:
                    has_scraped_embeddings = True
            elif hasattr(scraped_data_embeddings, '__len__') and len(scraped_data_embeddings) > 0:
                has_scraped_embeddings = True
        except:
            pass
    
    if has_scraped_embeddings and len(scraped_data_pages) > 0:
        all_chunks.extend(scraped_data_pages)
        all_embeddings.append(scraped_data_embeddings)
        all_sources.extend(['scraped'] * len(scraped_data_pages))
        print(f"📚 Added {len(scraped_data_pages)} scraped pages to retrieval pool")
    else:
        print(f"⚠️ Scraped data not available: pages={len(scraped_data_pages)}, embeddings={'None' if scraped_data_embeddings is None else 'empty'}")
    
    # Add structured data (contact info, addresses, etc.)
    has_structured_embeddings = False
    if structured_data_embeddings is not None:
        try:
            if hasattr(structured_data_embeddings, 'shape') and len(structured_data_embeddings.shape) > 0:
                if structured_data_embeddings.shape[0] > 0:
                    has_structured_embeddings = True
            elif hasattr(structured_data_embeddings, '__len__') and len(structured_data_embeddings) > 0:
                has_structured_embeddings = True
        except:
            pass
    
    if has_structured_embeddings and len(structured_data_texts) > 0:
        all_chunks.extend([(f"Structured: {source['url']}", text) for text, source in zip(structured_data_texts, structured_data_sources)])
        all_embeddings.append(structured_data_embeddings)
        all_sources.extend(['structured'] * len(structured_data_texts))
        print(f"📋 Added {len(structured_data_texts)} structured entries to retrieval pool")
    
    # Add custom data
    has_custom_embeddings = False
    if custom_data_embeddings is not None:
        try:
            if hasattr(custom_data_embeddings, 'shape') and len(custom_data_embeddings.shape) > 0:
                if custom_data_embeddings.shape[0] > 0:
                    has_custom_embeddings = True
            elif hasattr(custom_data_embeddings, '__len__') and len(custom_data_embeddings) > 0:
                has_custom_embeddings = True
        except:
            pass
    
    if has_custom_embeddings and len(custom_data_texts) > 0:
        all_chunks.extend([(f"Custom: {source['title']}", text) for text, source in zip(custom_data_texts, custom_data_sources)])
        all_embeddings.append(custom_data_embeddings)
        all_sources.extend(['custom'] * len(custom_data_texts))
        print(f"📝 Added {len(custom_data_texts)} custom entries to retrieval pool")
    
    if not all_chunks:
        print("❌ No data available for retrieval - check if embeddings exist or run scraping first")
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
        
        # Detect query type for boosting
        is_address_query_retrieval = any(keyword in user_input.lower() for keyword in ['address', 'location', 'where', 'office', 'contact', 'located'])
        is_about_query_retrieval = any(keyword in user_input.lower() for keyword in ['tell me about', 'what is', 'describe', 'who is', 'about the company', 'about this company', 
                                                                                     'company information', 'company overview', 'company background', 'what does the company do',
                                                                                     'what is the company', 'company details', 'about us', 'company story', 'company history'])
        
        # For address queries, boost chunks that contain address-related keywords
        address_keywords = ['address', 'location', 'thrikkakara', 'kakkanad', 'kochi', 'dubai', 'uae', 'office', 'building', 'street', 'postal', '682021']
        if is_address_query_retrieval:
            # Boost similarity scores for chunks containing address keywords
            for i in range(len(all_chunks)):
                chunk_text = all_chunks[i][1].lower() if isinstance(all_chunks[i], tuple) else str(all_chunks[i]).lower()
                if any(keyword in chunk_text for keyword in address_keywords):
                    # Boost similarity by 0.2 for address-related chunks
                    similarities[i] = similarities[i] + 0.2
                    print(f"Boosted chunk {i} (contains address keywords)")
        
        # For product/service queries, boost chunks that contain product/service-related keywords
        is_product_query_retrieval = any(keyword in user_input.lower() for keyword in ['product', 'service', 'offering', 'what do you', 'what does', 'what can', 'offer', 'provide', 'sell', 'solution'])
        product_keywords_retrieval = ['product', 'service', 'offering', 'solution', 'app', 'software', 'platform', 'tool', 
                                     'what we do', 'our services', 'our products', 'services we offer', 'what we offer']
        if is_product_query_retrieval:
            # Boost similarity scores for chunks containing product/service keywords
            for i in range(len(all_chunks)):
                chunk_text = all_chunks[i][1].lower() if isinstance(all_chunks[i], tuple) else str(all_chunks[i]).lower()
                # Check URL and text for product/service-related content
                chunk_url = all_chunks[i][0].lower() if isinstance(all_chunks[i], tuple) else ""
                
                # Strong boost for product/service URLs
                url_boost = 0.0
                if any(keyword in chunk_url for keyword in ['product', 'service', 'solution', 'offering', 'services', 'products']):
                    url_boost = 0.4  # Very strong boost for product/service pages
                
                # Boost for product/service content
                content_boost = 0.0
                if any(keyword in chunk_text for keyword in product_keywords_retrieval):
                    content_boost = 0.3
                
                # Penalize contact/privacy/terms pages for product queries
                penalty = 0.0
                if any(keyword in chunk_url for keyword in ['contact', 'privacy', 'terms', 'policy', 'legal']):
                    penalty = -0.2  # Reduce priority of non-product pages
                
                total_boost = url_boost + content_boost + penalty
                if total_boost != 0:
                    similarities[i] = similarities[i] + total_boost
                    if total_boost > 0:
                        print(f"Boosted chunk {i} (product/service content, boost: {total_boost:.2f})")
                    else:
                        print(f"Penalized chunk {i} (non-product page, penalty: {penalty:.2f})")
        
        # For about queries, boost chunks that contain company/about-related keywords
        about_keywords = ['about', 'company', 'mission', 'vision', 'history', 'story', 'culture', 'values', 'team', 
                         'who we are', 'what we do', 'overview', 'background', 'overview', 'company information',
                         'company overview', 'our story', 'our mission', 'our vision']
        if is_about_query_retrieval:
            # Boost similarity scores for chunks containing about/company keywords
            for i in range(len(all_chunks)):
                chunk_text = all_chunks[i][1].lower() if isinstance(all_chunks[i], tuple) else str(all_chunks[i]).lower()
                # Check URL and text for about-related content
                chunk_url = all_chunks[i][0].lower() if isinstance(all_chunks[i], tuple) else ""
                if any(keyword in chunk_text for keyword in about_keywords) or any(keyword in chunk_url for keyword in ['about', 'company', 'story', 'mission', 'vision']):
                    # Boost similarity by 0.3 for about-related chunks (higher than address)
                    similarities[i] = similarities[i] + 0.3
                    print(f"Boosted chunk {i} (contains company/about keywords)")
        
        # Get top k results - always return top-k even if below threshold
        # This ensures we always have context to work with
        # Get more candidates to filter (increased multiplier for better coverage)
        top_k_indices = similarities.argsort(descending=True)[:top_k * 3]  # Increased from 2 to 3 for better coverage
        
        # For product queries, prioritize product/service pages
        if is_product_query_retrieval:
            # Separate product pages from non-product pages
            product_page_indices = []
            other_page_indices = []
            
            for i in top_k_indices:
                chunk = all_chunks[i]
                chunk_url = chunk[0].lower() if isinstance(chunk, tuple) else str(chunk).lower()
                
                # Check if it's a product/service page
                is_product_page = any(keyword in chunk_url for keyword in ['product', 'service', 'solution', 'offering', 'services', 'products'])
                is_non_product_page = any(keyword in chunk_url for keyword in ['contact', 'privacy', 'terms', 'policy', 'legal'])
                
                if is_product_page:
                    product_page_indices.append(i)
                elif not is_non_product_page:  # Only add non-product pages if they're not contact/privacy
                    other_page_indices.append(i)
            
            # Prioritize product pages first, then others
            prioritized_indices = product_page_indices + other_page_indices
            if len(prioritized_indices) < len(top_k_indices):
                # Add remaining indices
                remaining = [i for i in top_k_indices if i not in prioritized_indices]
                prioritized_indices.extend(remaining)
            
            top_k_indices = prioritized_indices[:top_k * 3]
        
        results = []
        
        for i in top_k_indices:
            similarity_score = similarities[i].item()
            chunk = all_chunks[i]
            
            # Check if chunk is valid (not mostly error messages)
            chunk_text = chunk[1] if isinstance(chunk, tuple) else str(chunk)
            cleaned_chunk = clean_text_content(chunk_text)
            
            # Be more lenient with content validation - only skip if really invalid
            if not is_valid_content(cleaned_chunk, min_length=20):  # Lowered from 30 to 20
                # Skip chunks that are mostly error messages
                continue
            
            results.append(chunk)
            source_type = all_sources[i]
            if similarity_score >= similarity_threshold:
                print(f"Selected {source_type} chunk {i} with similarity {similarity_score:.3f}")
            else:
                print(f"Selected {source_type} chunk {i} with similarity {similarity_score:.3f} (below threshold {similarity_threshold}, but included as top-k)")
            
            # Stop once we have enough valid chunks
            if len(results) >= top_k:
                break
        
        # If we don't have enough results, add more even with lower similarity
        if len(results) < top_k and len(top_k_indices) > len(results):
            print(f"⚠️ Only found {len(results)} valid chunks, adding more with lower similarity...")
            for i in top_k_indices[len(results):]:
                if len(results) >= top_k:
                    break
                chunk = all_chunks[i]
                chunk_text = chunk[1] if isinstance(chunk, tuple) else str(chunk)
                cleaned_chunk = clean_text_content(chunk_text)
                if len(cleaned_chunk.strip()) > 20:  # Very lenient check
                    results.append(chunk)
                    print(f"Added additional chunk {i} to reach {len(results)} total")
        
        total_chunks = len(scraped_data_pages) + len(structured_data_texts) + len(custom_data_texts)
        print(f"Retrieved {len(results)} relevant chunks out of {total_chunks} total (after filtering error messages)")
        return results
        
    except Exception as e:
        print(f"Error in semantic retrieval: {e}")
        return []

# === Enhanced OpenAI Response ===
def generate_openai_response(user_input, relevant_chunks, is_about_query=False, is_address_query=False, is_product_query=False):
    if not OPENAI_API_KEY:
        return "OpenAI API key not configured. Please set your OPENAI_API_KEY environment variable.", []

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Check if query is about address, location, or contact info
        address_keywords = ['address', 'location', 'where', 'office', 'contact', 'phone', 'email', 'located']
        is_address_query = any(keyword in user_input.lower() for keyword in address_keywords)
        
        # Check if query is about products, services, or offerings
        product_keywords = ['product', 'service', 'offering', 'what do you', 'what does', 'what can', 'offer', 'provide', 'sell']
        is_product_query = any(keyword in user_input.lower() for keyword in product_keywords)
        
        # Check if query is asking about the company (tell me about, what is, describe, etc.)
        about_keywords = ['tell me about', 'what is', 'describe', 'who is', 'about the company', 'about this company', 
                          'company information', 'company overview', 'company background', 'what does the company do',
                          'what is the company', 'company details', 'about us', 'company story', 'company history']
        is_about_query = any(keyword in user_input.lower() for keyword in about_keywords)
        
        if relevant_chunks:
            context_str = "\n\n--- Context ---\n"
            source_urls = set()
            
            # Determine if this is a generic/greeting query
            is_generic_query = user_input.lower().strip() in ['hi', 'hello', 'hey', 'greetings', 'what can you do', 'help', '?']
            
            # For address queries, product queries, about queries, or generic queries, use more context
            if is_about_query:
                text_limit = 5000  # Use maximum context for company/about queries
            elif is_address_query:
                text_limit = 3000
            elif is_product_query:
                text_limit = 3000  # Use more context for product queries
            elif is_generic_query:
                text_limit = 2000
            else:
                text_limit = 1500  # Increased default from 1000
            
            valid_chunks_used = 0
            for i, (url, text) in enumerate(relevant_chunks):
                # Clean the text to remove error messages
                cleaned_text = clean_text_content(text)
                
                # Skip chunks that are mostly error messages or invalid
                if not is_valid_content(cleaned_text, min_length=30):
                    print(f"    ⚠️ Skipping chunk {i+1} - contains mostly error messages or invalid content")
                    continue
                
                # Handle both scraped URLs and custom data titles
                if url.startswith('Custom:'):
                    context_str += f"Custom Data {valid_chunks_used+1} ({url}):\n{cleaned_text[:text_limit]}...\n\n"
                    source_urls.add(url)
                    valid_chunks_used += 1
                elif url.startswith('Structured:'):
                    # For structured data, include full text (it's usually short)
                    context_str += f"Structured Data {valid_chunks_used+1} ({url}):\n{cleaned_text}\n\n"
                    source_urls.add(url)
                    valid_chunks_used += 1
                else:
                    # For scraped data, use larger context for address queries or product queries
                    if is_address_query:
                        # Check if text contains address keywords - if so, include more context
                        address_keywords_in_text = ['thrikkakara', 'kakkanad', 'kochi', 'dubai', 'uae', 'address:', '682021', 'sheikh rashid', 'opp. bmc']
                        if any(keyword in cleaned_text.lower() for keyword in address_keywords_in_text):
                            # Include full text or at least 3000 chars for address-containing chunks
                            context_str += f"Source {valid_chunks_used+1} ({url}) [CONTAINS ADDRESS INFO]:\n{cleaned_text[:3000]}...\n\n"
                        else:
                            context_str += f"Source {valid_chunks_used+1} ({url}):\n{cleaned_text[:text_limit]}...\n\n"
                    elif is_about_query:
                        # For about queries, prioritize main content and include maximum context
                        about_keywords_in_text = ['about', 'company', 'mission', 'vision', 'history', 'story', 'culture', 
                                                  'values', 'team', 'who we are', 'what we do', 'overview', 'background']
                        if any(keyword in cleaned_text.lower() for keyword in about_keywords_in_text):
                            # Include full or maximum context for about-related chunks
                            context_str += f"Source {valid_chunks_used+1} ({url}) [COMPANY INFO - HIGH PRIORITY]:\n{cleaned_text[:6000]}...\n\n"
                        else:
                            # For about queries, still use more context even if not explicitly about-related
                            context_str += f"Source {valid_chunks_used+1} ({url}):\n{cleaned_text[:text_limit]}...\n\n"
                    elif is_product_query:
                        # For product queries, prioritize product/service pages and include maximum context
                        product_keywords_in_text = ['product', 'service', 'offering', 'solution', 'app', 'software', 'platform', 'tool',
                                                    'what we do', 'our services', 'our products', 'services we offer', 'what we offer',
                                                    'mobile app', 'web development', 'digital marketing', 'software development']
                        product_keywords_in_url = ['product', 'service', 'solution', 'offering', 'services', 'products']
                        
                        # Check if URL is a product/service page
                        is_product_page = any(keyword in url.lower() for keyword in product_keywords_in_url)
                        has_product_content = any(keyword in cleaned_text.lower() for keyword in product_keywords_in_text)
                        
                        if is_product_page:
                            # Maximum context for product/service pages
                            context_str += f"Source {valid_chunks_used+1} ({url}) [PRODUCT/SERVICE PAGE - HIGHEST PRIORITY]:\n{cleaned_text[:5000]}...\n\n"
                        elif has_product_content:
                            # High context for pages with product content
                            context_str += f"Source {valid_chunks_used+1} ({url}) [CONTAINS PRODUCT INFO]:\n{cleaned_text[:4000]}...\n\n"
                        else:
                            # Regular context for other pages
                            context_str += f"Source {valid_chunks_used+1} ({url}):\n{cleaned_text[:text_limit]}...\n\n"
                    else:
                        context_str += f"Source {valid_chunks_used+1} ({url}):\n{cleaned_text[:text_limit]}...\n\n"
                    source_urls.add(url)
                    valid_chunks_used += 1
                
                # Limit number of chunks to avoid token limits - but be more generous for about and product queries
                max_chunks = 15 if is_about_query else 12 if is_product_query else 10 if is_address_query else 8
                if valid_chunks_used >= max_chunks:
                    break
            
            context_str += "--- End Context ---"
            print(f"Context length: {len(context_str)} characters")
        else:
            context_str = "No relevant content found."
            source_urls = set()
            print("⚠️ No relevant chunks found for context")

        # Enhanced prompt that mentions custom data and emphasizes finding addresses or products
        special_instruction = ""
        if is_address_query:
            special_instruction = """
CRITICAL: The user is asking about address or location. You MUST:
1. Search through EVERY piece of context provided below
2. Look for ANY mention of:
   - Street addresses, building names/numbers (e.g., "Opp. BMC", "Sheikh Rashid Building", "R133")
   - Cities, states, countries (e.g., "Thrikkakara", "Kakkanad", "Kochi", "Kerala", "Dubai", "UAE")
   - Postal codes (e.g., "682021")
   - Phrases like "Address:", "located at", "office at"
3. Extract and present ALL address information found, even if incomplete
4. If multiple locations exist (India office, UAE office), list BOTH
5. DO NOT say "I don't have that information" if you see ANY location-related text in the context below
6. Even partial addresses are valuable - include them

The context below contains the information. Read it carefully and extract all address details.
"""
        elif is_about_query:
            special_instruction = """
CRITICAL: The user is asking about the company itself. You MUST provide a COMPREHENSIVE company description. You MUST:
1. Search through EVERY piece of context provided below
2. Extract and present ALL of the following information if available:
   - Company name and what the company does
   - Company mission, vision, values, and philosophy
   - Company history, background, and story
   - Services and products offered
   - Company culture, work environment, team
   - Key achievements, milestones, or notable facts
   - Industries served or target markets
   - Company size, locations, or presence
3. Structure your response as a comprehensive company overview, NOT just contact information
4. Prioritize main content pages (homepage, about, company pages) over contact pages
5. DO NOT just list contact information - provide a full company description
6. If contact info is relevant, include it at the end, but focus on company description first
7. Be detailed and comprehensive - this is a company overview query

The context below contains the information. Read it carefully and provide a complete company description.
"""
        elif is_product_query:
            special_instruction = """
CRITICAL: The user is asking about products, services, or offerings. You MUST:
1. Search through EVERY piece of context provided below - prioritize product/service pages over contact/privacy pages
2. Look for ANY mention of:
   - Product names, service names, offerings, solutions
   - Features, capabilities, solutions, applications
   - Applications, platforms, tools, software, apps
   - Services provided, what the company does, what they offer
   - Service categories, product categories, service lists
3. Extract and present ALL product/service information found, even if incomplete
4. List ALL products/services mentioned in the context - be comprehensive
5. DO NOT just mention one service - list ALL services/products found
6. DO NOT say "I don't have that information" if you see ANY product/service-related text in the context below
7. Be specific and detailed - include product names, descriptions, features, and capabilities
8. If you see product/service pages in the context, prioritize information from those pages
9. Ignore contact information, privacy policies, and terms pages unless they contain product/service details

The context below contains the information. Read it carefully and extract ALL product/service details comprehensively.
"""
        
        # Handle generic greetings - extract company info for better greeting
        is_greeting = user_input.lower().strip() in ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        company_info = ""
        
        if relevant_chunks:
            # Try to extract company name or main info from first chunk (generic approach)
            first_chunk_text = relevant_chunks[0][1] if relevant_chunks else ""
            
            # Extract company name from page title or first heading
            company_name = ""
            if "PAGE TITLE:" in first_chunk_text:
                title_line = [line for line in first_chunk_text.split('\n') if 'PAGE TITLE:' in line]
                if title_line:
                    company_name = title_line[0].replace('PAGE TITLE:', '').strip()
                    # Extract just the company name (before common separators)
                    for sep in [' - ', ' | ', ' :: ', ' – ']:
                        if sep in company_name:
                            company_name = company_name.split(sep)[0].strip()
            
            # Extract key services or info (generic keywords)
            services = []
            service_keywords = {
                'mobile app development': ['mobile app', 'mobile development', 'ios', 'android', 'react native'],
                'web development': ['web development', 'web design', 'website', 'web application'],
                'digital marketing': ['digital marketing', 'seo', 'social media marketing', 'ppc'],
                'software development': ['software development', 'custom software', 'application development'],
                'consulting': ['consulting', 'consultancy', 'advisory'],
                'design': ['ui/ux', 'user interface', 'user experience', 'graphic design']
            }
            
            text_lower = first_chunk_text.lower()
            for service_name, keywords in service_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    services.append(service_name)
            
            if services and company_name:
                company_info = f" {company_name} offers {', '.join(services[:3])} and more."
            elif services:
                company_info = f" The company offers {', '.join(services[:3])} and more."
        
        # Build a much more forceful and comprehensive prompt
        prompt = f"""You are an expert assistant for a company website. Your PRIMARY job is to provide COMPREHENSIVE, DETAILED answers using ALL the information provided in the context below.

{special_instruction}

ABSOLUTE MANDATORY RULES - YOU MUST FOLLOW THESE:
1. **ALWAYS use the context provided below** - it contains REAL information from the website. DO NOT make up information.
2. **NEVER say "I don't have that information" or "I cannot find"** - the context below HAS information, you MUST extract and use it.
3. **NEVER repeat error messages** like "You need to enable JavaScript" - ignore such messages completely.
4. **Provide COMPREHENSIVE answers** - don't give short, incomplete responses. Extract ALL relevant information from the context.
5. **Be DETAILED and SPECIFIC** - include names, descriptions, features, and all relevant details from the context.
6. **If the question is about the company**, provide a FULL company description including: what they do, services, mission, values, history, culture, achievements, and any other relevant information found in the context.
7. **Extract and share ANY relevant information** from the context, even if it seems partial - it's still valuable.
8. **Be conversational and helpful** - write in a natural, engaging way.
9. **Focus on meaningful content** - ignore HTML errors, JavaScript messages, or placeholder text.
10. **If asked a greeting** (hi, hello), respond warmly and offer to help with company information{company_info}

CRITICAL: The context below contains REAL information from the website. Read it THOROUGHLY and extract EVERYTHING relevant to answer the question comprehensively.

WEBSITE CONTENT (READ THIS CAREFULLY - it contains the answer):
{context_str}

Question: {user_input}

INSTRUCTIONS FOR YOUR ANSWER:
- Read ALL the context above thoroughly
- Extract EVERY piece of relevant information
- Provide a COMPREHENSIVE, DETAILED answer
- Be specific and include all relevant details
- DO NOT give short or incomplete answers
- If the question is about the company, provide a FULL company overview

Answer (MUST be comprehensive and use ALL relevant information from the context above):"""

        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert assistant for a company website. You provide comprehensive, detailed answers using ONLY the information provided in the context. You NEVER say you don't have information - you ALWAYS extract and use what's available in the context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=3000  # Increased from 2000 to allow much longer, more comprehensive responses
        )
        
        # Extract the response
        if not response.choices or len(response.choices) == 0:
            raise Exception("No response from OpenAI API")
        
        answer = response.choices[0].message.content.strip()
        
        # Log full response for debugging
        print(f"🤖 Full OpenAI response ({len(answer)} chars): {answer}")
        
        # Clean the answer to remove any error messages that might have slipped through
        answer = clean_text_content(answer)
        
        # Check if answer contains error messages - if so, generate a better response
        error_message_patterns = ["you need to enable javascript", "enable javascript", "javascript is disabled"]
        if any(pattern in answer.lower() for pattern in error_message_patterns):
            print("⚠️ Response contains error message, generating fallback...")
            answer = "I'm having trouble processing that content. Could you please rephrase your question?"
            if relevant_chunks:
                # Try to extract meaningful content
                for url, text in relevant_chunks[:3]:
                    cleaned = clean_text_content(text)
                    if is_valid_content(cleaned, min_length=100):
                        # Use the first valid chunk as fallback
                        answer = f"Based on the available information:\n\n{cleaned[:300]}...\n\nWould you like more details?"
                        break
        
        print(f"✅ Final answer length: {len(answer)} characters")
        
        # Fallback: If OpenAI says it doesn't have info but we have chunks, extract directly
        no_info_phrases = ["i don't have", "i do not have", "i cannot find", "no information", "don't have that information", 
                          "unable to find", "i'm not able to", "i cannot provide", "i don't know", "i'm not sure"]
        if any(phrase in answer.lower() for phrase in no_info_phrases) and relevant_chunks:
            print("⚠️ OpenAI said no info, but we have chunks. Extracting comprehensive fallback response...")
            
            # Build a comprehensive answer from all chunks
            fallback_parts = []
            
            if is_about_query:
                # For about queries, extract comprehensive company information
                for url, text in relevant_chunks[:10]:  # Use up to 10 chunks
                    cleaned = clean_text_content(text)
                    if is_valid_content(cleaned, min_length=100):
                        # Extract key sections
                        if "PAGE TITLE:" in cleaned:
                            title_line = [line for line in cleaned.split('\n') if 'PAGE TITLE:' in line]
                            if title_line:
                                fallback_parts.append(f"**{title_line[0].replace('PAGE TITLE:', '').strip()}**")
                        
                        # Extract main content (skip metadata headers)
                        content_lines = []
                        skip_headers = ['PAGE TITLE:', 'PAGE DESCRIPTION:', '===', '---']
                        for line in cleaned.split('\n'):
                            if any(header in line for header in skip_headers):
                                continue
                            if len(line.strip()) > 20:  # Only meaningful lines
                                content_lines.append(line.strip())
                        
                        if content_lines:
                            fallback_parts.append('\n'.join(content_lines[:30]))  # First 30 meaningful lines
                
                if fallback_parts:
                    answer = "Based on the website content, here's what I found:\n\n" + "\n\n".join(fallback_parts[:5])
                    if len(relevant_chunks) > 5:
                        answer += f"\n\n[Information from {len(relevant_chunks)} pages]"
                else:
                    # Last resort: use first chunk
                    first_chunk_text = relevant_chunks[0][1]
                    cleaned_snippet = clean_text_content(first_chunk_text[:1000].strip())
                    answer = f"Based on the website content:\n\n{cleaned_snippet}\n\nWould you like more specific information?"
            elif is_greeting:
                # For greetings, provide a friendly intro
                answer = f"Hello! I'm here to help you learn about the company.{company_info} What would you like to know?"
            else:
                # For other queries, provide comprehensive snippets from multiple chunks
                answer_parts = []
                for url, text in relevant_chunks[:5]:
                    cleaned = clean_text_content(text[:800].strip())
                    if is_valid_content(cleaned, min_length=100):
                        answer_parts.append(f"**From {url}:**\n{cleaned}")
                
                if answer_parts:
                    answer = "Based on the website content:\n\n" + "\n\n".join(answer_parts)
                else:
                    # Last resort
                    first_chunk_text = relevant_chunks[0][1]
                    cleaned_snippet = clean_text_content(first_chunk_text[:500].strip())
                    answer = f"Based on the website content:\n\n{cleaned_snippet}\n\nWould you like more specific information?"
            
            print(f"✅ Comprehensive fallback response generated from {len(relevant_chunks)} chunks")
        
        return answer, list(source_urls)
        
    except Exception as e:
        import traceback
        print(f"OpenAI error: {e}")
        print(f"Full traceback:")
        traceback.print_exc()
        error_msg = str(e)
        if "invalid_api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            return "Invalid API key. Please check your OpenAI API key configuration.", []
        elif "permission" in error_msg.lower() or "forbidden" in error_msg.lower():
            return "Permission denied. Please check your API key permissions.", []
        elif "429" in error_msg or "rate_limit" in error_msg.lower() or "quota" in error_msg.lower():
            return "API rate limit exceeded. Please try again later or upgrade your plan.", []
        elif "context_length" in error_msg.lower() or "token" in error_msg.lower():
            return "The context is too long. Please try a more specific question or reduce the amount of data.", []
        else:
            return f"Error generating response: {error_msg}. Please check the server logs for more details.", []

# === Pydantic Models for Request/Response ===
class QuestionRequest(BaseModel):
    question: str
    chatbot_type: Optional[str] = "openai"

class WebsiteRequest(BaseModel):
    website: str
    auto_scrape: Optional[bool] = False  # Optionally start scraping automatically

class CustomDataRequest(BaseModel):
    title: str
    category: str
    content: str

class RemoveCustomDataRequest(BaseModel):
    index: int

# === Load Once ===
@app.on_event("startup")
async def load_data_once():
    global scraped_data_pages, scraped_data_texts, scraped_data_embeddings
    global custom_data_texts, custom_data_embeddings, custom_data_sources
    global structured_data_texts, structured_data_embeddings, structured_data_sources
    
    # Load scraped data from embeddings (preferred) or text file (fallback)
    if embedder:
        print("📂 Loading scraped data from embeddings...")
        data = load_scraped_data()
        if data:
            print(f"✅ Loaded {len(data)} pages with embeddings")
        else:
            # Try fallback to text file if embeddings don't exist
            if os.path.exists("scraped_data.txt"):
                print("📂 Loading scraped data from text file (fallback)...")
                data = load_scraped_data("scraped_data.txt")
            else:
                print("⚠️ No embeddings or text file found. Run scraping first.")
    else:
        print("⚠️ Sentence transformer embedder not available. Cannot load embeddings.")
    
    # Load structured data from embeddings (preferred) or text file (fallback)
    if embedder:
        print("📋 Loading structured data from embeddings...")
        load_structured_data()
        if not structured_data_texts and os.path.exists("structured_data.txt"):
            print("📋 Loading structured data from text file (fallback)...")
            load_structured_data("structured_data.txt")
    
    # Load custom data if file exists
    if embedder and os.path.exists("custom_data.txt"):
        print("📝 Loading custom data...")
        load_custom_data()
    
    # Print summary
    print(f"\n📊 Data Summary:")
    print(f"   - Scraped pages: {len(scraped_data_pages)}")
    print(f"   - Structured entries: {len(structured_data_texts)}")
    print(f"   - Custom entries: {len(custom_data_texts)}")
    embeddings_available = scraped_data_embeddings is not None and (
        (hasattr(scraped_data_embeddings, 'shape') and scraped_data_embeddings.shape[0] > 0) or
        (hasattr(scraped_data_embeddings, '__len__') and len(scraped_data_embeddings) > 0)
    )
    print(f"   - Embeddings available: {embeddings_available}\n")

# === Routes ===
@app.get('/', response_class=HTMLResponse)
async def index():
    with open('templates/index.html', 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

@app.post('/ask')
async def ask(question: str = Form(...), chatbot_type: str = Form("openai")):
    try:
        question = question.strip()
        chatbot_type = chatbot_type or 'openai'  # 'openai' or 'basic'
        
        if not question:
            return JSONResponse(content={"answer": "Please enter a question.", "source_urls": []})

        if not embedder:
            return JSONResponse(content={
                "answer": "Sentence transformer model not available. Please check your installation.",
                "source_urls": []
            })

        if chatbot_type == 'basic':
            # Use basic chatbot functionality
            answer, urls = basic_chatbot_response(question)
        else:
            # Use OpenAI chatbot
            # Increase top_k for address/location queries, product queries, about queries, and generic queries to get more context
            address_keywords = ['address', 'location', 'where', 'office', 'contact', 'phone', 'email', 'located']
            product_keywords = ['product', 'service', 'offering', 'what do you', 'what does', 'what can', 'offer', 'provide', 'sell']
            about_keywords = ['tell me about', 'what is', 'describe', 'who is', 'about the company', 'about this company', 
                              'company information', 'company overview', 'company background', 'what does the company do',
                              'what is the company', 'company details', 'about us', 'company story', 'company history']
            is_address_query = any(keyword in question.lower() for keyword in address_keywords)
            is_product_query = any(keyword in question.lower() for keyword in product_keywords)
            is_about_query = any(keyword in question.lower() for keyword in about_keywords)
            is_generic_query = question.lower().strip() in ['hi', 'hello', 'hey', 'greetings', 'what can you do', 'help', '?']
            
            # Use more chunks for address queries, product queries, about queries, or generic queries
            if is_about_query:
                top_k = 15  # Maximum chunks for comprehensive company description
            elif is_address_query or is_product_query:
                top_k = 12
            elif is_generic_query:
                top_k = 10
            else:
                top_k = 8  # Increased default from 5 to 8
            
            relevant_chunks = retrieve_relevant_chunks(question, top_k=top_k, similarity_threshold=0.0)
            # Pass query type flags to response generation
            answer, urls = generate_openai_response(question, relevant_chunks, 
                                                    is_about_query=is_about_query,
                                                    is_address_query=is_address_query,
                                                    is_product_query=is_product_query)

        return JSONResponse(content={"answer": answer, "source_urls": urls})

    except Exception as e:
        print(f"Server error: {e}")
        return JSONResponse(content={"answer": f"Internal server error: {e}", "source_urls": []})

# === Scraping Status Route ===
@app.get('/scraping_status')
async def get_scraping_status():
    return JSONResponse(content={
        "scraping_in_progress": scraping_in_progress,
        "scraping_complete": scraping_complete,
        "data_loaded": len(scraped_data_pages) > 0,
        "total_pages": len(scraped_data_pages),
        "target_website": target_website
    })

# === Manual Scraping Route ===
@app.post('/start_scraping')
async def start_scraping():
    if scraping_in_progress:
        return JSONResponse(content={"status": "already_running", "message": "Scraping is already in progress"})
    
    # Start scraping in a separate thread
    thread = threading.Thread(target=run_scraper)
    thread.daemon = True
    thread.start()
    
    return JSONResponse(content={"status": "started", "message": "Scraping started successfully"})

# === Set Website URL Route ===
@app.post('/set_website')
async def set_website(request_data: WebsiteRequest):
    global target_website, scraped_data_pages, scraped_data_texts, scraped_data_embeddings
    global scraping_complete, structured_data_texts, structured_data_embeddings, structured_data_sources
    
    try:
        new_website = request_data.website.strip()
        
        if not new_website:
            return JSONResponse(content={"status": "error", "message": "Website URL is required"})
        
        # Basic URL validation
        if not new_website.startswith(('http://', 'https://')):
            new_website = 'https://' + new_website
        
        # If website changed, clear old data
        if target_website != new_website:
            print(f"⚠️ Website changed from {target_website} to {new_website}. Clearing old data...")
            scraped_data_pages = []
            scraped_data_texts = []
            scraped_data_embeddings = None
            scraping_complete = False
            structured_data_texts = []
            structured_data_embeddings = None
            structured_data_sources = []
            print("✅ Old data cleared")
        
        target_website = new_website
        print(f"✅ Target website updated to: {target_website}")
        
        response_data = {
            "status": "success", 
            "message": f"Website updated to {target_website}",
            "website": target_website
        }
        
        # Optionally start scraping automatically if auto_scrape is True
        if request_data.auto_scrape:
            if scraping_in_progress:
                response_data["message"] = f"Website updated to {target_website}. Scraping already in progress."
                response_data["scraping_status"] = "already_running"
            else:
                # Start scraping in a separate thread
                thread = threading.Thread(target=run_scraper)
                thread.daemon = True
                thread.start()
                response_data["message"] = f"Website updated to {target_website}. Scraping started automatically."
                response_data["scraping_status"] = "started"
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": f"Error setting website: {str(e)}"})

# === Get Current Website Route ===
@app.get('/get_website')
async def get_website():
    return JSONResponse(content={
        "website": target_website,
        "status": "success"
    })

# === Reload Scraped Data Route ===
@app.post('/reload_data')
async def reload_data():
    """Manually reload scraped data from embeddings or text file"""
    global scraped_data_pages, scraped_data_texts, scraped_data_embeddings
    
    try:
        if not embedder:
            return JSONResponse(content={
                "status": "error",
                "message": "Sentence transformer embedder not available"
            })
        
        print("🔄 Manually reloading scraped data from embeddings...")
        data = load_scraped_data()
        
        if data:
            return JSONResponse(content={
                "status": "success",
                "message": f"Successfully reloaded {len(data)} pages from embeddings",
                "total_pages": len(data),
                "source": "embeddings"
            })
        else:
            # Try fallback to text file
            if os.path.exists("scraped_data.txt"):
                print("🔄 Trying to reload from text file (fallback)...")
                data = load_scraped_data("scraped_data.txt")
                if data:
                    return JSONResponse(content={
                        "status": "success",
                        "message": f"Successfully reloaded {len(data)} pages from text file (converted to embeddings)",
                        "total_pages": len(data),
                        "source": "text_file"
                    })
            
            return JSONResponse(content={
                "status": "error",
                "message": "No embeddings or text file found. Please run scraping first."
            })
        
    except Exception as e:
        print(f"❌ Error reloading data: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(content={
            "status": "error",
            "message": f"Error reloading data: {str(e)}"
        })

# === Reset Data Route ===
@app.post('/reset_data')
async def reset_data():
    global scraped_data_pages, scraped_data_texts, scraped_data_embeddings, scraping_complete
    
    try:
        # Clear all scraped data
        scraped_data_pages = []
        scraped_data_texts = []
        scraped_data_embeddings = None
        scraping_complete = False
        
        print(" All scraped data has been reset")
        
        return JSONResponse(content={
            "status": "success",
            "message": "All scraped data has been reset"
        })
        
    except Exception as e:
        return JSONResponse(content={
            "status": "error", 
            "message": f"Error resetting data: {str(e)}"
        })

# === Reset All Data Route ===
@app.post('/reset_all_data')
async def reset_all_data():
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
        
        print(" All data (scraped and custom) has been reset")
        
        return JSONResponse(content={
            "status": "success",
            "message": "All data has been reset"
        })
        
    except Exception as e:
        return JSONResponse(content={
            "status": "error", 
            "message": f"Error resetting all data: {str(e)}"
        })

# === Custom Data Routes ===
@app.post('/add_custom_data')
async def add_custom_data_route(request_data: CustomDataRequest):
    try:
        title = request_data.title.strip()
        category = request_data.category.strip()
        content = request_data.content.strip()
        
        if not title or not category or not content:
            return JSONResponse(content={
                "status": "error",
                "message": "All fields (title, category, content) are required"
            })
        
        success, message = add_custom_data(title, category, content)
        
        if success:
            # Save to file
            save_custom_data()
            return JSONResponse(content={
                "status": "success",
                "message": message,
                "total_entries": len(custom_data_texts)
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

@app.get('/get_custom_data')
async def get_custom_data_route():
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
        
        success, message = remove_custom_data(index)
        
        if success:
            # Save to file
            save_custom_data()
            return JSONResponse(content={
                "status": "success",
                "message": message,
                "total_entries": len(custom_data_texts)
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
        "total_custom_entries": len(custom_data_texts),
        "total_scraped_pages": len(scraped_data_pages),
        "custom_data_available": len(custom_data_texts) > 0,
        "scraped_data_available": len(scraped_data_pages) > 0
    })

# === Health Check Route ===
@app.get('/health')
async def health_check():
    status = {
        "api_key_configured": bool(OPENAI_API_KEY),
        "api_key_preview": OPENAI_API_KEY[:10] + "..." if OPENAI_API_KEY else None,
        "model_name": OPENAI_MODEL_NAME,
        "embedder_loaded": bool(embedder),
        "data_loaded": len(scraped_data_pages) > 0,
        "embeddings_created": scraped_data_embeddings is not None and scraped_data_embeddings.any(),
        "scraping_status": {
            "in_progress": scraping_in_progress,
            "complete": scraping_complete
        }
    }
    return JSONResponse(content=status)

# === Enhanced Debug Route ===
@app.get('/debug')
async def debug_info():
    return JSONResponse(content={
        "total_scraped_pages": len(scraped_data_pages),
        "total_custom_entries": len(custom_data_texts),
        "sample_scraped_pages": [{"url": url, "text_preview": text[:200]} for url, text in scraped_data_pages[:3]],
        "sample_custom_entries": [{"title": source['title'], "category": source['category'], "text_preview": text[:200]} for text, source in zip(custom_data_texts[:3], custom_data_sources[:3])],
        "scraped_embeddings_shape": scraped_data_embeddings.shape if scraped_data_embeddings is not None else None,
        "custom_embeddings_shape": custom_data_embeddings.shape if custom_data_embeddings is not None else None,
        "api_key_configured": bool(OPENAI_API_KEY),
        "embedder_loaded": bool(embedder),
        "scraping_status": {
            "in_progress": scraping_in_progress,
            "complete": scraping_complete
        }
    })

# === Test Query Route ===
@app.post('/test_query')
async def test_query(request_data: QuestionRequest):
    try:
        question = request_data.question.strip()
        if not question:
            return JSONResponse(content={"error": "Please enter a question."})

        # Test semantic retrieval
        relevant_chunks = retrieve_relevant_chunks(question, top_k=5, similarity_threshold=0.05)
        
        return JSONResponse(content={
            "question": question,
            "relevant_chunks_count": len(relevant_chunks),
            "relevant_chunks": [{"url": url, "text_preview": text[:300]} for url, text in relevant_chunks],
            "total_pages_available": len(scraped_data_pages)
        })
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

# === API Key Test Route ===
@app.get('/test_api')
async def test_api_key():
    if not OPENAI_API_KEY:
        return JSONResponse(content={"error": "No API key configured"})
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[
                {"role": "user", "content": "Hello, this is a test."}
            ],
            max_tokens=50
        )
        return JSONResponse(content={
            "status": "success",
            "api_key_preview": OPENAI_API_KEY[:10] + "...",
            "model": OPENAI_MODEL_NAME,
            "response": response.choices[0].message.content,
            "message": "API key is working"
        })
    except Exception as e:
        error_msg = str(e)
        return JSONResponse(content={
            "status": "error",
            "api_key_preview": OPENAI_API_KEY[:10] + "..." if OPENAI_API_KEY else None,
            "error": error_msg,
            "is_quota_error": "quota" in error_msg.lower() or "429" in error_msg or "rate_limit" in error_msg.lower()
        })

# === Project Info Route ===
@app.get('/project_info')
async def get_project_info():
    if not OPENAI_API_KEY:
        return JSONResponse(content={"error": "No API key configured"})
    
    try:
        return JSONResponse(content={
            "api_key_preview": OPENAI_API_KEY[:10] + "...",
            "model": OPENAI_MODEL_NAME,
            "note": "OpenAI API key is configured",
            "action_required": "Check your OpenAI account for usage limits and billing information"
        })
    except Exception as e:
        return JSONResponse(content={
            "error": str(e),
            "api_key_preview": OPENAI_API_KEY[:10] + "..." if OPENAI_API_KEY else None
        })

# === Run App ===
if __name__ == '__main__':
    import uvicorn
    
    print("\n=== Integrated Web Chatbot System ===")
    print(f"API Key Configured: {'✅' if OPENAI_API_KEY else '❌'}")
    print(f"Model: {OPENAI_MODEL_NAME if OPENAI_API_KEY else 'N/A'}")
    print(f"Embedder Loaded: {'✅' if embedder else '❌'}")
    print("================================\n")
    
    if not OPENAI_API_KEY:
        print("⚠️ WARNING: App will run but won't be able to generate OpenAI responses without API key")
        print("💡 Basic chatbot will still work with scraped data")
    else:
        print("💡 Note: Check your OpenAI account for usage limits and billing information.")
    
    # Load data on startup (backup in case startup event doesn't fire)
    if embedder:
        print("📂 Loading data on startup...")
        load_scraped_data()  # Will try embeddings first, then text file
        load_structured_data()  # Will try embeddings first, then text file
        load_custom_data()
    
    # Start scraping automatically in background (only if data doesn't exist AND target_website is set)
    if target_website and (not scraped_data_pages or len(scraped_data_pages) == 0):
        print(f"Starting automatic web scraping for {target_website} (no existing data found)...")
        scraping_thread = threading.Thread(target=run_scraper)
        scraping_thread.daemon = True
        scraping_thread.start()
    elif scraped_data_pages and len(scraped_data_pages) > 0:
        print(f"✅ Using existing scraped data ({len(scraped_data_pages)} pages)")
    else:
        print("⚠️ No target website set. Use /set_website endpoint to configure a website.")
    
    print("Starting FastAPI web server...")
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")