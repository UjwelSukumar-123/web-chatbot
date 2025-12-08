"""
Chatbot Module
Handles all chatbot logic including semantic retrieval and response generation
"""

import re
import numpy as np
import random
from sentence_transformers import SentenceTransformer, util
import openai
from typing import List, Tuple, Optional, Dict

# Try to import torch, fallback to numpy if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


def estimate_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Estimate the number of tokens in a text string.
    Uses tiktoken if available for accuracy, otherwise uses a simple approximation.
    """
    if TIKTOKEN_AVAILABLE:
        try:
            # Try to get encoding for the model
            encoding_name = "cl100k_base"  # Default for gpt-3.5-turbo and gpt-4
            if "gpt-4" in model_name.lower():
                encoding_name = "cl100k_base"
            elif "gpt-3.5" in model_name.lower():
                encoding_name = "cl100k_base"
            
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except Exception:
            # Fallback to approximation
            pass
    
    # Simple approximation: ~4 characters per token (conservative estimate)
    return len(text) // 4


def get_model_context_limit(model_name: str) -> int:
    """
    Get the maximum context length for a given OpenAI model.
    Returns a conservative limit (leaving some buffer for system messages and response).
    """
    model_lower = model_name.lower()
    
    # Model context limits (leaving ~2000 token buffer for system message, prompt template, and response)
    if "gpt-4-turbo" in model_lower or "gpt-4-1106" in model_lower or "gpt-4o" in model_lower:
        return 120000  # 128k - 8k buffer
    elif "gpt-4" in model_lower:
        if "32k" in model_lower:
            return 30000  # 32k - 2k buffer
        elif "16k" in model_lower:
            return 14000  # 16k - 2k buffer
        else:
            return 6000  # 8k - 2k buffer
    elif "gpt-3.5-turbo" in model_lower:
        if "16k" in model_lower:
            return 14000  # 16k - 2k buffer (actual limit is 16385, so 14k is safe)
        else:
            return 12000  # 4k - 2k buffer (conservative for older models)
    else:
        # Default conservative limit for unknown models (assume 16k model based on error message)
        # The error showed 16385 limit, so default to 14k to be safe
        return 14000


def get_greeting_response(company_info="", user_name=""):
    """Return a random greeting response with variety, optionally personalized with user name"""
    name_part = f", {user_name}!" if user_name else "!"
    
    greetings = [
        f"Hello{name_part} I'm here to help you.{company_info} What would you like to know?",
        f"Hi there{name_part} Welcome!{company_info} How can I assist you today?",
        f"Hey{name_part} Great to see you!{company_info} What can I help you with?",
        f"Greetings{name_part} I'm ready to help.{company_info} What questions do you have?",
        f"Hello{name_part} Nice to meet you!{company_info} Feel free to ask me anything.",
        f"Hi{name_part} I'm here and ready to assist.{company_info} What would you like to know?",
        f"Hey there{name_part} Welcome!{company_info} I'm here to answer your questions.",
        f"Hello{name_part} How can I help you today?{company_info}",
        f"Hi{name_part} Good to see you!{company_info} What can I do for you?",
        f"Greetings{name_part} I'm your friendly assistant.{company_info} Ask me anything!",
        f"Hello{name_part} I'm excited to help.{company_info} What would you like to learn about?",
        f"Hi there{name_part} I'm here to assist you.{company_info} What questions can I answer?",
        f"Hey{name_part} Welcome aboard!{company_info} How can I be of service?",
        f"Hello{name_part} It's great to have you here.{company_info} What can I help you discover?",
        f"Hi{name_part} I'm ready to help.{company_info} What would you like to know more about?",
        f"Hey{name_part} {company_info} What can I help you with today?",
        f"Hello there{name_part} I'm glad you're here.{company_info} How may I assist you?",
        f"Hi{name_part} Welcome!{company_info} I'm here to answer any questions you might have.",
        f"Hey{name_part} Good to have you here!{company_info} What would you like to know?",
        f"Hello{name_part} I'm your virtual assistant.{company_info} Feel free to ask me anything!",
        f"Hi there{name_part} Thanks for visiting!{company_info} How can I be of help?",
        f"Hey{name_part} I'm here and ready!{company_info} What can I do for you today?",
        f"Hello{name_part} Welcome to our chatbot!{company_info} What questions do you have?",
        f"Hi{name_part} It's a pleasure to meet you!{company_info} How can I assist?",
        f"Hey there{name_part} I'm excited to help.{company_info} What would you like to learn?",
        f"Hello{name_part} I'm here to make your day easier.{company_info} What can I help with?",
        f"Hi{name_part} Welcome aboard!{company_info} Ask me anything you'd like to know.",
        f"Hey{name_part} Great to chat with you!{company_info} What can I help you discover?",
        f"Hello{name_part} I'm your helpful assistant.{company_info} What would you like to explore?",
        f"Hi there{name_part} I'm ready to assist.{company_info} What questions can I answer for you?",
        f"Hey{name_part} Thanks for reaching out!{company_info} How can I help you today?",
        f"Hello{name_part} I'm here to provide information.{company_info} What would you like to know?",
        f"Hi{name_part} Good day!{company_info} What can I do to help you?",
        f"Hey there{name_part} I'm your friendly helper.{company_info} What can I assist you with?",
        f"Hello{name_part} Welcome!{company_info} I'm here to answer your questions.",
    ]
    return random.choice(greetings)


def extract_name_from_introduction(user_input):
    """Extract name from user input if it's an introduction like 'I am John', 'my name is Sarah', etc."""
    if not user_input:
        return None
    
    user_lower = user_input.lower().strip()
    
    # Patterns to match name introductions - capture name until end of sentence or end of input
    patterns = [
        r'^(?:i\s+am|i\'m|im)\s+([^.!?]+?)(?:[.!?]|$)',  # "I am John", "I'm Sarah", "im Mike"
        r'^(?:my\s+name\s+is|my\s+name\'s|name\s+is|name\'s)\s+([^.!?]+?)(?:[.!?]|$)',  # "my name is John", "name is Sarah"
        r'^(?:this\s+is|it\'s|its)\s+([^.!?]+?)(?:[.!?]|$)',  # "this is John", "it's Sarah"
        r'^(?:i\s+go\s+by|call\s+me)\s+([^.!?]+?)(?:[.!?]|$)',  # "I go by John", "call me Sarah"
        r'^(?:you\s+can\s+call\s+me|just\s+call\s+me)\s+([^.!?]+?)(?:[.!?]|$)',  # "you can call me John"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, user_lower, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Remove common trailing phrases
            name = re.sub(r'\s+(here|speaking|talking|and|,).*$', '', name, flags=re.IGNORECASE)
            # Clean up the name - remove extra punctuation, but keep valid name characters
            name = re.sub(r'^[^\w]+|[^\w]+$', '', name)  # Remove leading/trailing non-word chars
            # Remove any remaining trailing commas or conjunctions
            name = re.sub(r',\s*$', '', name)
            if name and len(name) > 0 and len(name) <= 50:  # Reasonable name length
                # Capitalize first letter of each word
                name = ' '.join(word.capitalize() for word in name.split())
                return name
    
    return None


def is_greeting(user_input):
    """Check if user input is a greeting, handling spelling variations and typos"""
    if not user_input:
        return False
    
    # Normalize input: lowercase, strip, normalize spaces
    user_lower = user_input.lower().strip()
    user_lower = re.sub(r'\s+', ' ', user_lower)  # Normalize spaces
    
    # Remove trailing punctuation for matching (but keep for pattern matching)
    user_clean = re.sub(r'[\.!?]+$', '', user_lower).strip()
    
    # Exact matches (without punctuation)
    exact_greetings = [
        'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
        'good evening', 'good night', 'morning', 'afternoon', 'evening',
        'hi there', 'hello there', 'hey there', 'howdy', 'sup', 'what\'s up',
        'whats up', 'greet', 'salutations', 'hiya', 'hola', 'bonjour', 'g\'day',
        'gday', 'thanks', 'thank you', 'thankyou', 'ty', 'thx', 'thank u'
    ]
    
    # Check exact matches first (with and without punctuation)
    if user_clean in exact_greetings or user_lower in exact_greetings:
        return True
    
    # Check if starts with exact greeting
    if any(user_clean.startswith(keyword) or user_lower.startswith(keyword) 
           for keyword in exact_greetings):
        return True
    
    # Pattern matching for common variations and typos (check original input)
    greeting_patterns = [
        r'^h+i+[\.!]*\s*$',  # hi, hii, hiii, hi., hi.., etc.
        r'^h+i+[\.!]{2,}\s*$',  # hi.., hi..., etc.
        r'^h+e+l+o+[\.!]*\s*$',  # hello, helllo, hello., etc.
        r'^h+e+y+[\.!]*\s*$',  # hey, heyy, heyyy, hey., etc.
        r'^h+a+i+[\.!]*\s*$',  # hai, haii, haiii (common misspelling)
        r'^h+a+i+[\.!]{2,}\s*$',  # haii.., haiii..., etc.
        r'^h+e+l+[\.!]*\s*$',  # hel, hell (partial hello)
        r'^h+y+[\.!]*\s*$',  # hy, hyy (partial hey)
        r'^g+o+o+d+\s+(morning|afternoon|evening|night)',  # good morning variations
        r'^(morning|afternoon|evening|night)[\.!]*\s*$',  # just time of day
        r'^h+i+\s+t+h+e+r+e+',  # hi there variations
        r'^h+e+l+l+o+\s+t+h+e+r+e+',  # hello there variations
        r'^h+e+y+\s+t+h+e+r+e+',  # hey there variations
        r'^t+h+a+n+k+s*[\.!]*',  # thanks, thank, thx variations
        r'^t+h+a+n+k+\s+y+o+u+[\.!]*',  # thank you variations
        r'^g+r+e+e+t+i+n+g+s*[\.!]*',  # greetings variations
        r'^h+o+w+d+y+[\.!]*',  # howdy variations
        r'^w+h+a+t+s*\s+u+p+[\.!]*',  # what's up variations
        r'^s+u+p+[\.!]*',  # sup variations
    ]
    
    # Check against patterns on original input
    for pattern in greeting_patterns:
        if re.match(pattern, user_lower):
            return True
    
    # Check for common misspellings with repeated characters
    # Remove repeated characters to check base word (hii -> hi, heyy -> hey)
    base_word = re.sub(r'(.)\1+', r'\1', user_clean)
    
    base_greetings = ['hi', 'hey', 'helo', 'hello', 'hai', 'greet', 'hel', 'hy']
    if any(base_word.startswith(greeting) for greeting in base_greetings):
        # Additional check: if it's mostly greeting-like (short, starts with h/g/t)
        if len(user_clean) <= 20:
            # Check if it starts with greeting-like characters
            if user_clean and user_clean[0] in 'hgt':
                # Make sure it's not a full word that's not a greeting
                if len(user_clean) <= 10 or base_word in base_greetings:
                    return True
    
    # Special case: handle "hi.." with multiple dots
    if re.match(r'^h+i+\.{2,}', user_lower):
        return True
    if re.match(r'^h+a+i+\.{2,}', user_lower):
        return True
    
    return False


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


def basic_chatbot_response(user_input, embedder, scraped_data_pages, scraped_data_embeddings):
    """Basic chatbot response using semantic similarity"""
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


def retrieve_relevant_chunks(
    user_input, 
    embedder,
    scraped_data_pages,
    scraped_data_embeddings,
    structured_data_texts,
    structured_data_embeddings,
    structured_data_sources,
    custom_data_texts,
    custom_data_embeddings,
    custom_data_sources,
    top_k=3, 
    similarity_threshold=0.0
):
    """Enhanced semantic retrieval that combines all data sources"""
    if not embedder:
        print("❌ Sentence transformer not available")
        return []

    # Normalize potentially missing inputs so downstream len() checks are safe
    scraped_data_pages = scraped_data_pages or []
    structured_data_texts = structured_data_texts or []
    structured_data_sources = structured_data_sources or []
    custom_data_texts = custom_data_texts or []
    custom_data_sources = custom_data_sources or []

    all_chunks = []
    all_embeddings = []
    all_sources = []
    
    # Add scraped data
    has_scraped_embeddings = False
    if scraped_data_embeddings is not None:
        try:
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
        print(f"Added {len(scraped_data_pages)} scraped pages to retrieval pool")
    else:
        print(f"Scraped data not available: pages={len(scraped_data_pages)}, embeddings={'None' if scraped_data_embeddings is None else 'empty'}")
    
    # Add structured data
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
        print(f"Added {len(structured_data_texts)} structured entries to retrieval pool")
    
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
        custom_chunks = [(f"Custom: {source['title']}", text) for text, source in zip(custom_data_texts, custom_data_sources)]
        all_chunks.extend(custom_chunks)
        all_embeddings.append(custom_data_embeddings)
        all_sources.extend(['custom'] * len(custom_data_texts))
        print(f"Added {len(custom_data_texts)} custom entries to retrieval pool")
        for i, (title, text) in enumerate(custom_chunks):
            print(f"   Custom entry {i+1}: {title} (content preview: {text[:50]}...)")
    
    if not all_chunks:
        print("No data available for retrieval - check if embeddings exist or run scraping first")
        return []

    try:
        # Concatenate all embeddings
        if len(all_embeddings) == 1:
            combined_embeddings = all_embeddings[0]
        else:
            if TORCH_AVAILABLE:
                combined_embeddings = torch.cat(all_embeddings, dim=0)
            else:
                combined_embeddings = np.concatenate(all_embeddings, axis=0)
        
        query_embedding = embedder.encode(user_input, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, combined_embeddings)[0]
        
        # PRIORITIZE CUSTOM DATA: Give significant boost to custom data entries
        custom_data_start_idx = len(scraped_data_pages) + len(structured_data_texts)
        custom_data_end_idx = custom_data_start_idx + len(custom_data_texts)
        
        # Boost all custom data entries significantly
        for i in range(custom_data_start_idx, min(custom_data_end_idx, len(similarities))):
            similarities[i] = similarities[i] + 0.5  # Large boost for custom data
        
        # Additional boost if query keywords match custom data content or title
        query_lower = user_input.lower().strip()
        query_words = set(query_lower.split())
        
        for i in range(custom_data_start_idx, min(custom_data_end_idx, len(all_chunks))):
            chunk = all_chunks[i]
            chunk_text = chunk[1].lower() if isinstance(chunk, tuple) else str(chunk).lower()
            chunk_title = chunk[0].lower() if isinstance(chunk, tuple) else ""
            
            # Extract title from "Custom: title" format
            if chunk_title.startswith('custom:'):
                title_only = chunk_title.replace('custom:', '').strip()
            else:
                title_only = chunk_title
            
            # Check for exact title match (highest priority)
            if title_only and (query_lower == title_only or query_lower in title_only or title_only in query_lower):
                similarities[i] = similarities[i] + 0.8  # Very high boost for exact title match
                print(f"🎯 Exact title match found in custom data: '{title_only}' for query '{query_lower}'")
            # Check for keyword matches in title or content
            elif any(word in chunk_text or word in title_only for word in query_words if len(word) > 1):
                similarities[i] = similarities[i] + 0.4  # Additional boost for keyword match
                print(f"✅ Keyword match found in custom data: '{title_only}' for query '{query_lower}'")
        
        # Detect query type for boosting
        is_address_query_retrieval = any(keyword in user_input.lower() for keyword in ['address', 'location', 'where', 'office', 'contact', 'located'])
        is_about_query_retrieval = any(keyword in user_input.lower() for keyword in ['tell me about', 'what is', 'describe', 'who is', 'about', 'information', 'overview', 
                                                                                     'background', 'what does', 'details', 'about us', 'story', 'history', 'what are'])
        is_product_query_retrieval = any(keyword in user_input.lower() for keyword in ['product', 'service', 'offering', 'what do you', 'what does', 'what can', 'offer', 'provide', 'sell', 'solution'])
        
        # Boost chunks based on query type
        # Generic address keywords that work for any website
        address_keywords = ['address', 'location', 'office', 'building', 'street', 'avenue', 'road', 'postal', 'zip code', 'zip', 
                           'city', 'state', 'country', 'contact', 'located', 'headquarters', 'hq', 'suite', 'floor', 'room', 
                           'p.o. box', 'po box', 'postcode', 'postal code']
        if is_address_query_retrieval:
            for i in range(len(all_chunks)):
                chunk_text = all_chunks[i][1].lower() if isinstance(all_chunks[i], tuple) else str(all_chunks[i]).lower()
                # Check for address keywords or postal code patterns (5-6 digit numbers)
                has_address_keyword = any(keyword in chunk_text for keyword in address_keywords)
                has_postal_code = bool(re.search(r'\b\d{5,6}\b', chunk_text))  # 5-6 digit postal codes
                if has_address_keyword or has_postal_code:
                    similarities[i] = similarities[i] + 0.2
        
        product_keywords_retrieval = ['product', 'service', 'offering', 'solution', 'app', 'software', 'platform', 'tool', 
                                     'what we do', 'our services', 'our products', 'services we offer', 'what we offer']
        if is_product_query_retrieval:
            for i in range(len(all_chunks)):
                chunk_text = all_chunks[i][1].lower() if isinstance(all_chunks[i], tuple) else str(all_chunks[i]).lower()
                chunk_url = all_chunks[i][0].lower() if isinstance(all_chunks[i], tuple) else ""
                
                url_boost = 0.4 if any(keyword in chunk_url for keyword in ['product', 'service', 'solution', 'offering', 'services', 'products']) else 0.0
                content_boost = 0.3 if any(keyword in chunk_text for keyword in product_keywords_retrieval) else 0.0
                penalty = -0.2 if any(keyword in chunk_url for keyword in ['contact', 'privacy', 'terms', 'policy', 'legal']) else 0.0
                
                total_boost = url_boost + content_boost + penalty
                if total_boost != 0:
                    similarities[i] = similarities[i] + total_boost
        
        about_keywords = ['about', 'mission', 'vision', 'history', 'story', 'culture', 'values', 'team', 
                         'who we are', 'what we do', 'overview', 'background', 'our story', 'our mission', 'our vision',
                         'introduction', 'welcome', 'home', 'main page']
        if is_about_query_retrieval:
            for i in range(len(all_chunks)):
                chunk_text = all_chunks[i][1].lower() if isinstance(all_chunks[i], tuple) else str(all_chunks[i]).lower()
                chunk_url = all_chunks[i][0].lower() if isinstance(all_chunks[i], tuple) else ""
                if any(keyword in chunk_text for keyword in about_keywords) or any(keyword in chunk_url for keyword in ['about', 'story', 'mission', 'vision', 'home', 'main']):
                    similarities[i] = similarities[i] + 0.3
        
        # Get top k results
        top_k_indices = similarities.argsort(descending=True)[:top_k * 3]
        
        # PRIORITIZE CUSTOM DATA: Reorder results to put custom data first
        custom_data_indices = []
        other_indices = []
        
        for i in top_k_indices:
            if custom_data_start_idx <= i < custom_data_end_idx:
                custom_data_indices.append(i)
            else:
                other_indices.append(i)
        
        # Put custom data first, then others
        top_k_indices = custom_data_indices + other_indices
        if len(top_k_indices) > top_k * 3:
            top_k_indices = top_k_indices[:top_k * 3]
        
        if custom_data_indices:
            print(f"🎯 Prioritized {len(custom_data_indices)} custom data entries in results")
        
        # For product queries, prioritize product pages (but keep custom data first)
        if is_product_query_retrieval:
            # Separate custom data from others to keep it first
            custom_in_top_k = [i for i in top_k_indices if custom_data_start_idx <= i < custom_data_end_idx]
            non_custom_in_top_k = [i for i in top_k_indices if not (custom_data_start_idx <= i < custom_data_end_idx)]
            
            product_page_indices = []
            other_page_indices = []
            
            for i in non_custom_in_top_k:
                chunk = all_chunks[i]
                chunk_url = chunk[0].lower() if isinstance(chunk, tuple) else str(chunk).lower()
                
                is_product_page = any(keyword in chunk_url for keyword in ['product', 'service', 'solution', 'offering', 'services', 'products'])
                is_non_product_page = any(keyword in chunk_url for keyword in ['contact', 'privacy', 'terms', 'policy', 'legal'])
                
                if is_product_page:
                    product_page_indices.append(i)
                elif not is_non_product_page:
                    other_page_indices.append(i)
            
            # Keep custom data first, then product pages, then others
            prioritized_indices = custom_in_top_k + product_page_indices + other_page_indices
            if len(prioritized_indices) < len(top_k_indices):
                remaining = [i for i in top_k_indices if i not in prioritized_indices]
                prioritized_indices.extend(remaining)
            
            top_k_indices = prioritized_indices[:top_k * 3]
        
        results = []
        max_similarity = 0.0
        
        for i in top_k_indices:
            similarity_score = similarities[i].item()
            max_similarity = max(max_similarity, similarity_score)
            
            # Apply similarity threshold - only include chunks that meet the threshold
            if similarity_score < similarity_threshold:
                continue
            
            chunk = all_chunks[i]
            
            chunk_text = chunk[1] if isinstance(chunk, tuple) else str(chunk)
            cleaned_chunk = clean_text_content(chunk_text)
            
            if not is_valid_content(cleaned_chunk, min_length=20):
                continue
            
            results.append(chunk)
            if len(results) >= top_k:
                break
        
        # Add more if needed (still respecting similarity threshold)
        if len(results) < top_k and len(top_k_indices) > len(results):
            for i in top_k_indices[len(results):]:
                if len(results) >= top_k:
                    break
                similarity_score = similarities[i].item()
                if similarity_score < similarity_threshold:
                    continue
                chunk = all_chunks[i]
                chunk_text = chunk[1] if isinstance(chunk, tuple) else str(chunk)
                cleaned_chunk = clean_text_content(chunk_text)
                if len(cleaned_chunk.strip()) > 20:
                    results.append(chunk)
        
        total_chunks = len(scraped_data_pages) + len(structured_data_texts) + len(custom_data_texts)
        print(f"Retrieved {len(results)} relevant chunks out of {total_chunks} total (max similarity: {max_similarity:.3f}, threshold: {similarity_threshold:.3f})")
        
        # If no results meet the threshold, return empty list
        if len(results) == 0:
            print(f"⚠️ No chunks found above similarity threshold of {similarity_threshold:.3f} (best match: {max_similarity:.3f})")
        
        return results
        
    except Exception as e:
        print(f"Error in semantic retrieval: {e}")
        import traceback
        traceback.print_exc()
        return []


def generate_openai_response(
    user_input, 
    relevant_chunks, 
    openai_api_key,
    openai_model_name,
    is_about_query=False, 
    is_address_query=False, 
    is_product_query=False,
    user_name=""
):
    """Generate OpenAI response with context"""
    if not openai_api_key:
        return "OpenAI API key not configured. Please set your OPENAI_API_KEY environment variable.", [], None

    # Check if no relevant chunks were found - return early with no information message
    if not relevant_chunks or len(relevant_chunks) == 0:
        # Check if it's a greeting - still respond to greetings even without context
        if is_greeting(user_input):
            return get_greeting_response("", user_name), [], None
        else:
            return "I don't have information about that topic in the website content. Please ask me something related to the website.", [], None

    try:
        client = openai.OpenAI(api_key=openai_api_key)
        
        # Detect query types
        address_keywords = ['address', 'location', 'where', 'office', 'contact', 'phone', 'email', 'located']
        is_address_query = any(keyword in user_input.lower() for keyword in address_keywords)
        
        product_keywords = ['product', 'service', 'offering', 'what do you', 'what does', 'what can', 'offer', 'provide', 'sell']
        is_product_query = any(keyword in user_input.lower() for keyword in product_keywords)
        
        about_keywords = ['tell me about', 'what is', 'describe', 'who is', 'about', 'information', 'overview', 
                          'background', 'what does', 'details', 'about us', 'story', 'history', 'what are']
        is_about_query = any(keyword in user_input.lower() for keyword in about_keywords)
        
        # Detect person/name queries (CEO, founder, team member, etc.)
        person_keywords = ['who is', 'ceo', 'founder', 'co-founder', 'president', 'director', 'manager', 
                          'team member', 'employee', 'staff', 'leader', 'head of', 'name of', 'person']
        is_person_query = any(keyword in user_input.lower() for keyword in person_keywords)
        
        if relevant_chunks:
            # Get model context limit
            context_limit = get_model_context_limit(openai_model_name)
            
            # Estimate tokens for system message
            system_message = "You are a friendly chatbot for a website. You answer questions naturally and conversationally, using ONLY the information provided. Always use first person (we, our, us) to represent the website/organization - never use third person (they, their, them). Never mention sources, data, or context - just respond naturally as a helpful chatbot would. Never make up information that isn't in the provided context."
            
            # Estimate tokens for base prompt template (without context and special instructions)
            base_prompt_template = """You are a helpful chatbot for this website. Answer the user's question naturally and conversationally, as if you're a friendly representative of the website.

CRITICAL RULES:
1. **ONLY use information from the context below** - DO NOT make up, guess, or hallucinate any information. If the information isn't in the context, you cannot provide it.
2. **Respond naturally** - Write as a friendly chatbot would, not as an AI assistant explaining its process.
3. **ALWAYS use FIRST PERSON** - Use "we", "our", "us" instead of "they", "their", "them". You are representing the website/organization directly, so speak as if you are part of it.
4. **NEVER use phrases like**: "Based on the context", "According to the information provided", "From the context above", etc.
5. **Just answer directly** - State facts naturally as if you know them, without explaining where you got them from.
6. **Be comprehensive** - Include all relevant details from the context in your answer.
7. **IMPORTANT: For person names** - Always provide the COMPLETE and FULL name. Never truncate, shorten, or abbreviate names. If you see a full name in the context, use the entire name exactly as it appears.
8. **Ignore error messages** - Skip any JavaScript errors, HTML errors, or placeholder text in the context.
9. **If asked a greeting** (hi, hello, hey, good morning, etc.), respond warmly and offer to help. Use a friendly, conversational tone.

WEBSITE CONTENT:
{context_placeholder}

User's question: {user_input}

Answer the question naturally and conversationally, using ONLY the information from the website content above. Use first person (we, our, us) to represent the website/organization. Do not mention sources, data, or context - just provide a helpful, natural response:"""
            
            # Estimate special instruction size (will be calculated later, use average for now)
            avg_special_instruction_tokens = 150
            
            # Estimate tokens for fixed parts (system message + base prompt template + user input + special instruction + response buffer)
            fixed_tokens = estimate_tokens(system_message, openai_model_name) + \
                          estimate_tokens(base_prompt_template.replace("{context_placeholder}", ""), openai_model_name) + \
                          estimate_tokens(user_input, openai_model_name) + \
                          avg_special_instruction_tokens + \
                          2000  # Buffer for response
            
            # Available tokens for context
            available_tokens = context_limit - fixed_tokens
            
            # Safety margin: use only 90% of available tokens to avoid edge cases
            max_context_tokens = int(available_tokens * 0.9)
            
            context_str = "\n\n--- Context ---\n"
            source_urls = []  # Use list to preserve order - most relevant first
            source_urls_set = set()  # Track duplicates
            
            is_generic_query = is_greeting(user_input) or user_input.lower().strip() in ['what can you do', 'help', '?']
            
            # Set text limits based on query type
            # For person queries, use higher limits to ensure full names are captured
            if is_person_query:
                text_limit = 8000  # Higher limit for person queries to capture full names
            elif is_about_query:
                text_limit = 5000
            elif is_address_query:
                text_limit = 3000
            elif is_product_query:
                text_limit = 3000
            elif is_generic_query:
                text_limit = 2000
            else:
                text_limit = 1500
            
            valid_chunks_used = 0
            current_context_tokens = estimate_tokens(context_str, openai_model_name)
            
            def add_chunk_to_context(chunk_text, chunk_url, chunk_type="Source"):
                """Helper function to add a chunk and check token limits"""
                nonlocal context_str, current_context_tokens, valid_chunks_used, source_urls, source_urls_set
                
                # Estimate tokens for this chunk
                chunk_str = f"{chunk_type} {valid_chunks_used+1} ({chunk_url}):\n{chunk_text}...\n\n"
                chunk_tokens = estimate_tokens(chunk_str, openai_model_name)
                
                # Check if adding this chunk would exceed the limit
                if current_context_tokens + chunk_tokens > max_context_tokens:
                    return False  # Can't add this chunk
                
                # Add the chunk
                context_str += chunk_str
                current_context_tokens += chunk_tokens
                
                if chunk_url not in source_urls_set:
                    source_urls.append(chunk_url)
                    source_urls_set.add(chunk_url)
                valid_chunks_used += 1
                return True
            
            for i, (url, text) in enumerate(relevant_chunks):
                cleaned_text = clean_text_content(text)
                
                if not is_valid_content(cleaned_text, min_length=30):
                    continue
                
                # Check if we've already used too many tokens
                if current_context_tokens >= max_context_tokens:
                    break
                
                if url.startswith('Custom:'):
                    # For person queries, use full text or higher limit
                    if is_person_query:
                        chunk_text = cleaned_text[:8000]
                    else:
                        chunk_text = cleaned_text[:text_limit]
                    
                    if not add_chunk_to_context(chunk_text, url, "Custom Data"):
                        # Try with reduced size
                        reduced_limit = min(text_limit, 2000)
                        chunk_text = cleaned_text[:reduced_limit]
                        if not add_chunk_to_context(chunk_text, url, "Custom Data"):
                            break
                elif url.startswith('Structured:'):
                    if not add_chunk_to_context(cleaned_text, url, "Structured Data"):
                        break
                else:
                    # Handle different query types with appropriate context
                    # Check if text contains team/person keywords and increase limit
                    person_keywords_in_text = ['ceo', 'founder', 'co-founder', 'president', 'director', 'manager', 
                                              'team', 'member', 'employee', 'staff', 'leader', 'head', 'name']
                    has_person_content = any(keyword in cleaned_text.lower() for keyword in person_keywords_in_text)
                    
                    chunk_text = None
                    chunk_label = "Source"
                    
                    if is_address_query:
                        # Generic address detection that works for any website
                        address_keywords_in_text = ['address', 'location', 'office', 'building', 'street', 'avenue', 'road', 
                                                   'postal', 'zip code', 'zip', 'city', 'state', 'country', 'contact', 
                                                   'located', 'headquarters', 'hq', 'suite', 'floor', 'room', 'p.o. box', 
                                                   'po box', 'postcode', 'postal code', 'address:', 'location:']
                        has_address_keyword = any(keyword in cleaned_text.lower() for keyword in address_keywords_in_text)
                        has_postal_code = bool(re.search(r'\b\d{5,6}\b', cleaned_text))  # 5-6 digit postal codes
                        # Check for common address patterns (street numbers, common address words)
                        has_address_pattern = bool(re.search(r'\d+\s+(street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln)', cleaned_text.lower()))
                        
                        if has_address_keyword or has_postal_code or has_address_pattern:
                            chunk_text = cleaned_text[:3000]
                            chunk_label = "Source [CONTAINS ADDRESS INFO]"
                        else:
                            chunk_text = cleaned_text[:text_limit]
                    elif is_about_query:
                        about_keywords_in_text = ['about', 'mission', 'vision', 'history', 'story', 'culture', 
                                                  'values', 'team', 'who we are', 'what we do', 'overview', 'background', 'introduction']
                        if any(keyword in cleaned_text.lower() for keyword in about_keywords_in_text):
                            # For person queries or person content, use even higher limit
                            limit = 10000 if (is_person_query or has_person_content) else 6000
                            chunk_text = cleaned_text[:limit]
                            chunk_label = "Source [ABOUT INFO - HIGH PRIORITY]"
                        else:
                            limit = 8000 if (is_person_query or has_person_content) else text_limit
                            chunk_text = cleaned_text[:limit]
                    elif is_product_query:
                        product_keywords_in_text = ['product', 'service', 'offering', 'solution', 'app', 'software', 'platform', 'tool',
                                                    'what we do', 'our services', 'our products', 'services we offer', 'what we offer',
                                                    'mobile app', 'web development', 'digital marketing', 'software development']
                        product_keywords_in_url = ['product', 'service', 'solution', 'offering', 'services', 'products']
                        
                        is_product_page = any(keyword in url.lower() for keyword in product_keywords_in_url)
                        has_product_content = any(keyword in cleaned_text.lower() for keyword in product_keywords_in_text)
                        
                        if is_product_page:
                            chunk_text = cleaned_text[:5000]
                            chunk_label = "Source [PRODUCT/SERVICE PAGE - HIGHEST PRIORITY]"
                        elif has_product_content:
                            chunk_text = cleaned_text[:4000]
                            chunk_label = "Source [CONTAINS PRODUCT INFO]"
                        else:
                            chunk_text = cleaned_text[:text_limit]
                    else:
                        # For person queries or person content, use higher limit
                        limit = 8000 if (is_person_query or has_person_content) else text_limit
                        chunk_text = cleaned_text[:limit]
                    
                    # Try to add the chunk, with fallback to smaller size if needed
                    if chunk_text:
                        if not add_chunk_to_context(chunk_text, url, chunk_label):
                            # Try with reduced size
                            reduced_limit = min(len(chunk_text), 2000)
                            chunk_text = cleaned_text[:reduced_limit]
                            if not add_chunk_to_context(chunk_text, url, chunk_label):
                                break
                
                # Also check max_chunks limit (but token limit takes priority)
                max_chunks = 15 if is_about_query else 12 if is_product_query else 10 if is_address_query else 8
                if valid_chunks_used >= max_chunks:
                    break
            
            context_str += "--- End Context ---"
            # Limit source URLs to only the most relevant ones
            # For most queries, return only the top 1-2 sources (most relevant)
            # Only return more if it's a complex query that might legitimately need multiple sources
            if is_about_query and len(source_urls) > 2:
                # About queries might need 2 sources max
                source_urls = source_urls[:2]
            elif len(source_urls) > 1:
                # For most other queries, return only the top 1 source (most relevant)
                source_urls = source_urls[:1]
            # If there's only 1 or 0 sources, keep as is
        else:
            context_str = "No relevant content found."
            source_urls = []
        
        # Build prompt
        special_instruction = ""
        if is_address_query:
            special_instruction = """
The user is asking about address or location. Look carefully through the website content for:
- Street addresses, building names/numbers
- Cities, states, countries
- Postal codes
- Any location information

If multiple locations exist, mention all of them using first person. Say "We have offices at..." or "Our offices are located at..." NOT "They have offices at..." or "They also have...". Include all address details you find, even if incomplete.
"""
        elif is_about_query:
            person_note = ""
            if is_person_query:
                person_note = "\nIMPORTANT: If the user is asking about a person (CEO, founder, team member, etc.), make sure to provide the COMPLETE and FULL name. Do not truncate or shorten names. Include the person's full name exactly as it appears in the context."
            
            special_instruction = f"""
The user is asking about the website/organization/topic. Provide a comprehensive description including:
- What it is about or what it does
- Mission, vision, values (if applicable)
- History and background
- Services, products, or offerings (if applicable)
- Key information, features, or highlights
- Achievements and milestones (if applicable)
- Any other relevant details
{person_note}

Focus on providing comprehensive information, not just contact details. Be detailed and comprehensive.
"""
        elif is_product_query:
            special_instruction = """
The user is asking about products, services, or offerings. Look for:
- Product names, service names, offerings, solutions
- Features, capabilities, applications
- Platforms, tools, software, apps
- All services, products, or offerings available

List ALL products and services you find. Be specific and detailed with names, descriptions, and features. Prioritize product/service information over contact or policy pages.
"""
        
        is_greeting_query = is_greeting(user_input)
        company_info = ""
        
        if relevant_chunks:
            first_chunk_text = relevant_chunks[0][1] if relevant_chunks else ""
            website_name = ""
            if "PAGE TITLE:" in first_chunk_text:
                title_line = [line for line in first_chunk_text.split('\n') if 'PAGE TITLE:' in line]
                if title_line:
                    website_name = title_line[0].replace('PAGE TITLE:', '').strip()
                    for sep in [' - ', ' | ', ' :: ', ' – ']:
                        if sep in website_name:
                            website_name = website_name.split(sep)[0].strip()
            
            if website_name:
                company_info = f" I can help you with information about {website_name}."
        
        prompt = f"""You are a helpful chatbot for this website. Answer the user's question naturally and conversationally, as if you're a friendly representative of the website.

{special_instruction}

CRITICAL RULES:
1. **ONLY use information from the context below** - DO NOT make up, guess, or hallucinate any information. If the information isn't in the context, you cannot provide it.
2. **Respond naturally** - Write as a friendly chatbot would, not as an AI assistant explaining its process.
3. **ALWAYS use FIRST PERSON** - Use "we", "our", "us" instead of "they", "their", "them". You are representing the website/organization directly, so speak as if you are part of it.
4. **NEVER use phrases like**: "Based on the context", "According to the information provided", "From the context above", etc.
5. **Just answer directly** - State facts naturally as if you know them, without explaining where you got them from.
6. **Be comprehensive** - Include all relevant details from the context in your answer.
7. **IMPORTANT: For person names** - Always provide the COMPLETE and FULL name. Never truncate, shorten, or abbreviate names. If you see a full name in the context, use the entire name exactly as it appears.
8. **Ignore error messages** - Skip any JavaScript errors, HTML errors, or placeholder text in the context.
9. **If asked a greeting** (hi, hello, hey, good morning, etc.), respond warmly and offer to help{company_info}. Use a friendly, conversational tone.

WEBSITE CONTENT:
{context_str}

User's question: {user_input}

Answer the question naturally and conversationally, using ONLY the information from the website content above. Use first person (we, our, us) to represent the website/organization. Do not mention sources, data, or context - just provide a helpful, natural response:"""

        response = client.chat.completions.create(
            model=openai_model_name,
            messages=[
                {"role": "system", "content": "You are a friendly chatbot for a website. You answer questions naturally and conversationally, using ONLY the information provided. Always use first person (we, our, us) to represent the website/organization - never use third person (they, their, them). Never mention sources, data, or context - just respond naturally as a helpful chatbot would. Never make up information that isn't in the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=3000
        )
        
        if not response.choices or len(response.choices) == 0:
            raise Exception("No response from OpenAI API")
        
        answer = response.choices[0].message.content.strip()
        
        token_usage = None
        if hasattr(response, 'usage') and response.usage:
            token_usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        
        answer = clean_text_content(answer)
        
        # Fallback handling
        no_info_phrases = ["i don't have", "i do not have", "i cannot find", "no information", "don't have that information"]
        if any(phrase in answer.lower() for phrase in no_info_phrases) and relevant_chunks:
            if is_greeting_query:
                answer = get_greeting_response(company_info, user_name)
            else:
                answer_parts = []
                for url, text in relevant_chunks[:5]:
                    cleaned = clean_text_content(text[:800].strip())
                    if is_valid_content(cleaned, min_length=100):
                        answer_parts.append(cleaned)
                
                if answer_parts:
                    answer = "\n\n".join(answer_parts)
        
        # source_urls is already a list, return it directly
        return answer, source_urls, token_usage
        
    except Exception as e:
        import traceback
        print(f"OpenAI error: {e}")
        traceback.print_exc()
        error_msg = str(e)
        if "invalid_api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            return "Invalid API key. Please check your OpenAI API key configuration.", [], None
        elif "permission" in error_msg.lower() or "forbidden" in error_msg.lower():
            return "Permission denied. Please check your API key permissions.", [], None
        elif "429" in error_msg or "rate_limit" in error_msg.lower() or "quota" in error_msg.lower():
            return "API rate limit exceeded. Please try again later or upgrade your plan.", [], None
        elif "context_length" in error_msg.lower() or "token" in error_msg.lower():
            return "The context is too long. Please try a more specific question or reduce the amount of data.", [], None
        else:
            return f"Error generating response: {error_msg}. Please check the server logs for more details.", [], None

