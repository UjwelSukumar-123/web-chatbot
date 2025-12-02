"""
Chatbot Module
Handles all chatbot logic including semantic retrieval and response generation
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
import openai
from typing import List, Tuple, Optional, Dict

# Try to import torch, fallback to numpy if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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
        print(f"📚 Added {len(scraped_data_pages)} scraped pages to retrieval pool")
    else:
        print(f"⚠️ Scraped data not available: pages={len(scraped_data_pages)}, embeddings={'None' if scraped_data_embeddings is None else 'empty'}")
    
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
        custom_chunks = [(f"Custom: {source['title']}", text) for text, source in zip(custom_data_texts, custom_data_sources)]
        all_chunks.extend(custom_chunks)
        all_embeddings.append(custom_data_embeddings)
        all_sources.extend(['custom'] * len(custom_data_texts))
        print(f"📝 Added {len(custom_data_texts)} custom entries to retrieval pool")
        for i, (title, text) in enumerate(custom_chunks):
            print(f"   Custom entry {i+1}: {title} (content preview: {text[:50]}...)")
    
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
        address_keywords = ['address', 'location', 'thrikkakara', 'kakkanad', 'kochi', 'dubai', 'uae', 'office', 'building', 'street', 'postal', '682021']
        if is_address_query_retrieval:
            for i in range(len(all_chunks)):
                chunk_text = all_chunks[i][1].lower() if isinstance(all_chunks[i], tuple) else str(all_chunks[i]).lower()
                if any(keyword in chunk_text for keyword in address_keywords):
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
        for i in top_k_indices:
            similarity_score = similarities[i].item()
            chunk = all_chunks[i]
            
            chunk_text = chunk[1] if isinstance(chunk, tuple) else str(chunk)
            cleaned_chunk = clean_text_content(chunk_text)
            
            if not is_valid_content(cleaned_chunk, min_length=20):
                continue
            
            results.append(chunk)
            if len(results) >= top_k:
                break
        
        # Add more if needed
        if len(results) < top_k and len(top_k_indices) > len(results):
            for i in top_k_indices[len(results):]:
                if len(results) >= top_k:
                    break
                chunk = all_chunks[i]
                chunk_text = chunk[1] if isinstance(chunk, tuple) else str(chunk)
                cleaned_chunk = clean_text_content(chunk_text)
                if len(cleaned_chunk.strip()) > 20:
                    results.append(chunk)
        
        total_chunks = len(scraped_data_pages) + len(structured_data_texts) + len(custom_data_texts)
        print(f"Retrieved {len(results)} relevant chunks out of {total_chunks} total")
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
    is_product_query=False
):
    """Generate OpenAI response with context"""
    if not openai_api_key:
        return "OpenAI API key not configured. Please set your OPENAI_API_KEY environment variable.", [], None

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
        
        if relevant_chunks:
            context_str = "\n\n--- Context ---\n"
            source_urls = set()
            
            is_generic_query = user_input.lower().strip() in ['hi', 'hello', 'hey', 'greetings', 'what can you do', 'help', '?']
            
            # Set text limits based on query type
            if is_about_query:
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
            for i, (url, text) in enumerate(relevant_chunks):
                cleaned_text = clean_text_content(text)
                
                if not is_valid_content(cleaned_text, min_length=30):
                    continue
                
                if url.startswith('Custom:'):
                    context_str += f"Custom Data {valid_chunks_used+1} ({url}):\n{cleaned_text[:text_limit]}...\n\n"
                    source_urls.add(url)
                    valid_chunks_used += 1
                elif url.startswith('Structured:'):
                    context_str += f"Structured Data {valid_chunks_used+1} ({url}):\n{cleaned_text}\n\n"
                    source_urls.add(url)
                    valid_chunks_used += 1
                else:
                    # Handle different query types with appropriate context
                    if is_address_query:
                        address_keywords_in_text = ['thrikkakara', 'kakkanad', 'kochi', 'dubai', 'uae', 'address:', '682021', 'sheikh rashid', 'opp. bmc']
                        if any(keyword in cleaned_text.lower() for keyword in address_keywords_in_text):
                            context_str += f"Source {valid_chunks_used+1} ({url}) [CONTAINS ADDRESS INFO]:\n{cleaned_text[:3000]}...\n\n"
                        else:
                            context_str += f"Source {valid_chunks_used+1} ({url}):\n{cleaned_text[:text_limit]}...\n\n"
                    elif is_about_query:
                        about_keywords_in_text = ['about', 'mission', 'vision', 'history', 'story', 'culture', 
                                                  'values', 'team', 'who we are', 'what we do', 'overview', 'background', 'introduction']
                        if any(keyword in cleaned_text.lower() for keyword in about_keywords_in_text):
                            context_str += f"Source {valid_chunks_used+1} ({url}) [ABOUT INFO - HIGH PRIORITY]:\n{cleaned_text[:6000]}...\n\n"
                        else:
                            context_str += f"Source {valid_chunks_used+1} ({url}):\n{cleaned_text[:text_limit]}...\n\n"
                    elif is_product_query:
                        product_keywords_in_text = ['product', 'service', 'offering', 'solution', 'app', 'software', 'platform', 'tool',
                                                    'what we do', 'our services', 'our products', 'services we offer', 'what we offer',
                                                    'mobile app', 'web development', 'digital marketing', 'software development']
                        product_keywords_in_url = ['product', 'service', 'solution', 'offering', 'services', 'products']
                        
                        is_product_page = any(keyword in url.lower() for keyword in product_keywords_in_url)
                        has_product_content = any(keyword in cleaned_text.lower() for keyword in product_keywords_in_text)
                        
                        if is_product_page:
                            context_str += f"Source {valid_chunks_used+1} ({url}) [PRODUCT/SERVICE PAGE - HIGHEST PRIORITY]:\n{cleaned_text[:5000]}...\n\n"
                        elif has_product_content:
                            context_str += f"Source {valid_chunks_used+1} ({url}) [CONTAINS PRODUCT INFO]:\n{cleaned_text[:4000]}...\n\n"
                        else:
                            context_str += f"Source {valid_chunks_used+1} ({url}):\n{cleaned_text[:text_limit]}...\n\n"
                    else:
                        context_str += f"Source {valid_chunks_used+1} ({url}):\n{cleaned_text[:text_limit]}...\n\n"
                    source_urls.add(url)
                    valid_chunks_used += 1
                
                max_chunks = 15 if is_about_query else 12 if is_product_query else 10 if is_address_query else 8
                if valid_chunks_used >= max_chunks:
                    break
            
            context_str += "--- End Context ---"
        else:
            context_str = "No relevant content found."
            source_urls = set()
        
        # Build prompt
        special_instruction = ""
        if is_address_query:
            special_instruction = """
The user is asking about address or location. Look carefully through the website content for:
- Street addresses, building names/numbers
- Cities, states, countries
- Postal codes
- Any location information

If multiple locations exist (e.g., India office, UAE office), mention all of them using first person. Say "We have offices at..." or "Our offices are located at..." NOT "They have offices at..." or "They also have...". Include all address details you find, even if incomplete.
"""
        elif is_about_query:
            special_instruction = """
The user is asking about the website/organization/topic. Provide a comprehensive description including:
- What it is about or what it does
- Mission, vision, values (if applicable)
- History and background
- Services, products, or offerings (if applicable)
- Key information, features, or highlights
- Achievements and milestones (if applicable)
- Any other relevant details

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
        
        is_greeting = user_input.lower().strip() in ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
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
7. **Ignore error messages** - Skip any JavaScript errors, HTML errors, or placeholder text in the context.
8. **If asked a greeting** (hi, hello), respond warmly and offer to help{company_info}

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
            if is_greeting:
                answer = f"Hello! I'm here to help you.{company_info} What would you like to know?"
            else:
                answer_parts = []
                for url, text in relevant_chunks[:5]:
                    cleaned = clean_text_content(text[:800].strip())
                    if is_valid_content(cleaned, min_length=100):
                        answer_parts.append(cleaned)
                
                if answer_parts:
                    answer = "\n\n".join(answer_parts)
        
        return answer, list(source_urls), token_usage
        
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

