"""
Scraper Module
Wrapper around deep_scraper for scraping websites
"""

import os
from deep_scraper import crawl_website
from data_manager import save_embeddings_data, save_structured_embeddings_data
from custom_data import get_custom_data
import re


def clean_text_content(text):
    """Remove common error messages, HTML noise, and unwanted content from text."""
    if not text:
        return text
    
    error_patterns = [
        "You need to enable JavaScript to run this app",
        "Please enable JavaScript",
        "JavaScript is disabled",
        "Enable JavaScript",
    ]
    
    cleaned = text
    for pattern in error_patterns:
        cleaned = re.sub(re.escape(pattern), "", cleaned, flags=re.IGNORECASE)
    
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
    cleaned = re.sub(r' +', ' ', cleaned)
    
    return cleaned.strip()


def is_valid_content(text, min_length=50):
    """Check if text is valid content (not just error messages or too short)."""
    if not text or len(text.strip()) < min_length:
        return False
    
    error_keywords = ["you need to enable javascript", "enable javascript", "javascript is disabled"]
    text_lower = text.lower()
    error_count = sum(1 for keyword in error_keywords if keyword in text_lower)
    
    words = text_lower.split()
    if len(words) < 10:
        if error_count > 0:
            return False
    
    cleaned = clean_text_content(text)
    if len(cleaned.strip()) < min_length:
        return False
    
    return True


def run_scraper(target_website, embedder, max_pages=200):
    """
    Run the scraper and process the results
    
    Returns:
        tuple: (scraped_data_pages, scraped_data_embeddings, structured_data_texts, structured_data_embeddings, structured_data_sources)
    """
    print("Starting web scraping...")
    
    if not target_website:
        print("❌ No target website set. Please set a website first.")
        return [], None, [], None, []
    
    print(f"Scraping website: {target_website}")
    print(f"Maximum pages to scrape: {max_pages}")
    
    try:
        general_scraped_data, structured_scraped_data = crawl_website(target_website, max_pages=max_pages)
        
        # Process general scraped data
        url_text_pairs = []
        for url, content in general_scraped_data.items():
            cleaned_text = clean_text_content(content.strip())
            
            if is_valid_content(cleaned_text, min_length=50):
                url_text_pairs.append((url.strip(), cleaned_text))
        
        if not url_text_pairs:
            print("⚠️ No valid pages scraped")
            return [], None, [], None, []
        
        print(f"✅ Processed {len(url_text_pairs)} valid pages")
        
        # Create embeddings
        scraped_data_embeddings = None
        if embedder:
            texts = [text for _, text in url_text_pairs]
            print(f"Creating embeddings for {len(texts)} pages...")
            scraped_data_embeddings = embedder.encode(texts, convert_to_tensor=True)
            
            # Save embeddings and metadata
            save_embeddings_data(url_text_pairs, scraped_data_embeddings)
            
            print(f"Created and saved embeddings for {len(url_text_pairs)} pages")
        else:
            print("Embedder not available, cannot create embeddings")
        
        # Process structured data
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
        structured_data_embeddings = None
        if embedder and structured_data_texts:
            print(f"🔄 Creating embeddings for {len(structured_data_texts)} structured entries...")
            structured_data_embeddings = embedder.encode(structured_data_texts, convert_to_tensor=True)
            save_structured_embeddings_data(structured_data_texts, structured_data_sources, structured_data_embeddings)
            print(f"✅ Created and saved structured data embeddings")
        
        print("✅ Web scraping completed successfully! Data stored as embeddings.")
        return url_text_pairs, scraped_data_embeddings, structured_data_texts, structured_data_embeddings, structured_data_sources
        
    except Exception as e:
        print(f"Error during scraping: {e}")
        import traceback
        traceback.print_exc()
        return [], None, [], None, []

