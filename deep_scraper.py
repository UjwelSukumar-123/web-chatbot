import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import re 

visited = set()
session = requests.Session()

def is_internal(base_url, link):
    """Check if a link is internal or a subdomain."""
    base_domain = urlparse(base_url).netloc
    link_domain = urlparse(link).netloc
    return link_domain == base_domain or link_domain.endswith('.' + base_domain)

def normalize_url(base_url, link):
    """Return a clean full URL from a possibly relative link."""
    
    full_url = urljoin(base_url, link)
    return full_url.split('#')[0]

def extract_contact_info(soup):
    """
    Extracts potential contact numbers and email addresses using regex.
    This is a heuristic and might require refinement for specific website structures.
    """
    contact_info = {}
    page_text = soup.get_text(" ", strip=True)

    phone_numbers = re.findall(r'(?:\+91[\s-]?)?\d{10}(?:[,\s]|\b)', page_text)
    
    if phone_numbers:
        
        cleaned_numbers = []
        for num in phone_numbers:
            
            clean_num = re.sub(r'[^0-9+]', '', num)
            if clean_num not in cleaned_numbers:
                cleaned_numbers.append(clean_num)
        if cleaned_numbers:
            contact_info['phone_numbers'] = list(set(cleaned_numbers)) # Use set to ensure uniqueness

    
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', page_text)
    if emails:
        contact_info['emails'] = list(set(emails))

    return contact_info

def extract_job_opportunities(soup):
    """
    Extracts job opportunity titles/summaries.
    This is highly dependent on the website's HTML structure for job listings.
    You will likely need to inspect the HTML of inciem.com's careers/jobs page.
    """
    jobs = []
    job_section = soup.find('div', class_='careers-section') # Adjust this class name
    if job_section:
        job_titles = job_section.find_all(['h2', 'h3', 'h4'], string=re.compile(r'job|opening|position|vacancy', re.IGNORECASE))
        for title_tag in job_titles:
            job_text = title_tag.get_text(strip=True)
            
            jobs.append(job_text)
    
    potential_job_elements = soup.find_all(re.compile(r'h[1-6]|div|p|li'), class_=re.compile(r'job|career|opening|position|vacancy', re.IGNORECASE))
    for elem in potential_job_elements:
        text = elem.get_text(strip=True)
        if len(text) > 10 and len(text) < 200: 
            if any(keyword in text.lower() for keyword in ['job', 'opening', 'position', 'vacancy', 'hiring']):
                jobs.append(text)

    return list(set(jobs))


def crawl_website(base_url, max_pages=100):
    to_visit = [base_url]
    data_collected = {} # Stores general page content
    specific_data_collected = { # Stores structured data like contacts, jobs
        'contact_info': [],
        'job_opportunities': []
    }

    
    if 'contact' not in base_url and 'about' not in base_url:
        to_visit.append(urljoin(base_url, '/contact-us/')) 
        to_visit.append(urljoin(base_url, '/contact/'))
    if 'career' not in base_url and 'job' not in base_url:
        to_visit.append(urljoin(base_url, '/careers/')) 
        to_visit.append(urljoin(base_url, '/jobs/'))
        to_visit.append(urljoin(base_url, '/opportunities/'))


    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        url = normalize_url(base_url, url)

        if url in visited:
            continue
        visited.add(url)

        print(f"[+] Crawling: {url}")
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()

           
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type.lower(): # Use .lower() for robustness
                print(f"[-] Skipping non-HTML content at {url}: {content_type}")
                continue

        except requests.exceptions.RequestException as e:
            print(f"[!] Failed to access {url}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        for script_or_style in soup(["script", "style", "header", "footer", "nav"]):
            script_or_style.extract()
        text = soup.get_text(separator=' ', strip=True)
        data_collected[url] = text

        # --- Specific Data Extraction ---
        contact_info = extract_contact_info(soup)
        if contact_info:
            specific_data_collected['contact_info'].append({'url': url, 'data': contact_info})

        job_opportunities = extract_job_opportunities(soup)
        if job_opportunities:
            specific_data_collected['job_opportunities'].append({'url': url, 'data': job_opportunities})


        # Extract all <a href=""> links
        for link_tag in soup.find_all("a", href=True):
            raw_link = link_tag['href']
            full_url = normalize_url(url, raw_link)
            if is_internal(base_url, full_url) and full_url not in visited:
                
                if full_url != url: 
                    to_visit.append(full_url)

        time.sleep(0.5)  # Be polite

    return data_collected, specific_data_collected #


if __name__ == "__main__":
    website = "https://inciem.com"
    print(f"Starting crawl for: {website}")
    general_scraped_data, structured_scraped_data = crawl_website(website, max_pages=50) # Reduced max_pages for testing

    # Save general scraped data for the chatbot
    output_general_file = "scraped_data.txt"
    with open(output_general_file, "w", encoding="utf-8") as f:
        for url, content in general_scraped_data.items():
            f.write(f"--- {url} ---\n")
            f.write(content + "\n\n")
    print(f"General scraped data saved to {output_general_file}")

    # Save specific structured data to a separate file (or database)
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