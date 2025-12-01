from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import re
import requests
from xml.etree import ElementTree as ET 

# Try to use webdriver-manager for automatic ChromeDriver management
try:
    from webdriver_manager.chrome import ChromeDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False
    print("[!] webdriver-manager not installed. Install it with: pip install webdriver-manager")

# Try to use Playwright as a fallback
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("[!] Playwright not installed. Install it with: pip install playwright && playwright install chromium")

visited = set()
driver = None
playwright_browser = None
playwright_context = None

def discover_links_from_sitemap(base_url):
    """Discover links from sitemap.xml if available."""
    sitemap_urls = []
    sitemap_paths = [
        '/sitemap.xml',
        '/sitemap_index.xml',
        '/sitemap-index.xml',
        '/sitemap1.xml'
    ]
    
    for path in sitemap_paths:
        try:
            sitemap_url = urljoin(base_url, path)
            response = requests.get(sitemap_url, timeout=10)
            if response.status_code == 200:
                print(f"[+] Found sitemap at {sitemap_url}")
                try:
                    root = ET.fromstring(response.content)
                    # Handle sitemap index
                    if root.tag.endswith('sitemapindex'):
                        for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                            loc = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                            if loc is not None:
                                sitemap_urls.append(loc.text)
                    # Handle regular sitemap
                    elif root.tag.endswith('urlset'):
                        for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                            loc = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                            if loc is not None:
                                sitemap_urls.append(loc.text)
                except ET.ParseError:
                    # Try parsing as plain text sitemap
                    for line in response.text.split('\n'):
                        if '<loc>' in line:
                            url = line.split('<loc>')[1].split('</loc>')[0].strip()
                            if url:
                                sitemap_urls.append(url)
                break
        except Exception as e:
            continue
    
    return sitemap_urls

def discover_links_from_robots(base_url):
    """Discover links from robots.txt if available."""
    robots_urls = []
    try:
        robots_url = urljoin(base_url, '/robots.txt')
        response = requests.get(robots_url, timeout=10)
        if response.status_code == 200:
            print(f"[+] Found robots.txt at {robots_url}")
            for line in response.text.split('\n'):
                if line.lower().startswith('sitemap:'):
                    sitemap_url = line.split(':', 1)[1].strip()
                    robots_urls.append(sitemap_url)
    except Exception as e:
        pass
    
    return robots_urls

def get_common_paths(base_url):
    """Get common website paths to check."""
    common_paths = [
        '/about', '/about-us', '/aboutus', '/company', '/our-story', '/story',
        '/team', '/our-team', '/leadership', '/people',
        '/services', '/service', '/products', '/product', '/solutions', '/solution',
        '/portfolio', '/projects', '/work', '/case-studies', '/cases',
        '/blog', '/news', '/articles', '/updates', '/press', '/media',
        '/careers', '/jobs', '/opportunities', '/hiring', '/join-us', '/work-with-us',
        '/contact', '/contact-us', '/contactus', '/get-in-touch', '/reach-us',
        '/career', '/job', '/life', '/culture', '/values', '/mission', '/vision',
        '/culture', '/company-life', '/life-at-company', '/work-life', '/employee-life',
        '/faq', '/faqs', '/help', '/support', '/terms', '/privacy', '/policy',
        '/testimonials', '/reviews', '/clients', '/customers', '/partners',
        '/locations', '/offices', '/address', '/where-we-are'
    ]
    
    urls = []
    for path in common_paths:
        urls.append(urljoin(base_url, path))
        # Also try with trailing slash
        if not path.endswith('/'):
            urls.append(urljoin(base_url, path + '/'))
    
    return urls

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
    Works generically for most websites with common job listing patterns.
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

def extract_page_metadata(soup, url):
    """Extract page metadata including title, description, keywords, and Open Graph tags."""
    metadata = {
        'url': url,
        'title': '',
        'description': '',
        'keywords': '',
        'og_title': '',
        'og_description': '',
        'og_image': '',
        'canonical': ''
    }
    
    # Title
    title_tag = soup.find('title')
    if title_tag:
        metadata['title'] = title_tag.get_text(strip=True)
    
    # Meta description
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc and meta_desc.get('content'):
        metadata['description'] = meta_desc.get('content', '').strip()
    
    # Meta keywords
    meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
    if meta_keywords and meta_keywords.get('content'):
        metadata['keywords'] = meta_keywords.get('content', '').strip()
    
    # Open Graph tags
    og_title = soup.find('meta', property='og:title')
    if og_title and og_title.get('content'):
        metadata['og_title'] = og_title.get('content', '').strip()
    
    og_desc = soup.find('meta', property='og:description')
    if og_desc and og_desc.get('content'):
        metadata['og_description'] = og_desc.get('content', '').strip()
    
    og_img = soup.find('meta', property='og:image')
    if og_img and og_img.get('content'):
        metadata['og_image'] = og_img.get('content', '').strip()
    
    # Canonical URL
    canonical = soup.find('link', rel='canonical')
    if canonical and canonical.get('href'):
        metadata['canonical'] = canonical.get('href', '').strip()
    
    return metadata

def extract_headings_structure(soup):
    """Extract headings with their hierarchy and context."""
    headings = []
    for level in range(1, 7):
        for heading in soup.find_all(f'h{level}'):
            text = heading.get_text(strip=True)
            if text and len(text) > 0:
                # Get parent section context if available
                parent_section = heading.find_parent(['section', 'article', 'div'], class_=True)
                section_class = parent_section.get('class', []) if parent_section else []
                headings.append({
                    'level': level,
                    'text': text,
                    'section': ' '.join(section_class) if section_class else ''
                })
    return headings

def extract_images_with_context(soup, base_url):
    """Extract images with alt text, src, and surrounding context."""
    images = []
    for img in soup.find_all('img'):
        img_data = {
            'src': '',
            'alt': '',
            'title': '',
            'context': ''
        }
        
        # Source URL
        src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
        if src:
            img_data['src'] = urljoin(base_url, src)
        
        # Alt text
        img_data['alt'] = img.get('alt', '').strip()
        
        # Title
        img_data['title'] = img.get('title', '').strip()
        
        # Context - get parent section or nearby text
        parent = img.find_parent(['section', 'article', 'div', 'figure'])
        if parent:
            # Get nearby heading or paragraph
            nearby_heading = parent.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if nearby_heading:
                img_data['context'] = nearby_heading.get_text(strip=True)
            else:
                nearby_p = parent.find('p')
                if nearby_p:
                    img_data['context'] = nearby_p.get_text(strip=True)[:200]
        
        if img_data['src'] or img_data['alt']:
            images.append(img_data)
    
    return images

def extract_links_with_context(soup, base_url):
    """Extract links with their text, URLs, and context."""
    links = []
    for link in soup.find_all('a', href=True):
        link_data = {
            'text': '',
            'url': '',
            'title': '',
            'is_external': False,
            'context': ''
        }
        
        # Link text
        link_data['text'] = link.get_text(strip=True)
        
        # URL
        href = link.get('href', '')
        if href:
            full_url = urljoin(base_url, href)
            link_data['url'] = full_url
            link_data['is_external'] = not is_internal(base_url, full_url)
        
        # Title attribute
        link_data['title'] = link.get('title', '').strip()
        
        # Context - get parent section
        parent = link.find_parent(['section', 'article', 'div', 'nav', 'li'])
        if parent:
            parent_heading = parent.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if parent_heading:
                link_data['context'] = parent_heading.get_text(strip=True)
        
        if link_data['url']:
            links.append(link_data)
    
    return links

def extract_social_media_links(soup, base_url):
    """Extract social media links from the page."""
    social_platforms = {
        'facebook': ['facebook.com', 'fb.com'],
        'twitter': ['twitter.com', 'x.com'],
        'linkedin': ['linkedin.com'],
        'instagram': ['instagram.com'],
        'youtube': ['youtube.com', 'youtu.be'],
        'github': ['github.com'],
        'pinterest': ['pinterest.com']
    }
    
    social_links = {}
    for link in soup.find_all('a', href=True):
        href = link.get('href', '').lower()
        if not href:
            continue
        
        for platform, domains in social_platforms.items():
            if any(domain in href for domain in domains):
                full_url = urljoin(base_url, link.get('href'))
                if platform not in social_links:
                    social_links[platform] = []
                if full_url not in social_links[platform]:
                    social_links[platform].append(full_url)
    
    return social_links

def extract_addresses(soup):
    """Extract addresses from the page using common patterns."""
    addresses = []
    page_text = soup.get_text(" ", strip=True)
    
    # Look for address patterns (street, city, state, zip/postal code)
    # Common patterns: "Street, City, State ZIP" or "Street, City, Country"
    address_patterns = [
        r'\d+[\s\w]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Circle|Ct)[,\s]+[\w\s]+(?:,\s*)?[\w\s]+(?:,\s*)?(?:\d{5,6})?',
        r'[\w\s]+(?:,\s*)?[\w\s]+(?:,\s*)?(?:Kerala|India|UAE|United Kingdom|UK|Dubai|Kochi)[,\s]*(?:\d{5,6})?'
    ]
    
    for pattern in address_patterns:
        matches = re.findall(pattern, page_text, re.IGNORECASE)
        for match in matches:
            cleaned = ' '.join(match.split())
            if len(cleaned) > 10 and cleaned not in addresses:
                addresses.append(cleaned)
    
    # Also look for structured address elements
    address_elements = soup.find_all(['address', 'div', 'p'], class_=re.compile(r'address|location|office', re.IGNORECASE))
    for elem in address_elements:
        text = elem.get_text(separator=' ', strip=True)
        if len(text) > 20 and any(keyword in text.lower() for keyword in ['street', 'road', 'avenue', 'floor', 'building', 'office']):
            if text not in addresses:
                addresses.append(text)
    
    return addresses

def extract_services_products(soup):
    """Extract services and products with descriptions - comprehensive extraction."""
    services = []
    seen_services = set()  # Avoid duplicates
    
    # Look for common service/product indicators
    service_keywords = ['service', 'product', 'solution', 'offering', 'feature', 'capability', 'expertise']
    
    # Method 1: Find sections that might contain services/products
    for section in soup.find_all(['section', 'div', 'article'], class_=re.compile(r'service|product|solution|offering|feature|capability', re.IGNORECASE)):
        heading = section.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        description = section.find(['p', 'div'], class_=re.compile(r'description|content|text|detail', re.IGNORECASE))
        
        if heading:
            service_name = heading.get_text(strip=True)
            if service_name and service_name.lower() not in seen_services:
                service_data = {
                    'name': service_name,
                    'description': ''
                }
                
                if description:
                    service_data['description'] = description.get_text(strip=True)[:800]  # Increased from 500
                else:
                    # Get all paragraphs after heading
                    paragraphs = []
                    for p in heading.find_all_next(['p', 'div'], limit=5):
                        p_text = p.get_text(strip=True)
                        if len(p_text) > 20:
                            paragraphs.append(p_text)
                    if paragraphs:
                        service_data['description'] = ' '.join(paragraphs)[:800]
                
                if service_data['name']:
                    services.append(service_data)
                    seen_services.add(service_name.lower())
    
    # Method 2: Look for list items that might be services (more comprehensive)
    for li in soup.find_all('li'):
        text = li.get_text(strip=True)
        if not text or len(text) < 5:
            continue
            
        parent = li.find_parent(['ul', 'ol', 'div', 'section'])
        if parent:
            parent_class = ' '.join(parent.get('class', []))
            parent_id = parent.get('id', '')
            parent_text = parent.get_text(strip=True).lower()
            
            # Check if parent is service/product related
            is_service_parent = (
                any(keyword in parent_class.lower() for keyword in service_keywords) or
                any(keyword in parent_id.lower() for keyword in service_keywords) or
                any(keyword in parent_text[:200] for keyword in ['service', 'product', 'offering', 'solution'])
            )
            
            if is_service_parent and len(text) > 5 and len(text) < 500:
                # Try to get description from next sibling or parent
                description = ''
                next_sibling = li.find_next_sibling(['p', 'div', 'li'])
                if next_sibling:
                    desc_text = next_sibling.get_text(strip=True)
                    if len(desc_text) > 20 and len(desc_text) < 300:
                        description = desc_text
                
                service_name = text
                if service_name.lower() not in seen_services:
                    services.append({
                        'name': service_name,
                        'description': description
                    })
                    seen_services.add(service_name.lower())
    
    # Method 3: Look for cards/grid items that might be services
    for card in soup.find_all(['div', 'article', 'section'], class_=re.compile(r'card|item|grid-item|service-card|product-card', re.IGNORECASE)):
        heading = card.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if heading:
            service_name = heading.get_text(strip=True)
            if service_name and service_name.lower() not in seen_services:
                # Get description from card
                desc_elem = card.find(['p', 'div'], class_=re.compile(r'description|content|text|detail|summary', re.IGNORECASE))
                description = desc_elem.get_text(strip=True)[:800] if desc_elem else ''
                
                if not description:
                    # Try to get any paragraph in the card
                    p_elem = card.find('p')
                    if p_elem:
                        description = p_elem.get_text(strip=True)[:800]
                
                services.append({
                    'name': service_name,
                    'description': description
                })
                seen_services.add(service_name.lower())
    
    # Method 4: Extract from headings that might be service names
    for heading in soup.find_all(['h2', 'h3', 'h4'], class_=re.compile(r'service|product|solution', re.IGNORECASE)):
        service_name = heading.get_text(strip=True)
        if service_name and len(service_name) > 3 and service_name.lower() not in seen_services:
            # Get description from next elements
            description = ''
            for next_elem in heading.find_all_next(['p', 'div'], limit=3):
                desc_text = next_elem.get_text(strip=True)
                if len(desc_text) > 20:
                    description += desc_text + ' '
                    if len(description) > 500:
                        break
            
            services.append({
                'name': service_name,
                'description': description.strip()[:800]
            })
            seen_services.add(service_name.lower())
    
    return services

def extract_testimonials(soup):
    """Extract testimonials/reviews from the page."""
    testimonials = []
    
    # Look for testimonial sections
    testimonial_sections = soup.find_all(['div', 'section', 'blockquote'], 
                                         class_=re.compile(r'testimonial|review|quote|feedback', re.IGNORECASE))
    
    for section in testimonial_sections:
        testimonial_data = {
            'text': '',
            'author': '',
            'role': ''
        }
        
        # Get testimonial text
        text_elem = section.find(['p', 'div', 'blockquote'])
        if text_elem:
            testimonial_data['text'] = text_elem.get_text(strip=True)
        
        # Get author name
        author_elem = section.find(['span', 'div', 'p'], class_=re.compile(r'author|name|person', re.IGNORECASE))
        if author_elem:
            testimonial_data['author'] = author_elem.get_text(strip=True)
        
        # Get role/company
        role_elem = section.find(['span', 'div', 'p'], class_=re.compile(r'role|position|company|title', re.IGNORECASE))
        if role_elem:
            testimonial_data['role'] = role_elem.get_text(strip=True)
        
        if testimonial_data['text']:
            testimonials.append(testimonial_data)
    
    return testimonials

def extract_tables(soup):
    """Extract data from HTML tables."""
    tables_data = []
    
    for table in soup.find_all('table'):
        table_data = {
            'headers': [],
            'rows': []
        }
        
        # Extract headers
        headers = table.find_all('th')
        if headers:
            table_data['headers'] = [th.get_text(strip=True) for th in headers]
        else:
            # Try first row as headers
            first_row = table.find('tr')
            if first_row:
                cells = first_row.find_all(['td', 'th'])
                table_data['headers'] = [cell.get_text(strip=True) for cell in cells]
        
        # Extract rows
        rows = table.find_all('tr')
        for row in rows[1:] if table_data['headers'] else rows:  # Skip header row if headers found
            cells = row.find_all(['td', 'th'])
            if cells:
                row_data = [cell.get_text(strip=True) for cell in cells]
                if any(cell.strip() for cell in row_data):  # Only add non-empty rows
                    table_data['rows'].append(row_data)
        
        if table_data['headers'] or table_data['rows']:
            tables_data.append(table_data)
    
    return tables_data

def extract_company_life_culture(soup):
    """Extract company life, culture, values, mission, vision content."""
    life_culture_content = []
    
    # Look for sections about company life, culture, values
    culture_keywords = ['culture', 'life', 'values', 'mission', 'vision', 'philosophy', 
                       'work-life', 'employee', 'team', 'people', 'environment', 
                       'atmosphere', 'workplace', 'community', 'family']
    
    # Find sections with culture-related classes or IDs
    for section in soup.find_all(['section', 'div', 'article'], 
                                 class_=re.compile('|'.join(culture_keywords), re.IGNORECASE)):
        text = section.get_text(separator='\n', strip=True)
        if len(text) > 50:
            life_culture_content.append(text)
    
    # Find headings related to culture/life
    for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        heading_text = heading.get_text(strip=True).lower()
        if any(keyword in heading_text for keyword in culture_keywords):
            # Get the content after this heading
            next_siblings = []
            for sibling in heading.next_siblings:
                if sibling.name in ['p', 'div', 'section', 'ul', 'ol']:
                    sibling_text = sibling.get_text(separator=' ', strip=True)
                    if len(sibling_text) > 20:
                        next_siblings.append(sibling_text)
                if len(next_siblings) >= 5:  # Limit to 5 paragraphs
                    break
            if next_siblings:
                life_culture_content.append(f"{heading.get_text(strip=True)}: {' '.join(next_siblings)}")
    
    return '\n\n'.join(life_culture_content[:10])  # Limit to first 10 sections

def format_detailed_content(soup, url, base_url):
    """Format detailed content with all extracted information."""
    detailed_content = []
    
    # Page metadata
    metadata = extract_page_metadata(soup, url)
    if metadata['title']:
        detailed_content.append(f"PAGE TITLE: {metadata['title']}")
    if metadata['description']:
        detailed_content.append(f"PAGE DESCRIPTION: {metadata['description']}")
    if metadata['og_title']:
        detailed_content.append(f"OPEN GRAPH TITLE: {metadata['og_title']}")
    if metadata['og_description']:
        detailed_content.append(f"OPEN GRAPH DESCRIPTION: {metadata['og_description']}")
    
    # Headings structure
    headings = extract_headings_structure(soup)
    if headings:
        detailed_content.append("\n=== PAGE STRUCTURE (HEADINGS) ===")
        for heading in headings[:30]:  # Increased from 20 to 30
            indent = "  " * (heading['level'] - 1)
            detailed_content.append(f"{indent}H{heading['level']}: {heading['text']}")
    
    # Company Life and Culture (NEW)
    life_culture = extract_company_life_culture(soup)
    if life_culture:
        detailed_content.append("\n=== COMPANY LIFE & CULTURE ===")
        detailed_content.append(life_culture)
    
    # Main content sections - extract more thoroughly
    detailed_content.append("\n=== MAIN CONTENT ===")
    
    # Extract content from main semantic HTML5 elements
    main_content_parts = []
    
    # Priority 1: Get content from main, article, section elements (preserve structure)
    for tag in soup.find_all(['main', 'article', 'section']):
        # Get text with better structure preservation
        tag_text = tag.get_text(separator='\n', strip=True)
        if len(tag_text) > 100:
            main_content_parts.append(tag_text)
    
    # Priority 2: If no semantic elements, get content from divs with content classes
    if not main_content_parts:
        for div in soup.find_all('div', class_=re.compile(r'content|main|body|text|description', re.IGNORECASE)):
            div_text = div.get_text(separator='\n', strip=True)
            if len(div_text) > 200:  # Only substantial divs
                main_content_parts.append(div_text)
    
    # Priority 3: If still no content, get all text
    if not main_content_parts:
        main_content = soup.get_text(separator='\n', strip=True)
        main_content_parts.append(main_content)
    
    # Combine all content parts (limit to avoid too much content)
    combined_content = '\n\n'.join(main_content_parts[:10])  # Limit to first 10 sections
    detailed_content.append(combined_content)
    
    # Also extract any tables that might contain product/service information
    tables = extract_tables(soup)
    if tables:
        detailed_content.append("\n=== TABLES (May contain product/service data) ===")
        for table_idx, table in enumerate(tables[:5], 1):  # Limit to first 5 tables
            if table['headers']:
                detailed_content.append(f"\nTable {table_idx} Headers: {', '.join(table['headers'])}")
            if table['rows']:
                for row_idx, row in enumerate(table['rows'][:10], 1):  # First 10 rows
                    detailed_content.append(f"  Row {row_idx}: {' | '.join(row[:5])}")  # First 5 columns
    
    # Services/Products - Enhanced extraction and formatting
    services = extract_services_products(soup)
    if services:
        detailed_content.append("\n=== SERVICES/PRODUCTS (COMPREHENSIVE LIST) ===")
        for idx, service in enumerate(services[:30], 1):  # Increased from 15 to 30
            detailed_content.append(f"\n{idx}. {service['name']}")
            if service['description']:
                detailed_content.append(f"   {service['description']}")
        detailed_content.append(f"\n[Total services/products found: {len(services)}]")
    
    # Also extract any tables that might contain product/service information
    tables = extract_tables(soup)
    if tables:
        detailed_content.append("\n=== TABLES (May contain product/service data) ===")
        for table_idx, table in enumerate(tables[:5], 1):  # Limit to first 5 tables
            if table['headers']:
                detailed_content.append(f"\nTable {table_idx} Headers: {', '.join(table['headers'])}")
            if table['rows']:
                for row_idx, row in enumerate(table['rows'][:10], 1):  # First 10 rows
                    detailed_content.append(f"  Row {row_idx}: {' | '.join(row[:5])}")  # First 5 columns
    
    # Contact information
    contact_info = extract_contact_info(soup)
    if contact_info:
        detailed_content.append("\n=== CONTACT INFORMATION ===")
        if 'phone_numbers' in contact_info:
            detailed_content.append(f"Phone Numbers: {', '.join(contact_info['phone_numbers'])}")
        if 'emails' in contact_info:
            detailed_content.append(f"Email Addresses: {', '.join(contact_info['emails'])}")
    
    # Addresses
    addresses = extract_addresses(soup)
    if addresses:
        detailed_content.append("\n=== ADDRESSES ===")
        for addr in addresses[:5]:  # Limit to first 5
            detailed_content.append(f"Address: {addr}")
    
    # Social media links
    social_links = extract_social_media_links(soup, base_url)
    if social_links:
        detailed_content.append("\n=== SOCIAL MEDIA LINKS ===")
        for platform, urls in social_links.items():
            detailed_content.append(f"{platform.title()}: {', '.join(urls)}")
    
    # Testimonials
    testimonials = extract_testimonials(soup)
    if testimonials:
        detailed_content.append("\n=== TESTIMONIALS/REVIEWS ===")
        for testimonial in testimonials[:10]:  # Limit to first 10
            detailed_content.append(f"Testimonial: {testimonial['text'][:300]}")
            if testimonial['author']:
                detailed_content.append(f"  - {testimonial['author']}")
            if testimonial['role']:
                detailed_content.append(f"  {testimonial['role']}")
    
    # Important images (with alt text)
    images = extract_images_with_context(soup, base_url)
    important_images = [img for img in images if img['alt'] and len(img['alt']) > 10]
    if important_images:
        detailed_content.append("\n=== KEY IMAGES ===")
        for img in important_images[:10]:  # Limit to first 10
            detailed_content.append(f"Image: {img['alt']}")
            if img['context']:
                detailed_content.append(f"  Context: {img['context']}")
    
    return '\n'.join(detailed_content)


def get_page_content_with_playwright(url, page, wait_time=30):
    """Fetch page content using Playwright to handle JavaScript rendering."""
    try:
        print(f"    Loading {url} with Playwright...")
        page.goto(url, wait_until="networkidle", timeout=wait_time * 1000)
        
        # Scroll down to trigger lazy loading
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(2000)
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(1000)
        
        # Wait for React content to load - increased wait time
        max_wait = 15  # Increased from 10
        wait_interval = 1
        waited = 0
        best_content = None
        best_text_length = 0
        
        while waited < max_wait:
            content = page.content()
            
            # Check if React content has loaded
            if "You need to enable JavaScript to run this app" not in content:
                # Additional wait for dynamic content
                page.wait_for_timeout(3000)  # Increased from 2000
                content = page.content()
                soup_check = BeautifulSoup(content, "html.parser")
                for tag in soup_check(["script", "style", "noscript"]):
                    tag.decompose()
                text_check = soup_check.get_text(separator=' ', strip=True)
                if len(text_check) > best_text_length:
                    best_content = content
                    best_text_length = len(text_check)
                if len(text_check) > 100:
                    print(f"    [OK] Content loaded ({len(text_check)} chars)")
                    return content
            
            # Check for meaningful content
            soup_temp = BeautifulSoup(content, "html.parser")
            for tag in soup_temp(["script", "style", "noscript"]):
                tag.decompose()
            text_temp = soup_temp.get_text(separator=' ', strip=True)
            
            if len(text_temp) > best_text_length:
                best_content = content
                best_text_length = len(text_temp)
            
            if len(text_temp) > 100 and "You need to enable JavaScript to run this app" not in text_temp:
                print(f"    [OK] Content found ({len(text_temp)} chars)")
                return content
            
            page.wait_for_timeout(wait_interval * 1000)
            waited += wait_interval
        
        # Final check - scroll and wait more
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(3000)
        content = page.content()
        soup_final = BeautifulSoup(content, "html.parser")
        for tag in soup_final(["script", "style", "noscript"]):
            tag.decompose()
        text_final = soup_final.get_text(separator=' ', strip=True)
        
        # Use best content if available
        if best_text_length > len(text_final):
            content = best_content
            text_final = soup_final.get_text(separator=' ', strip=True) if best_content else text_final
        
        if len(text_final) > 50:
            print(f"    [OK] Content extracted ({len(text_final)} chars)")
            return content
        else:
            print(f"    [WARN] Limited content extracted ({len(text_final)} chars)")
            return content
        
    except Exception as e:
        print(f"    [!] Playwright error for {url}: {e}")
        try:
            return page.content()
        except:
            return None

def get_page_content_with_selenium(url, driver, wait_time=40):
    """Fetch page content using Selenium to handle JavaScript rendering."""
    try:
        print(f"    Loading {url}...")
        driver.get(url)
        
        # Wait for document ready state
        WebDriverWait(driver, wait_time).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        
        # Scroll to trigger lazy loading
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)
        
        # Wait longer for React to hydrate
        print("    Waiting for React content to render...")
        time.sleep(5)  # Initial wait for React
        
        # Wait for React to hydrate - check if placeholder text is gone
        max_wait = 25  # Increased from 20
        wait_interval = 2
        waited = 0
        best_content = None
        best_text_length = 0
        
        while waited < max_wait:
            page_source = driver.page_source
            soup_temp = BeautifulSoup(page_source, "html.parser")
            
            # Remove scripts and styles for text extraction
            for tag in soup_temp(["script", "style", "noscript"]):
                tag.decompose()
            
            text_temp = soup_temp.get_text(separator=' ', strip=True)
            text_length = len(text_temp)
            
            # Track the best content we've seen
            if text_length > best_text_length:
                best_content = page_source
                best_text_length = text_length
            
            # Check if React content has loaded (placeholder should be gone)
            has_placeholder = "You need to enable JavaScript to run this app" in text_temp
            
            if not has_placeholder and text_length > 100:
                # Good content found!
                print(f"    ✓ Content loaded ({text_length} chars)")
                time.sleep(2)  # Increased wait time
                return driver.page_source
            
            if text_length > 200 and not has_placeholder:
                # Even better content
                print(f"    ✓ Rich content found ({text_length} chars)")
                time.sleep(2)
                return driver.page_source
            
            # Scroll again to trigger more content loading
            if waited % 6 == 0:  # Scroll every 6 seconds
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
            
            print(f"    Waiting... ({waited}/{max_wait}s, {text_length} chars, placeholder: {has_placeholder})")
            time.sleep(wait_interval)
            waited += wait_interval
        
        # Final scroll and check - use the best content we found, or current if it's decent
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
        final_source = driver.page_source
        soup_final = BeautifulSoup(final_source, "html.parser")
        for tag in soup_final(["script", "style", "noscript"]):
            tag.decompose()
        text_final = soup_final.get_text(separator=' ', strip=True)
        final_length = len(text_final)
        
        # Use the best content we found
        if best_text_length > final_length and best_content:
            final_source = best_content
            final_length = best_text_length
        
        has_placeholder_final = "You need to enable JavaScript to run this app" in text_final
        
        if final_length > 50 and not has_placeholder_final:
            print(f"    ✓ Content extracted ({final_length} chars)")
            return final_source
        elif final_length > 200:
            # Even if placeholder is there, if we have a lot of content, return it
            print(f"    ⚠ Content extracted with possible placeholder ({final_length} chars)")
            return final_source
        else:
            print(f"    ⚠ Limited content extracted ({final_length} chars, placeholder: {has_placeholder_final})")
            # Return it anyway - let the caller decide
            return final_source
        
    except TimeoutException:
        print(f"    [!] Timeout waiting for {url}")
        try:
            return driver.page_source if driver else None
        except:
            return None
    except WebDriverException as e:
        print(f"    [!] WebDriver error for {url}: {e}")
        return None
    except Exception as e:
        print(f"    [!] Unexpected error for {url}: {e}")
        import traceback
        traceback.print_exc()
        try:
            return driver.page_source if driver else None
        except:
            return None

def crawl_website(base_url, max_pages=100):
    global driver, visited, playwright_browser, playwright_context
    
    # Reset visited set for each crawl
    visited = set()
    
    use_playwright = False
    page = None
    playwright_instance = None
    
    # Try Playwright first (more reliable)
    if PLAYWRIGHT_AVAILABLE:
        try:
            print("[+] Attempting to use Playwright...")
            playwright_instance = sync_playwright().start()
            playwright_browser = playwright_instance.chromium.launch(headless=True)
            playwright_context = playwright_browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = playwright_context.new_page()
            print("[+] Playwright initialized successfully")
            use_playwright = True
        except Exception as e:
            print(f"[!] Playwright initialization failed: {e}")
            print("[+] Falling back to Selenium...")
            use_playwright = False
    
    # If Playwright failed, try Selenium
    if not use_playwright:
        # Setup Selenium WebDriver
        chrome_options = Options()
        chrome_options.add_argument('--headless=new')  # Use new headless mode
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        try:
            if WEBDRIVER_MANAGER_AVAILABLE:
                print("[+] Using webdriver-manager for ChromeDriver...")
                try:
                    service = Service(ChromeDriverManager().install())
                    driver = webdriver.Chrome(service=service, options=chrome_options)
                except Exception as e1:
                    print(f"[!] webdriver-manager failed: {e1}")
                    print("[+] Trying system ChromeDriver...")
                    try:
                        driver = webdriver.Chrome(options=chrome_options)
                    except Exception as e2:
                        print(f"[!] System ChromeDriver also failed: {e2}")
                        raise e2
            else:
                print("[+] Using system ChromeDriver...")
                driver = webdriver.Chrome(options=chrome_options)
            print("[+] Chrome WebDriver initialized successfully")
        except Exception as e:
            print(f"[!] Failed to initialize Chrome WebDriver: {e}")
            print("[!] Make sure Chrome browser is installed")
            print("[!] You may need to install Chrome from: https://www.google.com/chrome/")
            if not WEBDRIVER_MANAGER_AVAILABLE:
                print("[!] Consider installing webdriver-manager: pip install webdriver-manager")
            if not PLAYWRIGHT_AVAILABLE:
                print("[!] Or try Playwright: pip install playwright && playwright install chromium")
            # Return empty dicts with proper structure
            return {}, {
                'contact_info': [],
                'job_opportunities': [],
                'metadata': [],
                'services': [],
                'testimonials': [],
                'addresses': [],
                'social_links': []
            }
    
    to_visit = [base_url]
    data_collected = {} # Stores general page content
    specific_data_collected = { # Stores structured data like contacts, jobs
        'contact_info': [],
        'job_opportunities': [],
        'metadata': [],
        'services': [],
        'testimonials': [],
        'addresses': [],
        'social_links': []
    }

    # Discover links from sitemap
    print("[+] Discovering links from sitemap...")
    sitemap_urls = discover_links_from_sitemap(base_url)
    for url in sitemap_urls[:50]:  # Limit to first 50 from sitemap
        if is_internal(base_url, url):
            to_visit.append(url)
    print(f"[+] Added {len(sitemap_urls[:50])} URLs from sitemap")
    
    # Discover links from robots.txt
    print("[+] Discovering links from robots.txt...")
    robots_urls = discover_links_from_robots(base_url)
    for url in robots_urls:
        if is_internal(base_url, url):
            to_visit.append(url)
    
    # Add common paths
    print("[+] Adding common website paths...")
    common_paths = get_common_paths(base_url)
    for path_url in common_paths:
        if path_url not in to_visit:
            to_visit.append(path_url)
    print(f"[+] Added {len(common_paths)} common paths")
    
    # Add specific important paths
    if 'contact' not in base_url.lower() and 'about' not in base_url.lower():
        to_visit.append(urljoin(base_url, '/contact-us/')) 
        to_visit.append(urljoin(base_url, '/contact/'))
        to_visit.append(urljoin(base_url, '/get-in-touch/'))
    if 'career' not in base_url.lower() and 'job' not in base_url.lower():
        to_visit.append(urljoin(base_url, '/careers/')) 
        to_visit.append(urljoin(base_url, '/jobs/'))
        to_visit.append(urljoin(base_url, '/opportunities/'))
        to_visit.append(urljoin(base_url, '/join-us/'))
    if 'about' not in base_url.lower():
        to_visit.append(urljoin(base_url, '/about/'))
        to_visit.append(urljoin(base_url, '/about-us/'))
        to_visit.append(urljoin(base_url, '/company/'))
    if 'life' not in base_url.lower() and 'culture' not in base_url.lower():
        to_visit.append(urljoin(base_url, '/life/'))
        to_visit.append(urljoin(base_url, '/culture/'))
        to_visit.append(urljoin(base_url, '/company-life/'))
        to_visit.append(urljoin(base_url, '/life-at-company/'))
        to_visit.append(urljoin(base_url, '/work-life/'))
        to_visit.append(urljoin(base_url, '/employee-life/'))


    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        url = normalize_url(base_url, url)

        if url in visited:
            continue
        visited.add(url)

        print(f"[+] Crawling: {url}")
        
        # Get page content using Playwright or Selenium
        if use_playwright:
            page_source = get_page_content_with_playwright(url, page)
        else:
            page_source = get_page_content_with_selenium(url, driver)
        
        if page_source is None:
            print(f"[!] Failed to get content from {url}")
            continue

        soup = BeautifulSoup(page_source, "html.parser")

        # Create a copy for detailed extraction (keep original for link extraction)
        soup_for_extraction = BeautifulSoup(page_source, "html.parser")
        
        # Remove script, style, header, footer, nav tags for text extraction
        for script_or_style in soup_for_extraction(["script", "style", "header", "footer", "nav"]):
            script_or_style.extract()
        
        text = soup_for_extraction.get_text(separator=' ', strip=True)
        
        # Only save if we have meaningful content
        text_length = len(text.strip()) if text else 0
        has_placeholder = "You need to enable JavaScript to run this app" in text if text else False
        
        # Extract detailed content with structured information
        detailed_content = format_detailed_content(soup_for_extraction, url, base_url)
        detailed_length = len(detailed_content.strip()) if detailed_content else 0
        
        # More lenient check - save content if it's substantial
        # Always prefer detailed_content as it has more structured information
        if text and text_length > 50:  # Lowered threshold from 100 to capture more pages
            if not has_placeholder:
                # Use detailed content instead of plain text
                data_collected[url] = detailed_content if detailed_length > text_length else text
                saved_length = detailed_length if detailed_length > text_length else text_length
                print(f"    ✓ Saved detailed content: {saved_length} characters")
            else:
                # If we have a lot of content but placeholder is there, might be in HTML comments
                # Check if the actual visible text is substantial
                # Count non-placeholder words
                words = text.split()
                non_placeholder_words = [w for w in words if "enable" not in w.lower() and "javascript" not in w.lower()]
                if len(non_placeholder_words) > 15:  # Lowered from 20 to be more lenient
                    data_collected[url] = detailed_content if detailed_length > text_length else text
                    saved_length = detailed_length if detailed_length > text_length else text_length
                    print(f"    ✓ Saved detailed content despite placeholder: {saved_length} characters ({len(non_placeholder_words)} words)")
                else:
                    print(f"    [-] Skipping {url} - mostly placeholder text ({text_length} chars)")
        elif text and text_length > 30:  # Even lower threshold for short but important pages
            if not has_placeholder:
                data_collected[url] = detailed_content if detailed_length > text_length else text
                saved_length = detailed_length if detailed_length > text_length else text_length
                print(f"    ✓ Saved detailed content: {saved_length} characters")
            else:
                # For short pages, be more lenient
                words = text.split()
                non_placeholder_words = [w for w in words if "enable" not in w.lower() and "javascript" not in w.lower()]
                if len(non_placeholder_words) > 10:
                    data_collected[url] = detailed_content if detailed_length > text_length else text
                    saved_length = detailed_length if detailed_length > text_length else text_length
                    print(f"    ✓ Saved short content: {saved_length} characters")
                else:
                    print(f"    [-] Skipping {url} - placeholder detected in short content ({text_length} chars)")
        else:
            print(f"    [-] Skipping {url} - insufficient content ({text_length} chars)")
            if text_length <= 20:
                continue

        # --- Specific Data Extraction ---
        contact_info = extract_contact_info(soup_for_extraction)
        if contact_info:
            specific_data_collected['contact_info'].append({'url': url, 'data': contact_info})

        job_opportunities = extract_job_opportunities(soup_for_extraction)
        if job_opportunities:
            specific_data_collected['job_opportunities'].append({'url': url, 'data': job_opportunities})
        
        # Extract page metadata
        metadata = extract_page_metadata(soup_for_extraction, url)
        if metadata.get('title') or metadata.get('description'):
            specific_data_collected['metadata'].append({'url': url, 'data': metadata})
        
        # Extract services/products
        services = extract_services_products(soup_for_extraction)
        if services:
            specific_data_collected['services'].append({'url': url, 'data': services})
        
        # Extract testimonials
        testimonials = extract_testimonials(soup_for_extraction)
        if testimonials:
            specific_data_collected['testimonials'].append({'url': url, 'data': testimonials})
        
        # Extract addresses
        addresses = extract_addresses(soup_for_extraction)
        if addresses:
            specific_data_collected['addresses'].append({'url': url, 'data': addresses})
        
        # Extract social media links
        social_links = extract_social_media_links(soup_for_extraction, base_url)
        if social_links:
            specific_data_collected['social_links'].append({'url': url, 'data': social_links})


        # Extract all <a href=""> links - prioritize important links
        important_keywords = ['life', 'culture', 'about', 'company', 'team', 'values', 
                              'mission', 'vision', 'career', 'service', 'product', 'contact']
        
        priority_links = []
        regular_links = []
        
        for link_tag in soup.find_all("a", href=True):
            raw_link = link_tag['href']
            full_url = normalize_url(url, raw_link)
            if is_internal(base_url, full_url) and full_url not in visited:
                if full_url != url:
                    link_text = link_tag.get_text(strip=True).lower()
                    href_lower = raw_link.lower()
                    # Check if link is important
                    if any(keyword in link_text or keyword in href_lower for keyword in important_keywords):
                        priority_links.append(full_url)
                    else:
                        regular_links.append(full_url)
        
        # Add priority links first, then regular links
        to_visit.extend(priority_links)
        to_visit.extend(regular_links)

        time.sleep(1)  # Be polite

    # Close the browser
    if use_playwright:
        try:
            if playwright_context:
                playwright_context.close()
            if playwright_browser:
                playwright_browser.close()
            if playwright_instance:
                playwright_instance.stop()
            print("[+] Playwright closed successfully")
        except Exception as e:
            print(f"[!] Error closing Playwright: {e}")
    else:
        if driver:
            try:
                driver.quit()
                print("[+] Chrome WebDriver closed successfully")
            except Exception as e:
                print(f"[!] Error closing Selenium driver: {e}")
    
    print(f"\n[+] Crawl complete. Collected {len(data_collected)} pages with content.")
    if len(data_collected) == 0:
        print("[!] WARNING: No content was collected. Check the error messages above.")
        print("[!] Possible issues:")
        print("    - Website requires more time to load")
        print("    - Browser/ChromeDriver not working properly")
        print("    - Network connectivity issues")
        print("    - Website blocking automated access")
    
    return data_collected, specific_data_collected #


if __name__ == "__main__":
    import sys
    website = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    print("=" * 60)
    print(f"Starting crawl for: {website}")
    print("=" * 60)
    
    try:
        general_scraped_data, structured_scraped_data = crawl_website(website, max_pages=50)

        print("\n" + "=" * 60)
        print("Scraping Summary:")
        print(f"  Pages scraped: {len(general_scraped_data)}")
        contact_count = len(structured_scraped_data.get('contact_info', []))
        job_count = len(structured_scraped_data.get('job_opportunities', []))
        metadata_count = len(structured_scraped_data.get('metadata', []))
        services_count = len(structured_scraped_data.get('services', []))
        testimonials_count = len(structured_scraped_data.get('testimonials', []))
        addresses_count = len(structured_scraped_data.get('addresses', []))
        social_count = len(structured_scraped_data.get('social_links', []))
        print(f"  Contact info found: {contact_count}")
        print(f"  Job opportunities found: {job_count}")
        print(f"  Page metadata found: {metadata_count}")
        print(f"  Services/products found: {services_count}")
        print(f"  Testimonials found: {testimonials_count}")
        print(f"  Addresses found: {addresses_count}")
        print(f"  Social media links found: {social_count}")
        print("=" * 60 + "\n")

        # Save general scraped data for the chatbot
        output_general_file = "scraped_data.txt"
        with open(output_general_file, "w", encoding="utf-8") as f:
            if general_scraped_data:
                for url, content in general_scraped_data.items():
                    f.write(f"--- {url} ---\n")
                    f.write(content + "\n\n")
                print(f"[OK] General scraped data saved to {output_general_file} ({len(general_scraped_data)} pages)")
            else:
                f.write("No content was scraped. The website may require additional JavaScript rendering time.\n")
                print(f"[WARN] No content was scraped. Check the error messages above.")

        # Save specific structured data to a separate file (or database)
        output_structured_file = "structured_data.txt"
        with open(output_structured_file, "w", encoding="utf-8") as f:
            # Contact Information
            f.write("--- Contact Information ---\n")
            contact_info = structured_scraped_data.get('contact_info', [])
            if contact_info:
                for entry in contact_info:
                    f.write(f"URL: {entry['url']}\n")
                    for key, value in entry['data'].items():
                        if isinstance(value, list):
                            f.write(f"  {key.replace('_', ' ').title()}: {', '.join(value)}\n")
                        else:
                            f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
                    f.write("\n")
            else:
                f.write("No contact information found.\n")

            # Job Opportunities
            f.write("\n--- Job Opportunities ---\n")
            job_opportunities = structured_scraped_data.get('job_opportunities', [])
            if job_opportunities:
                for entry in job_opportunities:
                    f.write(f"URL: {entry['url']}\n")
                    f.write("  Jobs:\n")
                    for job in entry['data']:
                        f.write(f"    - {job}\n")
                    f.write("\n")
            else:
                f.write("No job opportunities found.\n")
            
            # Page Metadata
            f.write("\n--- Page Metadata ---\n")
            metadata = structured_scraped_data.get('metadata', [])
            if metadata:
                for entry in metadata:
                    f.write(f"URL: {entry['url']}\n")
                    data = entry['data']
                    if data.get('title'):
                        f.write(f"  Title: {data['title']}\n")
                    if data.get('description'):
                        f.write(f"  Description: {data['description']}\n")
                    if data.get('keywords'):
                        f.write(f"  Keywords: {data['keywords']}\n")
                    f.write("\n")
            else:
                f.write("No metadata found.\n")
            
            # Services/Products
            f.write("\n--- Services/Products ---\n")
            services = structured_scraped_data.get('services', [])
            if services:
                for entry in services:
                    f.write(f"URL: {entry['url']}\n")
                    for service in entry['data']:
                        f.write(f"  Service: {service.get('name', 'N/A')}\n")
                        if service.get('description'):
                            f.write(f"    Description: {service['description']}\n")
                    f.write("\n")
            else:
                f.write("No services/products found.\n")
            
            # Testimonials
            f.write("\n--- Testimonials/Reviews ---\n")
            testimonials = structured_scraped_data.get('testimonials', [])
            if testimonials:
                for entry in testimonials:
                    f.write(f"URL: {entry['url']}\n")
                    for testimonial in entry['data']:
                        if testimonial.get('text'):
                            f.write(f"  Testimonial: {testimonial['text'][:200]}...\n")
                        if testimonial.get('author'):
                            f.write(f"    Author: {testimonial['author']}\n")
                        if testimonial.get('role'):
                            f.write(f"    Role: {testimonial['role']}\n")
                    f.write("\n")
            else:
                f.write("No testimonials found.\n")
            
            # Addresses
            f.write("\n--- Addresses ---\n")
            addresses = structured_scraped_data.get('addresses', [])
            if addresses:
                for entry in addresses:
                    f.write(f"URL: {entry['url']}\n")
                    for addr in entry['data']:
                        f.write(f"  Address: {addr}\n")
                    f.write("\n")
            else:
                f.write("No addresses found.\n")
            
            # Social Media Links
            f.write("\n--- Social Media Links ---\n")
            social_links = structured_scraped_data.get('social_links', [])
            if social_links:
                for entry in social_links:
                    f.write(f"URL: {entry['url']}\n")
                    for platform, urls in entry['data'].items():
                        f.write(f"  {platform.title()}: {', '.join(urls)}\n")
                    f.write("\n")
            else:
                f.write("No social media links found.\n")
        print(f"[OK] Structured data saved to {output_structured_file}")
        
    except KeyboardInterrupt:
        print("\n[!] Scraping interrupted by user")
        if driver:
            driver.quit()
    except Exception as e:
        print(f"\n[!] Error during crawling: {e}")
        import traceback
        traceback.print_exc()
        if driver:
            driver.quit()