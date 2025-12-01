# Web Chatbot API Usage Guide

This chatbot system works with **any website**. You can configure it via API endpoints.

## Quick Start

### 1. Set Target Website

**Endpoint:** `POST /set_website`

**Request Body:**
```json
{
  "website": "https://example.com",
  "auto_scrape": true
}
```

**Parameters:**
- `website` (required): The target website URL (with or without http/https)
- `auto_scrape` (optional, default: false): If `true`, automatically starts scraping after setting the website

**Example with Postman:**
```
POST http://localhost:5000/set_website
Content-Type: application/json

{
  "website": "https://example.com",
  "auto_scrape": true
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Website updated to https://example.com. Scraping started automatically.",
  "website": "https://example.com",
  "scraping_status": "started"
}
```

### 2. Start Scraping (if not auto-started)

**Endpoint:** `POST /start_scraping`

**Example with Postman:**
```
POST http://localhost:5000/start_scraping
```

**Response:**
```json
{
  "status": "started",
  "message": "Scraping started successfully"
}
```

### 3. Check Scraping Status

**Endpoint:** `GET /scraping_status`

**Example with Postman:**
```
GET http://localhost:5000/scraping_status
```

**Response:**
```json
{
  "scraping_in_progress": false,
  "scraping_complete": true,
  "data_loaded": true,
  "total_pages": 45,
  "target_website": "https://example.com"
}
```

### 4. Ask Questions

**Endpoint:** `POST /ask`

**Request Body (form-data):**
- `question`: Your question
- `chatbot_type`: "openai" (default) or "basic"

**Example with Postman:**
```
POST http://localhost:5000/ask
Content-Type: application/x-www-form-urlencoded

question=What services does the company offer?
chatbot_type=openai
```

**Response:**
```json
{
  "answer": "Based on the website content, the company offers...",
  "source_urls": ["https://example.com/services", "https://example.com/about"]
}
```

## Complete Workflow Example

### Step 1: Set Website and Start Scraping
```bash
POST /set_website
{
  "website": "https://your-website.com",
  "auto_scrape": true
}
```

### Step 2: Wait for Scraping to Complete
```bash
GET /scraping_status
# Check until "scraping_complete": true
```

### Step 3: Ask Questions
```bash
POST /ask
question=Tell me about the company
```

## Important Notes

1. **Data Clearing**: When you set a new website, all previous scraped data is automatically cleared to avoid confusion.

2. **Scraping Time**: Scraping can take several minutes depending on the website size. The system scrapes up to 200 pages by default (configurable via `MAX_SCRAPE_PAGES` environment variable).

3. **Multiple Websites**: You can switch between different websites by calling `/set_website` with a new URL. The old data will be cleared automatically.

4. **No Default Website**: The system doesn't have a default website. You must set one via the API.

## Other Useful Endpoints

- `GET /get_website` - Get current target website
- `POST /reload_data` - Reload scraped data from saved files
- `POST /reset_data` - Clear all scraped data
- `GET /health` - Check system health and configuration

## Environment Variables (Optional)

- `TARGET_WEBSITE`: Default website (if not set via API)
- `MAX_SCRAPE_PAGES`: Maximum pages to scrape (default: 200)
- `OPENAI_API_KEY`: Required for OpenAI chatbot responses
- `OPENAI_MODEL_NAME`: OpenAI model to use (default: gpt-3.5-turbo)

