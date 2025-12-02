# Project Structure

This document describes the refactored project structure.

## Directory Structure

```
web-chatbot/
├── main.py                 # Main entry point with threading
├── api.py                  # All FastAPI routes and endpoints
├── chatbot.py              # Chatbot logic (semantic retrieval, OpenAI responses)
├── custom_data.py          # Custom data management
├── data_manager.py         # Embeddings and metadata loading/saving
├── scraper.py              # Web scraping wrapper
├── deep_scraper.py         # Core scraping functionality (unchanged)
├── basic_chatbot.py        # Standalone basic chatbot (unchanged)
├── demo_custom_data.py     # Demo script (unchanged)
├── data/                   # Embedding files (.npy, .pkl)
│   ├── embeddings.npy
│   ├── embeddings_metadata.pkl
│   ├── structured_embeddings.npy
│   └── structured_metadata.pkl
├── custom_data/            # Custom data files
│   └── custom_data.txt
└── templates/              # HTML templates
    └── index.html
```

## Module Descriptions

### main.py
- Main entry point for the application
- Handles threading for background scraping
- Initializes all modules and loads data
- Creates and runs the FastAPI app
- Manages global state

### api.py
- Contains all FastAPI routes and endpoints
- Handles HTTP requests and responses
- Integrates with chatbot, custom_data, and scraper modules
- Uses dependency injection for state management

### chatbot.py
- Core chatbot functionality
- Semantic retrieval from multiple data sources
- OpenAI response generation
- Text cleaning and validation utilities
- Basic chatbot response (without OpenAI)

### custom_data.py
- Manages custom data entries
- Handles loading, saving, adding, and removing custom data
- Creates embeddings for custom data
- Stores data in `custom_data/` directory

### data_manager.py
- Manages embeddings and metadata files
- Handles loading and saving of scraped data embeddings
- Handles loading and saving of structured data embeddings
- All embedding files stored in `data/` directory

### scraper.py
- Wrapper around deep_scraper.py
- Processes scraped data and creates embeddings
- Handles structured data extraction
- Integrates with data_manager for saving

### deep_scraper.py
- Core web scraping functionality (unchanged)
- Uses Selenium/Playwright for JavaScript rendering
- Extracts structured data (contacts, jobs, services, etc.)

## Running the Application

### Start the server:
```bash
python main.py
```

The server will:
1. Load environment variables from `.env`
2. Initialize the sentence transformer model
3. Load existing embeddings from `data/` folder
4. Load custom data from `custom_data/` folder
5. Start the FastAPI server on port 5000
6. Optionally auto-start scraping if website is set and no data exists

## Data Storage

### Embeddings (data/ folder)
- `embeddings.npy` - Scraped page embeddings
- `embeddings_metadata.pkl` - Scraped page metadata (URLs and texts)
- `structured_embeddings.npy` - Structured data embeddings
- `structured_metadata.pkl` - Structured data metadata

### Custom Data (custom_data/ folder)
- `custom_data.txt` - Custom data entries in text format

## Environment Variables

Required in `.env` file:
- `OPENAI_API_KEY` - OpenAI API key
- `OPENAI_MODEL_NAME` - Model name (default: gpt-3.5-turbo)
- `TARGET_WEBSITE` - Target website URL (optional)
- `MAX_SCRAPE_PAGES` - Maximum pages to scrape (default: 200)

## API Endpoints

All endpoints are defined in `api.py`:
- `GET /` - Web interface
- `POST /ask` - Ask a question
- `GET /scraping_status` - Get scraping status
- `POST /start_scraping` - Start scraping
- `POST /set_website` - Set target website
- `GET /get_website` - Get current website
- `POST /reload_data` - Reload data from files
- `POST /reset_data` - Reset scraped data
- `POST /reset_all_data` - Reset all data
- `POST /add_custom_data` - Add custom data
- `GET /get_custom_data` - Get all custom data
- `POST /remove_custom_data` - Remove custom data
- `GET /custom_data_stats` - Get statistics
- `GET /health` - Health check
- `GET /debug` - Debug information
- `POST /test_query` - Test semantic retrieval
- `GET /test_api` - Test OpenAI API
- `GET /project_info` - Project information

## Threading

The application uses threading for:
- Background web scraping (non-blocking)
- Allows API to remain responsive during scraping
- Scraping status can be checked via `/scraping_status` endpoint

## Migration Notes

If you have existing embedding files in the root directory, they have been moved to the `data/` folder. The application will automatically use the new location.

