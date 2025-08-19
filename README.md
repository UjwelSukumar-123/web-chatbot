# 🌐 Integrated Web Chatbot System

A comprehensive web chatbot system that combines deep web scraping, semantic search, and AI-powered responses through a unified Flask web interface.

## 🚀 Features

- **🕷️ Automatic Web Scraping**: Deep scraping of websites with intelligent content extraction
- **🤖 Dual Chatbot Interface**: 
  - **Gemini AI**: Advanced AI responses using Google's Gemini model
  - **Basic Semantic**: Fast semantic similarity search through scraped content
- **🌐 Web Interface**: Modern, responsive Flask web application
- **📊 Real-time Status**: Live scraping status and data loading indicators
- **🔄 Manual Control**: Start/stop scraping and check system status anytime

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Browser  │◄──►│   Flask App     │◄──►│  Deep Scraper   │
│   (Frontend)   │    │   (Backend)     │    │  (Background)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Data Storage   │
                       │  & Embeddings   │
                       └─────────────────┘
```

## 📋 Prerequisites

- Python 3.7+
- Google API key (optional, for Gemini AI features)
- Internet connection for web scraping

## 🛠️ Installation

1. **Clone or download the project files**
2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google API key** (optional):
   - Create a `.env` file in the project root
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

## 🚀 Quick Start

### Option 1: Use the Launcher Script (Recommended)
```bash
python run_system.py
```

### Option 2: Run Directly
```bash
python app.py
```

### Option 3: Run Individual Components
```bash
# Run scraper only
python deep_scraper.py

# Run basic chatbot only
python basic_chatbot.py

# Run Flask app only
python app.py
```

## 🌐 Using the Web Interface

1. **Open your browser** and go to `http://localhost:5000`
2. **Choose your chatbot**:
   - 🤖 **Gemini AI**: Advanced AI responses (requires API key)
   - 🔍 **Basic Semantic**: Fast semantic search through scraped data
3. **Ask questions** about the scraped website content
4. **Monitor scraping status** in real-time
5. **Manually trigger scraping** when needed

## 🔧 Configuration

### Website to Scrape
Edit `app.py` line 95 to change the target website:
```python
website = "https://your-website.com"  # Change this URL
```

### Scraping Limits
Modify `max_pages` parameter in `app.py` line 97:
```python
general_scraped_data, structured_scraped_data = crawl_website(website, max_pages=100)
```

### API Settings
- **Gemini Model**: Change in `app.py` line 25
- **Embedding Model**: Modify in `app.py` line 35

## 📁 File Structure

```
web-chatbot/
├── app.py                 # Main Flask application (integrated)
├── deep_scraper.py       # Web scraping functionality
├── basic_chatbot.py      # Basic semantic chatbot
├── run_system.py         # System launcher script
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
├── templates/
│   └── index.html       # Web interface template
├── scraped_data.txt     # Generated scraped content
└── structured_data.txt   # Generated structured data
```

## 🔍 How It Works

### 1. Web Scraping Process
- **Automatic**: Starts when the Flask app launches
- **Intelligent**: Follows internal links and extracts relevant content
- **Structured**: Saves both general content and specific data (contacts, jobs)
- **Efficient**: Uses threading to avoid blocking the web interface

### 2. Data Processing
- **Content Extraction**: Removes scripts, styles, and navigation elements
- **Text Cleaning**: Normalizes and cleans extracted text
- **Embedding Generation**: Creates semantic embeddings for fast search

### 3. Chatbot Responses
- **Gemini AI**: Uses scraped content as context for AI-generated responses
- **Basic Semantic**: Finds most similar content using cosine similarity
- **Source Attribution**: Always provides source URLs for transparency

## 🎯 Use Cases

- **Company Research**: Scrape company websites for information
- **Content Analysis**: Extract and analyze website content
- **Customer Support**: Build knowledge bases from website content
- **Competitive Intelligence**: Monitor competitor websites
- **Data Mining**: Extract structured data from web pages

## 🚨 Important Notes

- **Rate Limiting**: Built-in delays to be respectful to websites
- **Content Types**: Only scrapes HTML content (skips PDFs, images, etc.)
- **Legal Compliance**: Ensure you have permission to scrape target websites
- **API Limits**: Google Gemini has daily request limits (50 for free tier)

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Issues**:
   - Check `.env` file exists and contains `GOOGLE_API_KEY=...`
   - Verify API key is valid and has sufficient quota

3. **Scraping Fails**:
   - Check internet connection
   - Verify target website is accessible
   - Check if website blocks scraping

4. **Memory Issues**:
   - Reduce `max_pages` in scraping configuration
   - Close other applications to free memory

### Debug Routes

The system provides several debug endpoints:
- `/health` - System health check
- `/debug` - Detailed system information
- `/scraping_status` - Current scraping status
- `/test_query` - Test semantic retrieval

## 🔄 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/ask` | POST | Send question to chatbot |
| `/scraping_status` | GET | Get scraping status |
| `/start_scraping` | POST | Manually start scraping |
| `/health` | GET | System health check |
| `/debug` | GET | Debug information |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Please ensure compliance with target website terms of service when scraping.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the debug endpoints
3. Check console output for error messages
4. Verify all dependencies are installed

## 🎉 What's Next?

The system is designed to be extensible. Consider adding:
- Database storage for scraped data
- Scheduled scraping
- Multiple website support
- Advanced content filtering
- Export functionality
- User authentication
- API rate limiting