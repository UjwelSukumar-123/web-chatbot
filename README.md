# Integrated Web Chatbot System

A powerful, integrated web chatbot system that combines web scraping with AI-powered responses using OpenAI and semantic search capabilities.

## 🌟 New Feature: Company-Specific Data Management  

**Enhance your chatbot with custom company information!** This feature allows customers to add their own firm-specific data to make the chatbot more robust and tailored to their business needs.

### What You Can Add:
- **Company Information**: Mission statements, company history, values
- **Products & Services**: Detailed descriptions, specifications, pricing
- **Policies & Procedures**: Company policies, terms of service, guidelines
- **FAQ**: Frequently asked questions and answers
- **Contact & Support**: Support information, contact details, hours
- **Technical Details**: Technical specifications, requirements, documentation
- **Custom Categories**: Any other business-specific information

### Benefits:
- **Enhanced Accuracy**: Chatbot provides more accurate, company-specific responses
- **Better Context**: Combines scraped website data with custom company knowledge
- **Professional Responses**: Tailored answers that reflect your business
- **Easy Management**: Simple interface to add, edit, and remove custom data
- **Persistent Storage**: Custom data is saved and persists between sessions

## 🚀 Features

### Core Functionality
- **Web Scraping**: Automatically crawls and extracts content from target websites
- **Dual Chatbot Modes**: 
  - 🤖 **OpenAI**: Advanced AI-powered responses using OpenAI
  - 🔍 **Basic Semantic**: Fast semantic search-based responses
- **Real-time Processing**: Live scraping status and data management
- **Persistent Storage**: Saves scraped and custom data for future use

### AI Integration
- **OpenAI**: State-of-the-art language model for natural conversations
- **Semantic Search**: Intelligent content retrieval using sentence transformers
- **Context-Aware Responses**: Combines multiple data sources for comprehensive answers
- **Source Attribution**: Provides links to source content for transparency

### Data Management
- **Custom Data Addition**: Add company-specific information through web interface
- **Data Categorization**: Organize custom data with predefined categories
- **Data Persistence**: Automatic saving and loading of custom data
- **Data Reset Options**: Reset scraped data, custom data, or both

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- OpenAI API key (for OpenAI features)

### Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd web-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API key**
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     OPENAI_MODEL_NAME=gpt-3.5-turbo
     ```
     Note: You can use `gpt-4` or other models by changing `OPENAI_MODEL_NAME`

4. **Run the application**
   ```bash
   python app.py
   ```

## 📖 Usage

### Getting Started
1. **Launch the application** - The system will automatically start scraping the default website
2. **Add custom company data** - Use the "Company-Specific Data Management" section
3. **Start chatting** - Ask questions about the website content and your custom data

### Adding Custom Data
1. Click **"📝 Manage Data"** in the Company-Specific Data Management section
2. Fill in the form:
   - **Title**: Descriptive name for the information
   - **Category**: Choose from predefined categories
   - **Content**: Detailed information about the topic
3. Click **"Add Data"** to save
4. The chatbot will now use this information in responses

### Managing Data
- **View**: Click the 👁️ button to see full content
- **Remove**: Click the 🗑️ button to delete entries
- **Reset**: Use reset buttons to clear data as needed

### Chatbot Modes
- **OpenAI**: Best for complex questions and natural conversations
- **Basic Semantic**: Faster responses for simple queries

## 🔧 Configuration

### Target Website
- Change the target website using the URL input field
- Click "Update" to apply changes
- Data will be automatically reset when changing websites

### Data Management
- **Reset Scraped**: Clears only website-scraped data
- **Reset All**: Clears both scraped and custom data
- Custom data is automatically saved to `custom_data.txt`

## 📊 System Status

The application provides real-time status information:
- **Scraping Status**: Current scraping progress and completion
- **Data Counts**: Number of scraped pages and custom entries
- **API Status**: OpenAI API key configuration status
- **Embedding Status**: AI model loading status

## 🚨 Troubleshooting

### Common Issues
1. **API Key Errors**: Ensure your OpenAI API key is valid and has sufficient quota
2. **Scraping Failures**: Check website accessibility and internet connection
3. **Memory Issues**: Large websites may require more memory for processing
4. **Embedding Errors**: Ensure sentence-transformers is properly installed

### Performance Tips
- Use the Basic Semantic mode for faster responses
- Limit custom data entries to essential information
- Monitor API usage to avoid quota limits

## 🔒 Security Notes

- API keys are stored in environment variables
- Custom data is stored locally in plain text
- No external data transmission beyond OpenAI API calls
- Consider data sensitivity when adding custom information

## 📈 Future Enhancements

- **Data Import/Export**: CSV/JSON import/export functionality
- **Advanced Categorization**: Custom category creation
- **Data Analytics**: Usage statistics and performance metrics
- **Multi-language Support**: Internationalization features
- **API Endpoints**: RESTful API for external integrations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Transform your website chatbot into a comprehensive business assistant with company-specific knowledge!** 🚀