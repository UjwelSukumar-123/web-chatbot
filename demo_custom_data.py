#!/usr/bin/env python3
"""
Demo script for the Custom Data Management feature
This script demonstrates how to add, view, and manage custom company data
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5000"
DEMO_DATA = [
    {
        "title": "Company Mission Statement",
        "category": "Company Information",
        "content": "Our mission is to provide innovative technology solutions that empower businesses to achieve their digital transformation goals. We believe in creating sustainable, scalable solutions that drive growth and efficiency."
    },
    {
        "title": "Product: AI-Powered Analytics Platform",
        "category": "Products & Services",
        "content": "Our flagship AI-powered analytics platform provides real-time insights into business performance. Features include predictive analytics, automated reporting, customizable dashboards, and integration with major business tools. Pricing starts at $299/month for small businesses and scales up based on usage and features."
    },
    {
        "title": "Return Policy",
        "category": "Policies & Procedures",
        "content": "We offer a 30-day money-back guarantee on all our software products. For hardware products, returns are accepted within 14 days in original packaging. Custom solutions have project-specific terms outlined in individual contracts. Contact our support team for return authorization."
    },
    {
        "title": "Technical Support Hours",
        "category": "Contact & Support",
        "content": "Our technical support team is available Monday through Friday, 8 AM to 8 PM EST. Emergency support is available 24/7 for enterprise customers. Support channels include phone, email, live chat, and our help desk portal. Average response time is under 2 hours during business hours."
    },
    {
        "title": "System Requirements",
        "category": "Technical Details",
        "content": "Minimum system requirements: Windows 10/11 or macOS 10.15+, 8GB RAM, 2GB free disk space, internet connection. Recommended: Windows 11 or macOS 12+, 16GB RAM, 5GB free disk space, high-speed internet. Compatible with Chrome 90+, Firefox 88+, Safari 14+, Edge 90+."
    }
]

def test_api_health():
    """Test if the API is running"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API is running and healthy")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the Flask app is running.")
        return False

def add_demo_data():
    """Add all demo data entries"""
    print("\n🚀 Adding demo company data...")
    
    for i, data in enumerate(DEMO_DATA, 1):
        print(f"\n📝 Adding entry {i}/{len(DEMO_DATA)}: {data['title']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/add_custom_data",
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['status'] == 'success':
                    print(f"   ✅ Successfully added: {data['title']}")
                else:
                    print(f"   ❌ Failed to add: {result['message']}")
            else:
                print(f"   ❌ HTTP error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        # Small delay between requests
        time.sleep(0.5)

def view_custom_data():
    """View all custom data entries"""
    print("\n📋 Viewing custom data entries...")
    
    try:
        response = requests.get(f"{BASE_URL}/get_custom_data")
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                entries = data['entries']
                print(f"   📊 Total entries: {len(entries)}")
                
                for entry in entries:
                    print(f"\n   📝 {entry['title']}")
                    print(f"      Category: {entry['category']}")
                    print(f"      Preview: {entry['content'][:100]}...")
            else:
                print(f"   ❌ Error: {data['message']}")
        else:
            print(f"   ❌ HTTP error: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")

def get_custom_data_stats():
    """Get statistics about custom data"""
    print("\n📊 Getting custom data statistics...")
    
    try:
        response = requests.get(f"{BASE_URL}/custom_data_stats")
        if response.status_code == 200:
            data = response.json()
            print(f"   📈 Custom entries: {data['total_custom_entries']}")
            print(f"   🌐 Scraped pages: {data['total_scraped_pages']}")
            print(f"   ✅ Custom data available: {data['custom_data_available']}")
            print(f"   ✅ Scraped data available: {data['scraped_data_available']}")
        else:
            print(f"   ❌ HTTP error: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")

def test_chatbot_with_custom_data():
    """Test the chatbot with a question that should use custom data"""
    print("\n🤖 Testing chatbot with custom data...")
    
    test_questions = [
        "What is your company mission?",
        "Tell me about your AI analytics platform",
        "What are your return policy terms?",
        "When is technical support available?",
        "What are the system requirements?"
    ]
    
    for question in test_questions:
        print(f"\n   ❓ Question: {question}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/ask",
                data={
                    'question': question,
                    'chatbot_type': 'gemini'
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['answer']
                sources = result['source_urls']
                
                print(f"   🤖 Answer: {answer[:200]}...")
                if sources:
                    print(f"   📚 Sources: {', '.join(sources)}")
                else:
                    print(f"   📚 Sources: None")
            else:
                print(f"   ❌ HTTP error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        time.sleep(1)  # Delay between questions

def cleanup_demo_data():
    """Remove all demo data entries"""
    print("\n🧹 Cleaning up demo data...")
    
    try:
        # First get all entries
        response = requests.get(f"{BASE_URL}/get_custom_data")
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                entries = data['entries']
                print(f"   📊 Found {len(entries)} entries to remove")
                
                for entry in entries:
                    print(f"   🗑️ Removing: {entry['title']}")
                    
                    remove_response = requests.post(
                        f"{BASE_URL}/remove_custom_data",
                        json={'index': entry['index']},
                        headers={'Content-Type': 'application/json'}
                    )
                    
                    if remove_response.status_code == 200:
                        result = remove_response.json()
                        if result['status'] == 'success':
                            print(f"      ✅ Removed successfully")
                        else:
                            print(f"      ❌ Failed to remove: {result['message']}")
                    else:
                        print(f"      ❌ HTTP error: {remove_response.status_code}")
                    
                    time.sleep(0.2)  # Small delay
            else:
                print(f"   ❌ Error getting entries: {data['message']}")
        else:
            print(f"   ❌ HTTP error: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")

def main():
    """Main demo function"""
    print("🎯 Custom Data Management Feature Demo")
    print("=" * 50)
    
    # Check if API is running
    if not test_api_health():
        print("\n❌ Please start the Flask application first:")
        print("   python app.py")
        return
    
    # Run demo
    try:
        # Add demo data
        add_demo_data()
        
        # View the data
        view_custom_data()
        
        # Get statistics
        get_custom_data_stats()
        
        # Test chatbot
        test_chatbot_with_custom_data()
        
        # Ask user if they want to clean up
        print("\n" + "=" * 50)
        response = input("🧹 Would you like to clean up the demo data? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            cleanup_demo_data()
            print("✅ Demo data cleaned up!")
        else:
            print("✅ Demo data kept for further testing")
            
        print("\n🎉 Demo completed! You can now:")
        print("   - Visit http://localhost:5000 to see the web interface")
        print("   - Add more custom data through the web interface")
        print("   - Test the chatbot with different questions")
        print("   - Use the 'Reset All' button to clear everything")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")

if __name__ == "__main__":
    main()
