#!/usr/bin/env python3
"""Simple chatbot using your API"""

import requests
import json

# Your API configuration
API_KEY = "sk-sKKaeMc04KwtyLhVbfaYmEb6C102jd8lWEmkmpDe7Co"  # Groq key
API_BASE = "http://localhost:5000"

def chat_with_ai(message):
    """Send message to your AI API"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "messages": [{"role": "user", "content": message}]
    }
    
    try:
        response = requests.post(f"{API_BASE}/api/chat/completions", 
                               headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {e}"

def main():
    print("🤖 Simple Chatbot (Using Your API)")
    print("=" * 40)
    print("Type 'quit' to exit")
    print()
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break
        
        if user_input:
            print("AI: ", end="", flush=True)
            response = chat_with_ai(user_input)
            print(response)
            print()

if __name__ == "__main__":
    main()