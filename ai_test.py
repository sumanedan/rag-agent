import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load your API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Connect to Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

# Start a chat session
chat = model.start_chat(history=[])

print("🤖 AI Chatbot is ready! Type 'quit' to exit.")
print("-" * 40)

# Loop — keeps the conversation going
while True:
    user_input = input("You: ")

    # Exit condition
    if user_input.lower() == "quit":
        print("Bye! 👋")
        break

    # Send message to Gemini and get response
    response = chat.send_message(user_input)

    print(f"AI: {response.text}")
    print("-" * 40)