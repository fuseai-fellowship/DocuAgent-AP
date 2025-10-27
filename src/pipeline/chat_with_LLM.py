import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
headers = {"Content-Type": "application/json"}

# ------------------------------
# 🧠 "System-like" instruction
# ------------------------------
SYSTEM_INSTRUCTION = """
You are a document analysis and summarization assistant.
You will receive JSON data extracted from a document layout.
Each block includes:
- label (like "text", "paragraph_title", "figure_title", "table")
- coordinates (its position on the page)
- block_content (actual extracted text or HTML).

Your job:
1. Reconstruct the logical order of the document.
2. Convert it into a **clean, structured, simplified Markdown version**:
   - Use headings (`#`, `##`, etc.) for titles.
   - Render tables properly in Markdown.
   - Fix OCR text issues and spacing.
3. Keep all key information, remove junk or duplicates.
4. After generating the Markdown, stay in chat mode and answer questions based only on the document.

Return only valid Markdown for your first message.
"""

def chat_with_gemini(json_file_path, save_markdown=True):
    with open(json_file_path, "r", encoding="utf-8") as f:
        doc_data = json.load(f)

    print("[INFO] Document loaded. Type 'exit' anytime to stop chatting.\n")

    initial_prompt = f"""
{SYSTEM_INSTRUCTION}

Here is the document layout JSON:

{json.dumps(doc_data, indent=2)}

Please provide the structured Markdown summary first.
"""

    chat_history = [
        {"role": "user", "parts": [{"text": initial_prompt}]}
    ]

    first_response = True

    while True:
        response = requests.post(url, headers=headers, data=json.dumps({"contents": chat_history}))
        
        if response.status_code != 200:
            print("Error:", response.status_code, response.text)
            break

        result = response.json()
        model_reply = result["candidates"][0]["content"]["parts"][0]["text"]
        
        print("\n🧠 Gemini:\n", model_reply)

        # Save Markdown summary on first reply
        if save_markdown and first_response:
            md_path = os.path.splitext(json_file_path)[0] + "_summary.md"
            with open(md_path, "w", encoding="utf-8") as md_file:
                md_file.write(model_reply)
            print(f"\n📄 Structured Markdown summary saved to: {md_path}")
            first_response = False

        # Get user input for follow-up
        user_input = input("\n💬 You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 Chat ended.")
            break

        # Add user input to chat history
        chat_history.append({"role": "user", "parts": [{"text": user_input}]})
