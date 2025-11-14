import requests
import json
from dotenv import load_dotenv
import os
from openai import OpenAI
from pathlib import Path

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

GEMINI_BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
headers = {"Content-Type": "application/json"}


SYSTEM_INSTRUCTION = """
You are a document analysis and summarization assistant.
You will receive JSON data extracted from a document layout.
Each block includes:
- label (like "text", "paragraph_title", "figure_title", "table", "image")
- coordinates (its position on the page)
- block_content (actual extracted text, HTML, or base64 image data).

Your job:
1. Reconstruct the logical order of the document.
2. Convert it into a **clean, structured, simplified Markdown version**:
   - Use headings (`#`, `##`, etc.) for titles.
   - Render tables properly in Markdown.
   - For images:
       - Include the figure number if present (e.g., "Figure 1:").
       - Summarize the figure content in a caption under the image using Markdown.
       - Example:
         ![Figure 1: Diagram of workflow](data:image/png;base64,...)
   - Fix OCR text issues and spacing.
3. Keep all key information, remove junk or duplicates.
4. After generating the Markdown, stay in chat mode and answer questions based only on the document.

Return only valid Markdown for your first message.
"""


def chat_with_openai(json_file_path: str, save_markdown: bool = True):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in environment")

    client = OpenAI(api_key=OPENAI_API_KEY)

    with open(json_file_path, "r", encoding="utf-8") as f:
        doc_data = json.load(f)

    print("[INFO] Document loaded. Type 'exit' anytime to stop chatting.\n")

    content_text = ""
    for page_key in sorted(doc_data.keys()):
        page = doc_data[page_key]
        for box in page.get("boxes", []):
            label = box.get("label", "")
            content = box.get("block_content", "")
            if label == "image" and isinstance(content, dict) and "base_64_encoded" in content:
                content_text += f"\n![Image placeholder: base64 data]\n"
            elif content.strip():
                content_text += f"{content.strip()}\n\n"

    initial_prompt = SYSTEM_INSTRUCTION + "\n\nDocument content:\n" + content_text
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": initial_prompt}
        ]
    )

    markdown_summary = response.choices[0].message.content
    print("\n🧠 OpenAI Markdown Summary Generated:\n", markdown_summary)

    if save_markdown:
        md_path = Path(json_file_path).with_name(Path(json_file_path).stem + "_summary.md")
        with open(md_path, "w", encoding="utf-8") as md_file:
            md_file.write(markdown_summary)
        print(f"\n📄 Structured Markdown summary saved to: {md_path}")

    system_context = f"You are a helpful assistant. Use the following document summary to answer questions:\n{markdown_summary}"
    chat_history = [{"role": "system", "content": system_context}]

    while True:
        user_input = input("\n💬 You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 Chat ended.")
            break

        chat_history.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=chat_history
            )
            reply = response.choices[0].message.content
            print("\n🧠 OpenAI:\n", reply)

        except Exception as e:
            print("[ERROR] OpenAI request failed:", e)

def chat_with_gemini(json_file_path, save_markdown=True):
    with open(json_file_path, "r", encoding="utf-8") as f:
        doc_data = json.load(f)

    print("[INFO] Document loaded. Type 'exit' anytime to stop chatting.\n")

    parts = []
    for page_key in sorted(doc_data.keys()):
        page = doc_data[page_key]
        for box in page.get("boxes", []):
            if box["label"] == "image":
                b64_data = box["block_content"]["base_64_encoded"]
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": b64_data
                    }
                })
            else:
                parts.append({"text": box["block_content"]})

    initial_prompt = f"{SYSTEM_INSTRUCTION}\n\nHere is the document layout JSON converted to parts for processing."

    chat_history = [
        {
            "role": "user",
            "parts": [{"text": initial_prompt}] + parts
        }
    ]

    first_response = True

    while True:
        response = requests.post(GEMINI_BASE_URL, headers=headers, data=json.dumps({"contents": chat_history}))
        
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