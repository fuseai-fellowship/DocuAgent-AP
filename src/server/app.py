from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
from dotenv import load_dotenv
import os, shutil
from openai import OpenAI
from google import genai
from pathlib import Path
from src.pipeline.pdf_ingest import process_pdf_pipeline
from src.pipeline.chat_with_LLM import SYSTEM_INSTRUCTION


load_dotenv()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print("[WARNING] Failed to initialize OpenAI client:", e)
        openai_client = None

genai_client = None
if GEMINI_API_KEY:
    # initialize genai client
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
    
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_DIR = os.path.join(os.path.dirname(__file__), "..", "ui")
if os.path.isdir(UI_DIR):
    app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # print(f'Got the file {file}')

    if not file.filename:
        return {"filename": "Theres an error in /upload endpoint", "status": "error"}
    
    fname = os.path.basename(file.filename)
    saved_path = os.path.join(UPLOAD_DIR, fname)

    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # print(fname, saved_path)

    return {"filename": file.filename, "status" : "uploaded"}


@app.post("/analyze")
def analyze(filename: str = Form(...)):
    if not filename:
        return {"error": "No filename provided"}

    uploaded_file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(uploaded_file_path):
        return {"error": f"File not found: {uploaded_file_path}"}

    process_pdf_pipeline(uploaded_file_path, layout_result_save=True)

    fname_without_ext = os.path.splitext(filename)[0]
    json_file_path = os.path.join(OUTPUT_DIR, fname_without_ext, f"{fname_without_ext}_all_results.json")

    if not os.path.exists(json_file_path):
        return {"error": "JSON output not found"}
    
    with open(json_file_path, "r", encoding="utf-8") as f:
        doc_data = json.load(f)

    # Flatten content into text for prompt
    content_text = ""
    for page_key in sorted(doc_data.keys()):
        page = doc_data[page_key]
        for box in page.get("boxes", []):
            label = box.get("label", "")
            content = box.get("block_content", "")
            if label == "image" and isinstance(content, dict) and "base_64_encoded" in content:
                content_text += "![Figure placeholder]\n"
            elif isinstance(content, str) and content.strip():
                content_text += content.strip() + "\n\n"

    initial_prompt = SYSTEM_INSTRUCTION + "\n\nDocument content:\n" + content_text

    # Try OpenAI first
    markdown_summary = None
    openai_error = None
    gemini_error = None

    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model='gpt-5-mini',
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": initial_prompt}
                ]
            )
            markdown_summary = response.choices[0].message.content
        except Exception as e:
            openai_error = str(e)
            print("[ERROR] OpenAI call failed:", e)

    # Fallback to Gemini if OpenAI failed
    if markdown_summary is None and genai_client:
        try:
            gem_resp = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=initial_prompt
            )
            # genai response text typically in .text
        # Try to extract clean text from Gemini
            if hasattr(gem_resp, "text") and gem_resp.text:
                markdown_summary = gem_resp.text
            else:
                try:
                    markdown_summary = gem_resp.candidates[0].content.parts[0].text
                except:
                    markdown_summary = None

        except Exception as e:
            gemini_error = str(e)
            print("[ERROR] Gemini call failed:", e)

    # If still no summary, return combined error
    if markdown_summary is None:
        return JSONResponse(
            {
                "error": "Both OpenAI and Gemini calls failed or were unavailable.",
                "openai_error": openai_error,
                "gemini_error": gemini_error
            },
            status_code=500
        )

    # Save markdown summary next to JSON file (same folder, .md)
    md_path = Path(json_file_path).with_suffix(".md")
    os.makedirs(md_path.parent, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_summary)

    return {"summary": markdown_summary, "summary_path": str(md_path)}

@app.post('/chat')
async def chat(filename: str = Form(...), message: str = Form(...)):
    fname_without_ext = os.path.splitext(filename)[0]
    summary_path = os.path.join(OUTPUT_DIR, fname_without_ext, fname_without_ext + "_all_results.md")
    print(summary_path)

    if not os.path.exists(summary_path):
        return JSONResponse({"error": f"Summary file not found: {summary_path}"}, status_code=404)
    
    with open(summary_path, "r", encoding="utf-8") as f:
        context = f.read()

    prompt = f"You are a helpful assistant. Use the following document summary to answer questions:\n\n{context}\n\nUser question: {message}\n\nAnswer concisely."

    # Try OpenAI first
    answer = None
    openai_error = None
    gemini_error = None

    if OPENAI_API_KEY:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are a document analysis assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content
        except Exception as e:
            openai_error = str(e)
            print("[ERROR] OpenAI call failed:", e)

    # Fallback to Gemini
    if answer is None and genai_client:
        try:
            gem_resp = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            if hasattr(gem_resp, "text") and gem_resp.text:
                answer = gem_resp.text
            else:
                try:
                    answer = gem_resp.candidates[0].content.parts[0].text
                except:
                    answer = None
        except Exception as e:
            gemini_error = str(e)
            print("[ERROR] Gemini call failed:", e)

    if answer is None:
        return JSONResponse(
            {
                "error": "Both OpenAI and Gemini calls failed or were unavailable.",
                "openai_error": openai_error,
                "gemini_error": gemini_error
            },
            status_code=500
        )

    return {"answer": answer}
