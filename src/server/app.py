from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
from dotenv import load_dotenv
import os, shutil
from openai import OpenAI
from pathlib import Path
from src.pipeline.pdf_ingest import process_pdf_pipeline
from src.pipeline.chat_with_LLM import SYSTEM_INSTRUCTION

load_dotenv()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

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
    print(json_file_path)

    if not os.path.exists(json_file_path):
        return {"error": "JSON output not found"}
    
    with open(json_file_path, "r", encoding="utf-8") as f:
        doc_data = json.load(f)

    print('ABOVE FLATTEN CONTENT')
    # Flatten content into text for prompt
    content_text = ""
    for page_key in sorted(doc_data.keys()):
        page = doc_data[page_key]
        for box in page.get("boxes", []):
            label = box.get("label", "")
            content = box.get("block_content", "")
            if label == "image" and isinstance(content, dict) and "base_64_encoded" in content:
                content_text += f"![Figure placeholder]\n"
            elif content.strip():
                content_text += content.strip() + "\n\n"
    print('BELOW FLATTEN CONTENT')
    initial_prompt = SYSTEM_INSTRUCTION + "\n\nDocument content:\n" + content_text

    try:
        response = client.chat.completions.create(
            model = 'gpt-5-mini',
            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": initial_prompt}
            ]
        )
        # print("OpenAI response raw:", response)
        markdown_summary = response.choices[0].message.content
    except Exception as e:
        print("[ERROR] OpenAI call failed:", e)
        return JSONResponse({"error": str(e)}, status_code=500)

    md_path = Path(json_file_path).with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_summary)

    print(md_path)

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

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a document analysis assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content
    
    return {"answer": answer}