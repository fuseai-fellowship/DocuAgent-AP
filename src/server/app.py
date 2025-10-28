from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from google import genai
import json
import glob

from dotenv import load_dotenv
import os, shutil, requests
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

from pathlib import Path
from src.pipeline.pdf_ingest import process_pdf_pipeline
from src.pipeline.chat_with_LLM import chat_with_gemini
from src.utils.paths import DATA, OUTPUT

load_dotenv()  # reads .env in project root

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve UI from src/ui
UI_DIR = os.path.join(os.path.dirname(__file__), "..", "ui")
if os.path.isdir(UI_DIR):
    app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def extract_text_from_pdf(path: str) -> str:
    try:
        doc = fitz.open(path)
        parts = [page.get_text() for page in doc]
        return "\n".join(parts)
    except Exception:
        return ""

def extract_text_from_image(path: str) -> str:
    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception:
        return ""

def _parse_generative_response(data: dict) -> str:
    if not data:
        return ""
    # Google "candidates" style
    if "candidates" in data and isinstance(data["candidates"], list) and data["candidates"]:
        cand = data["candidates"][0]
        # candidate may have "content" or "output" or "message"
        return cand.get("content") or cand.get("output") or cand.get("message") or str(cand)
    # "output" list style
    if "output" in data:
        out = data["output"]
        if isinstance(out, list) and out:
            first = out[0]
            return first.get("content") or first.get("text") or str(first)
        if isinstance(out, str):
            return out
    # "responses" or "response" keys
    for key in ("response", "responses", "text", "result", "content", "candidates"):
        if key in data:
            return str(data[key])
    # try nested message formats
    if isinstance(data, dict):
        # common nested field search
        def walk(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if k in ("text", "content", "message", "output", "result"):
                        if isinstance(v, (str, int, float)):
                            return str(v)
                        if isinstance(v, list) and v:
                            return str(v[0])
                    res = walk(v)
                    if res:
                        return res
            elif isinstance(d, list) and d:
                for item in d:
                    res = walk(item)
                    if res:
                        return res
            return None
        res = walk(data)
        if res:
            return res
    return str(data)

def process_uploaded_file(saved_path: str, filename: str):
    try:
        # ensure DATA dir exists and move file there
        data_dir = Path(DATA)
        data_dir.mkdir(parents=True, exist_ok=True)
        dest = data_dir / filename
        # if a file with same name exists, overwrite
        if Path(saved_path).resolve() != dest.resolve():
            shutil.copy(saved_path, str(dest))
        else:
            # already in place
            dest = Path(saved_path)

        pdf_name = Path(filename).stem
        # run pipeline (blocking inside background task)
        process_pdf_pipeline(pdf_path=dest, layout_result_save=True, preprocess=False)

        # optional: call chat_with_gemini on produced JSON if exists
        result_json = OUTPUT / pdf_name / f"{pdf_name}_all_results.json"
        if result_json.exists():
            try:
                chat_with_gemini(result_json)
            except Exception as e:
                print("chat_with_gemini error:", e)
    except Exception as e:
        print("Background processing error:", e)

def call_gemini(prompt: str, max_tokens: int = 512) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    print(response.text)
    return response.text

@app.get("/ping")
async def ping():
    return {"ok": True}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    print("reached")
    fname = os.path.basename(file.filename)
    saved_path = os.path.join(UPLOAD_DIR, fname)
    print(saved_path)
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # schedule background task to move file to DATA and run pipeline
    if background_tasks is not None:
        background_tasks.add_task(process_uploaded_file, saved_path, fname)
        return {"filename": fname, "status": "processing_started"}
    else:
        # fallback: run synchronously (not recommended for production)
        process_uploaded_file(saved_path, fname)
        return {"filename": fname, "status": "processed"}

@app.post("/analyze")
async def analyze(filename: str = Form(...)):
    path = os.path.join(DATA, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "file not found"}, status_code=404)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(path)
    else:
        text = extract_text_from_image(path)
    prompt = "Summarize and give key W-H questions answered for the document (no bold text)\n\n" + (text[:20000] or "")
    summary = call_gemini(prompt, max_tokens=512)
    summary_path = path + ".summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    return {"summary": summary, "summary_path": os.path.basename(summary_path)}

@app.post("/chat")
async def chat(filename: str = Form(...), message: str = Form(...)):
    path = os.path.join(DATA, filename)
    summary_path = path + ".summary.txt"
    context = ""
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            context = f.read()
    prompt = f"Context:\n{context}\n\nUser question:\n{message}\n\nAnswer concisely."
    answer = call_gemini(prompt, max_tokens=256)
    return {"answer": answer}