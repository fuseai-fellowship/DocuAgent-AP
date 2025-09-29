**DocuAgent** — an AI solution to structurize your docs.

*Folder Structure* 

docuagent/
│
├── data/                # PDFs, datasets (train/val/test splits later)
│   ├── raw/             # original PDFs
│   ├── processed/       # preprocessed JSON, OCR outputs
│
├── src/                 # source code
│   ├── preprocessing/   # pdf2image, OCR, text extraction
│   ├── models/          # LayoutLM, TableFormer, summarizers
│   ├── utils/           # helper functions (logging, configs)
│   └── pipeline.py      # main pipeline script
│
├── notebooks/           # Jupyter/Colab notebooks for experiments
│
├── outputs/             # structured results (JSON, summaries, charts)
│
├── configs/             # YAML/JSON configs (paths, hyperparams, model settings)
│
├── tests/               # unit tests for each module
│
├── requirements.txt     # dependencies (pdfplumber, paddleocr, torch, transformers…)
├── README.md            # project description & setup guide
├── .gitignore           # ignore big files, venv, cache