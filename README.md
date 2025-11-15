# DocuAgent — AI Solution to Structure Your Documents

DocuAgent is an intelligent document-processing pipeline that leverages advanced AI models to automatically extract, structure, and summarize content from PDF documents.

## Overview

DocuAgent combines state-of-the-art computer vision, deep learning, and natural language processing techniques to transform unstructured PDFs into structured, machine-readable formats.

Whether you're processing financial reports, invoices, technical papers, or complex handwritten forms, DocuAgent can:

- Extract text from documents
- Detect layout regions and document structure
- Recognize and convert tables to structured formats
- Produce structured outputs in JSON format
- Generate concise summaries

## Features

- **PDF Processing** – Convert PDFs to images and extract text
- **Layout Recognition** – Detect headers, paragraphs, tables, figures, and other document elements
- **Table Detection & Extraction** – Parse complex tables into structured formats
- **Flexible Output** – JSON format with layout, table structure, and content
- **Configurable Pipeline** – YAML/JSON-based configuration system
- **FastAPI Backend** – Real-time document processing and serving

## Project Structure
```
docuagent/
├── data/                          # PDFs, datasets (train/val/test splits)
├── src/                           # Source code
│   ├── preprocessing/             # PDF to image, OCR, text extraction
│   ├── models/                    # LayoutLM, TableFormer, summarizers
│   ├── utils/                     # Helper functions (logging, configs)
│   └── pipeline.py                # Main pipeline orchestration script
├── notebooks/                     # Jupyter/Colab notebooks for experiments
├── outputs/                       # Structured results (JSON, summaries, charts)
├── requirements.txt               # Project dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules (large files, venv, cache)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- For GPU acceleration: CUDA 11.0 or higher

### Setup Instructions

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd docuagent
```

#### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install Dependencies

The installation process requires two steps due to PaddlePaddle dependencies.

**Step 1: Install PaddlePaddle with GPU Support**

Visit the official PaddlePaddle installation page and select the version compatible with your CUDA installation:

https://www.paddlepaddle.org.cn/en/

Follow the installation command provided for your specific CUDA version.

**Step 2: Install Remaining Dependencies**

Before proceeding, verify your PaddlePaddle installation is compatible with PaddleOCR. Refer to the compatibility table:

- PaddleOCR 3.0.0 requires PaddleX 3.0.0 and PaddlePaddle ≥ 3.0.0
- PaddleOCR 3.1.x requires PaddleX ≥ 3.1.0, < 3.2.0 and PaddlePaddle ≥ 3.0.0

Full compatibility documentation: https://www.paddleocr.ai/main/en/version3.x/paddleocr_and_paddlex.html

Once PaddlePaddle is installed and verified, install the remaining dependencies:
```bash
pip install -r requirements/base.txt
```

## Demo

Watch the demonstration video: https://youtu.be/nZaeeclYW-0

## Usage

Documentation for running the pipeline and API usage coming soon.

