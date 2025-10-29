# DocuAgent — AI Solution to Structurize Your Docs

DocuAgent is an intelligent document processing pipeline that leverages advanced AI models to automatically extract, structure, and summarize content from PDF documents.

## Overview

DocuAgent combines cutting-edge computer vision and natural language processing to transform unstructured PDFs into organized, machine-readable formats. Whether you're dealing with financial reports, technical documentation, or complex forms, DocuAgent extracts layout information, recognizes tables, and generates intelligent summaries.

## Features

- **PDF Processing**: Convert PDFs to images and extract text using OCR
- **Layout Recognition**: Identify document structure 
- **Table Detection & Extraction**: Automatically recognize and parse tables 
- **Flexible Output**: Structured JSON
- **Configurable Pipeline**: YAML/JSON configuration for customizable processing

## Project Structure

```
docuagent/
├── data/                          # PDFs, datasets (train/val/test splits)
│   ├── raw/                       # Original PDF files
│   └── processed/                 # Preprocessed JSON, OCR outputs
├── src/                           # Source code
│   ├── preprocessing/             # PDF to image, OCR, text extraction
│   ├── models/                    # LayoutLM, TableFormer, summarizers
│   ├── utils/                     # Helper functions (logging, configs)
│   └── pipeline.py                # Main pipeline orchestration script
├── notebooks/                     # Jupyter/Colab notebooks for experiments
├── outputs/                       # Structured results (JSON, summaries, charts)
├── configs/                       # YAML/JSON configuration files
│                                  # (paths, hyperparameters, model settings)
├── tests/                         # Unit tests for each module
├── requirements.txt               # Project dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules (large files, venv, cache)
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda
- For GPU acceleration: CUDA 11.0+

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd docuagent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

Key dependencies include:

- **PyMuPDF** — PDF to image conversion
- **paddleocr** — Optical character recognition and Layout Models

For the complete list, see `requirements.txt`.

## Demo Video

[Watch the demo here](https://youtu.be/Ws5KonFE9iM)



