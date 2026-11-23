# Named Entity Reconigtion (NER): GLiNER, PymuPDF and LLM

A Named Entity Recognition (NER) project designed to extract structured entities from both plain text and PDF documents using zero-shot encoder models (**GLiNER**) and generative Large Language Models (**LLM**), evaluated under strict and relaxed.

## Project structure
```text
ner_project/
├── data/
│   ├── pdf_clean/          
│   └── pdf_scanned/            
│   └── plain/
│       └── validation.jsonl              
├── ner/
│   ├── __init__.py
│   ├── evaluation.py        
│   ├── inference.py        
│   ├── pdf.py              
│   └── utils.py            
├── notebooks/
│   ├── 01_plain_text.ipynb  
│   ├── 02_pdf_clean.ipynb       
│   ├── 03_pdf_scanned.ipynb        
│   ├── 04_llm_evaluation.ipynb               
├── results/
│   ├── llm/                
│       └── predictions,jsonl
│       └── predictions_formatted.jsonl
│   └── pdf_clean/          
│       └── predictions_gliner.jsonl
│   └── pdf_scanned/
│       └── gliner_large_validation.jsonl
│   └── plain/              
│       └── gliner_small_validation.jsonl 
├── .gitignore
├── pyproject.toml
└── README.md
```
## Entity Schema
The project extracts following entity categories in a zero-shot setting (using label names directly without extended descriptions):

º PERSON_NAME

º ID_CARD_NUMBER

º ADDRESS

º REGISTRATION_NUMBER

º DATE

º AMOUNT

º ORGANIZATION

## Pipeline Overview

**1. PDF Text Extraction (ner/pdf.py):**
Uses PyMuPDF (fitz) to extract raw text from PDF files (extract_text_from_pdf), falling back to plain text records if a physical PDF is unavailable.

**2. Zero-Shot NER with GLiNER (ner/inference.py):**
Loads gliner-community/gliner_small-v2.5 with a confidence threshold of 0.5 to predict entities along with exact character offsets (start, end).

**3. LLM Inference (Gemini AI):**
Extracts entities via structured JSON prompting in an LLM chat interface, saves outputs under results/llm/, and maps text spans back to character offsets (start, end) prior to evaluation.

**4. Evaluation (ner/evaluation.py):**
Computes micro and per-entity-type Precision, Recall, and F1-Score against the gold standard across two modes:
- Strict Evaluation (evaluate_ner_strict): Requires an exact match of (start, end, label).
- Relaxed Evaluation (evaluate_ner_relaxed): Ignores character offsets and normalizes whitespace (\s+), evaluating (label, normalized_text) across both occurrences and unique entities per document.