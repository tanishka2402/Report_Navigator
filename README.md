
# Report Navigator

### AI-Powered Intelligence for Government Statistics Reports

Report Navigator is an evidence-first AI assistant that helps users
find accurate answers from lengthy government statistics reports.

Instead of manually searching hundreds of PDF pages, users can ask
natural-language questions and receive answers supported by
retrieved evidence and original PDF page citations.

## Problem

Government reports contain valuable information across thousands
of paragraphs, tables, and statistical datasets.

Finding specific facts and verifying numerical answers manually
is time-consuming.

Report Navigator addresses this problem through document-aware
retrieval and evidence-grounded answer generation.

## Key Features

- PDF document upload and processing
- Page-wise text extraction
- Table extraction and table-aware chunking
- Semantic search using FAISS
- Keyword search using BM25
- Hybrid retrieval using Reciprocal Rank Fusion
- Evidence-based answer generation
- PDF page-level citations
- Abstention when evidence is insufficient

## Technology Stack

### Backend
- Python
- FastAPI
- PyMuPDF
- pdfplumber
- Sentence Transformers
- FAISS
- BM25
- Ollama

### Frontend
- React / Vite (in development)

## Architecture

PDF Report
    |
    v
PDF Processing
    |
    v
Text and Table Extraction
    |
    v
Document Chunking
    |
    +------------------+
    |                  |
    v                  v
FAISS Search       BM25 Search
    |                  |
    +--------+---------+
             |
             v
     Reciprocal Rank Fusion
             |
             v
       Retrieved Evidence
             |
             v
       Answer Generation
             |
             v
   Answer + Page Citations

## Project Structure

backend/       Backend application
frontend/      User interface
docs/          Project documentation

## Getting Started

Backend setup instructions will be documented here
as the application structure is finalized.

## Development Status

- [x] PDF extraction
- [x] Page-level metadata preservation
- [x] Table extraction
- [x] Document chunking
- [x] FAISS indexing
- [x] BM25 retrieval
- [x] Hybrid retrieval with RRF
- [x] FastAPI backend
- [x] Evidence retrieval API
- [x] LLM answer generation
- [ ] Frontend integration
- [ ] Automated evaluation pipeline
- [ ] Comprehensive test suite

## Hackathon Project

Built for Hacktoberfest.

## License

To be determined.