
import json
import logging
import os
import uuid

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.pdf_processor import extract_pdf

from app.services.rag_engine import (
    build_document_index,
    DocumentRetriever
)

from app.services.answer_engine import generate_answer


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Application configuration
# ---------------------------------------------------------

app = FastAPI(
    title="Report Navigator API",
    description=(
        "Evidence-grounded question answering "
        "for long statistical PDF reports."
    ),
    version="2.0.0"
)


# ---------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------

FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Storage configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

STORAGE_DIR = BASE_DIR / "storage"

UPLOAD_DIR = STORAGE_DIR / "uploads"

PROCESSED_DIR = STORAGE_DIR / "processed"

INDEX_DIR = STORAGE_DIR / "indexes"

MAX_UPLOAD_SIZE = 50 * 1024 * 1024

ALLOWED_EXTENSION = ".pdf"

for directory in [
    UPLOAD_DIR,
    PROCESSED_DIR,
    INDEX_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ---------------------------------------------------------
# Retriever cache
# ---------------------------------------------------------

retriever_cache: Dict[str, DocumentRetriever] = {}


def validate_document_id(document_id: str) -> str:
    """
    Validate and normalize document UUID.
    """

    try:
        return str(uuid.UUID(document_id))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid document ID."
        )


def get_processed_path(document_id: str) -> Path:

    return PROCESSED_DIR / f"{document_id}.json"


def get_retriever(document_id: str) -> DocumentRetriever:
    """
    Load a retriever once and reuse it for subsequent queries.
    """

    document_id = validate_document_id(document_id)

    if document_id not in retriever_cache:

        processed_path = get_processed_path(document_id)

        if not processed_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Processed document not found."
            )

        try:
            retriever_cache[document_id] = (
                DocumentRetriever(document_id)
            )

        except Exception as exc:

            logger.exception(
                "Failed to load document retriever."
            )

            raise HTTPException(
                status_code=500,
                detail="Document index could not be loaded."
            ) from exc

    return retriever_cache[document_id]


def clear_retriever_cache(document_id: str) -> None:

    retriever_cache.pop(document_id, None)


# ---------------------------------------------------------
# Document loading helpers
# ---------------------------------------------------------


def load_processed_document(
    document_id: str
) -> Dict[str, Any]:
    """
    Load processed document data.

    Supports both:
    1. New dictionary format containing pages and metadata.
    2. Legacy list format containing only extracted pages.
    """

    document_id = validate_document_id(document_id)
    processed_path = get_processed_path(document_id)

    if not processed_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    try:
        with open(
            processed_path,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        # Handle older files where JSON contains
        # a list of pages instead of a dictionary.
        if isinstance(data, list):
            pages = data

            page_count = len(pages)

            text_pages = sum(
                1
                for page in pages
                if isinstance(page, dict)
                and get_page_text(page).strip()
            )

            scanned_pages = page_count - text_pages

            table_count = sum(
                len(page.get("tables", []) or [])
                for page in pages
                if isinstance(page, dict)
            )

            # Check whether the existing index is present.
            index_path = INDEX_DIR / document_id

            indexed = (
                (index_path / "vectors.faiss").is_file()
                and (index_path / "chunks.json").is_file()
            )

            data = {
                "document_id": document_id,
                "filename": None,
                "pages": pages,
                "page_count": page_count,
                "text_pages": text_pages,
                "scanned_pages": scanned_pages,
                "table_count": table_count,
                "indexed": indexed
            }

        elif not isinstance(data, dict):
            raise HTTPException(
                status_code=500,
                detail="Invalid processed document format."
            )

        return data

    except json.JSONDecodeError as exc:
        logger.exception(
            "Processed document JSON is invalid."
        )

        raise HTTPException(
            status_code=500,
            detail="Processed document data is corrupted."
        ) from exc



def get_page_text(page: Dict[str, Any]) -> str:

    return str(
        page.get(
            "text",
            page.get("content", page.get("page_text", ""))
        ) or ""
    )


def get_page_number(
    page: Dict[str, Any],
    fallback: int
) -> int:

    try:
        return int(page.get("page", fallback))
    except (TypeError, ValueError):
        return fallback


# ---------------------------------------------------------
# Request models
# ---------------------------------------------------------

class QueryRequest(BaseModel):

    document_id: str

    question: str = Field(
        min_length=3,
        max_length=1000
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=15
    )


# ---------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "service": "Report Navigator Backend",
        "version": "2.0.0",
        "status": "running",
        "documentation": "/docs"
    }


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/api/health")
def health_check():

    return {
        "status": "healthy",
        "service": "Report Navigator Backend",
        "features": [
            "PDF extraction",
            "Table extraction",
            "Page-aware citations",
            "Hybrid RAG",
            "FAISS semantic search",
            "BM25 keyword search",
            "Reciprocal Rank Fusion",
            "Evidence retrieval",
            "Gemini answer generation",
            "Citation validation",
            "Abstention for missing information"
        ]
    }


# ---------------------------------------------------------
# Upload and process PDF
# ---------------------------------------------------------

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    """
    Upload a PDF and extract page-wise text and tables.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename supplied."
        )

    original_filename = Path(file.filename).name

    if Path(original_filename).suffix.lower() != ALLOWED_EXTENSION:
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    document_id = str(uuid.uuid4())

    upload_path = (
        UPLOAD_DIR / f"{document_id}.pdf"
    )

    processed_path = get_processed_path(document_id)

    try:

        # Read one byte beyond the limit to detect oversized files.
        file_content = await file.read(
            MAX_UPLOAD_SIZE + 1
        )

        if len(file_content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail="PDF exceeds the 50 MB upload limit."
            )

        if not file_content.startswith(b"%PDF"):
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid PDF."
            )

        with open(upload_path, "wb") as output_file:
            output_file.write(file_content)

        # Extract PDF content.
        extracted_data = extract_pdf(
            str(upload_path)
        )

        if isinstance(extracted_data, list):
            extracted_data = {
                "pages": extracted_data
            }

        if not isinstance(extracted_data, dict):
            raise ValueError(
                "PDF processor returned an unexpected result."
            )

        pages = extracted_data.get("pages", [])

        if not isinstance(pages, list) or not pages:
            raise ValueError(
                "No pages were extracted from the PDF."
            )

        page_count = len(pages)

        text_pages = sum(
            1
            for page in pages
            if get_page_text(page).strip()
        )

        scanned_pages = page_count - text_pages

        table_count = sum(
            len(page.get("tables", []) or [])
            for page in pages
        )

        # Preserve all extracted data and add document metadata.
        processed_data = {
            **extracted_data,
            "document_id": document_id,
            "filename": original_filename,
            "page_count": page_count,
            "text_pages": text_pages,
            "scanned_pages": scanned_pages,
            "table_count": table_count,
            "indexed": False
        }

        with open(
            processed_path,
            "w",
            encoding="utf-8"
        ) as json_file:

            json.dump(
                processed_data,
                json_file,
                ensure_ascii=False,
                indent=2
            )

        logger.info(
            "PDF processed successfully: %s",
            document_id
        )

        return {
            "success": True,
            "document_id": document_id,
            "filename": original_filename,
            "page_count": page_count,
            "text_pages": text_pages,
            "scanned_pages": scanned_pages,
            "table_count": table_count,
            "file_size_bytes": len(file_content),
            "indexed": False,
            "message": "PDF uploaded and processed successfully."
        }

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "PDF upload or processing failed."
        )

        upload_path.unlink(missing_ok=True)
        processed_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(exc)}"
        ) from exc

    finally:
        await file.close()


# ---------------------------------------------------------
# Get processed document information
# ---------------------------------------------------------

@app.get("/api/documents/{document_id}")
def get_document(document_id: str):

    document_id = validate_document_id(document_id)

    data = load_processed_document(document_id)

    return {
        "success": True,
        "document_id": document_id,
        "filename": data.get("filename"),
        "page_count": data.get("page_count"),
        "text_pages": data.get("text_pages"),
        "scanned_pages": data.get("scanned_pages"),
        "table_count": data.get("table_count"),
        "indexed": data.get("indexed", False)
    }


# ---------------------------------------------------------
# Get individual PDF page
# ---------------------------------------------------------

@app.get(
    "/api/documents/{document_id}/pages/{page_number}"
)
def get_document_page(
    document_id: str,
    page_number: int
):

    document_id = validate_document_id(document_id)

    if page_number < 1:
        raise HTTPException(
            status_code=400,
            detail="Page number must start from 1."
        )

    data = load_processed_document(document_id)

    pages = data.get("pages", [])

    for index, page in enumerate(pages, start=1):

        current_page = get_page_number(
            page,
            index
        )

        if current_page == page_number:

            return {
                "success": True,
                "document_id": document_id,
                "page": current_page,
                "text": get_page_text(page),
                "tables": page.get("tables", []),
                "has_text": bool(
                    get_page_text(page).strip()
                )
            }

    raise HTTPException(
        status_code=404,
        detail="Requested PDF page not found."
    )


# ---------------------------------------------------------
# Build hybrid RAG index
# ---------------------------------------------------------

@app.post(
    "/api/documents/{document_id}/index"
)
def index_document(document_id: str):
    """
    Build FAISS + BM25 + RRF index for a document.
    """

    document_id = validate_document_id(document_id)

    data = load_processed_document(document_id)

    pages = data.get("pages", [])

    if not pages:
        raise HTTPException(
            status_code=400,
            detail="Document contains no extracted pages."
        )

    clear_retriever_cache(document_id)

    try:

        index_result = build_document_index(
            document_id,
            pages
        )

        # Mark the processed document as indexed.
        data["indexed"] = True

        processed_path = get_processed_path(document_id)

        with open(
            processed_path,
            "w",
            encoding="utf-8"
        ) as json_file:

            json.dump(
                data,
                json_file,
                ensure_ascii=False,
                indent=2
            )

        # Load the retriever after index creation.
        retriever_cache[document_id] = (
            DocumentRetriever(document_id)
        )

        # Support optional index statistics returned by
        # build_document_index().
        stats = (
            index_result
            if isinstance(index_result, dict)
            else {}
        )

        return {
            "success": True,
            "message": "Hybrid RAG index created successfully.",
            "document_id": document_id,
            "total_chunks": stats.get("total_chunks"),
            "text_chunks": stats.get("text_chunks"),
            "table_chunks": stats.get("table_chunks"),
            "index_type": [
                "FAISS",
                "BM25",
                "RRF"
            ]
        }

    except HTTPException:
        raise

    except Exception as exc:

        clear_retriever_cache(document_id)

        logger.exception(
            "Document indexing failed."
        )

        raise HTTPException(
            status_code=500,
            detail=f"Index creation failed: {str(exc)}"
        ) from exc


# ---------------------------------------------------------
# Internal evidence retrieval
# ---------------------------------------------------------

def retrieve_document_evidence(
    document_id: str,
    question: str,
    top_k: int
) -> List[Dict[str, Any]]:

    document_id = validate_document_id(document_id)

    data = load_processed_document(document_id)

    if not data.get("indexed", False):
        raise HTTPException(
            status_code=400,
            detail="Document has not been indexed yet."
        )

    retriever = get_retriever(document_id)

    evidence = retriever.retrieve(
        question,
        top_k=top_k
    )

    if evidence is None:
        return []

    if not isinstance(evidence, list):
        raise HTTPException(
            status_code=500,
            detail="Retriever returned an invalid evidence format."
        )

    return evidence


# ---------------------------------------------------------
# Query endpoint: evidence retrieval only
# ---------------------------------------------------------

@app.post("/api/query")
def query_document(request: QueryRequest):
    """
    Retrieve relevant text and table evidence.
    Does not call the LLM.
    """

    document_id = validate_document_id(
        request.document_id
    )

    evidence = retrieve_document_evidence(
        document_id=document_id,
        question=request.question,
        top_k=request.top_k
    )

    return {
        "success": True,
        "document_id": document_id,
        "question": request.question,
        "total_results": len(evidence),
        "retrieval_method": "Hybrid RAG (FAISS + BM25 + RRF)",
        "evidence": evidence
    }


# ---------------------------------------------------------
# Answer endpoint: retrieval + Gemini
# ---------------------------------------------------------

@app.post("/api/answer")
async def answer_question(request: QueryRequest):
    """
    Retrieve evidence and generate a grounded answer
    using the Gemini API.
    """

    document_id = validate_document_id(
        request.document_id
    )

    evidence = retrieve_document_evidence(
        document_id=document_id,
        question=request.question,
        top_k=request.top_k
    )

    result = await generate_answer(
        request.question,
        evidence
    )

    return {
        "success": result["status"] != "error",
        "document_id": document_id,
        "question": request.question,
        "answer": result["answer"],
        "citations": result["citations"],
        "status": result["status"],
        "evidence": evidence
    }


# ---------------------------------------------------------
# Startup message
# ---------------------------------------------------------

@app.on_event("startup")
async def startup_event():

    logger.info(
        "Report Navigator backend started."
    )

    if not os.getenv("GEMINI_API_KEY"):
        logger.warning(
            "GEMINI_API_KEY is not configured. "
            "Answer generation will not work."
        )
