
import json
import re
import uuid
from pathlib import Path
from typing import Dict

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.pdf_processor import extract_pdf
from app.services.answer_engine import generate_answer
from app.services.rag_engine import (
    DocumentRetriever,
    build_document_index,
)


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

STORAGE_DIR = BASE_DIR / "storage"

UPLOAD_DIR = STORAGE_DIR / "uploads"
PROCESSED_DIR = STORAGE_DIR / "processed"
INDEX_DIR = STORAGE_DIR / "indexes"

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


# Create required directories
for directory in [
    UPLOAD_DIR,
    PROCESSED_DIR,
    INDEX_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ==================================================
# 2. IN-MEMORY RETRIEVER CACHE
# ==================================================

# Keeps indexed documents in memory so FAISS,
# chunks, and BM25 are not reloaded for every query.
retriever_cache: Dict[str, DocumentRetriever] = {}


def get_retriever(
    document_id: str,
) -> DocumentRetriever:
    """
    Return a cached DocumentRetriever.

    If the retriever is not already loaded,
    load it once and store it in memory.
    """

    if document_id not in retriever_cache:

        retriever_cache[document_id] = (
            DocumentRetriever(document_id)
        )

    return retriever_cache[document_id]


def clear_retriever_cache(
    document_id: str,
):
    """
    Remove a document retriever from cache.

    Used after re-indexing so the next query
    loads the newest version.
    """

    retriever_cache.pop(
        document_id,
        None,
    )


# ==================================================
# 3. FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Report Navigator",
    description=(
        "Evidence-first government statistics "
        "report analysis using Hybrid RAG."
    ),
    version="1.0.0",
)


# ==================================================
# 4. CORS CONFIGURATION
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# 5. REQUEST MODELS
# ==================================================

class QueryRequest(BaseModel):

    document_id: str

    question: str = Field(
        min_length=3,
        max_length=1000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=15,
    )


# ==================================================
# 6. HELPER FUNCTIONS
# ==================================================

def validate_document_id(
    document_id: str,
):
    """
    Validate the document UUID.
    """

    if re.fullmatch(r"[a-f0-9]{8}-[a-f0-9]{3}", document_id):
        return

    try:
        uuid.UUID(document_id)

    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid document ID.",
        )


def get_processed_document(
    document_id: str,
):
    """
    Load processed page-wise document data.
    """

    validate_document_id(document_id)

    file_path = (
        PROCESSED_DIR /
        f"{document_id}.json"
    )

    if not file_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    try:

        return json.loads(
            file_path.read_text(
                encoding="utf-8",
            )
        )

    except json.JSONDecodeError:

        raise HTTPException(
            status_code=500,
            detail=(
                "Stored document data is corrupted."
            ),
        )


def document_index_exists(
    document_id: str,
) -> bool:
    """
    Check whether a document has been indexed.
    """

    document_dir = (
        INDEX_DIR / document_id
    )

    vector_file = (
        document_dir / "vectors.faiss"
    )

    chunk_file = (
        document_dir / "chunks.json"
    )

    return (
        vector_file.exists()
        and chunk_file.exists()
    )


def format_evidence(
    evidence,
):
    """
    Format retrieval results for API responses.
    """

    formatted = []

    for item in evidence:

        result = {
            "chunk_id": item.get("chunk_id"),
            "document_id": item.get(
                "document_id"
            ),
            "page": item.get("page"),
            "content": item.get(
                "content"
            ),
            "source_type": item.get(
                "source_type"
            ),
            "rrf_score": round(
                float(
                    item.get(
                        "rrf_score",
                        0,
                    )
                ),
                6,
            ),
        }

        # Preserve table metadata when available
        if "table_number" in item:
            result["table_number"] = (
                item["table_number"]
            )

        if "row_number" in item:
            result["row_number"] = (
                item["row_number"]
            )

        formatted.append(result)

    return formatted


# ==================================================
# 7. HOME ROUTE
# ==================================================

@app.get("/")
def home():

    return {
        "project": "Report Navigator",
        "description": (
            "Evidence-first government report "
            "intelligence system"
        ),
        "version": "1.0.0",
        "status": "running",
    }


# ==================================================
# 8. HEALTH CHECK
# ==================================================

@app.get("/api/health")
def health():

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
        ],
    }


# ==================================================
# 9. UPLOAD PDF DOCUMENT
# ==================================================

@app.post("/api/documents/upload")
async def upload_pdf(
    file: UploadFile = File(...),
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is missing.",
        )

    if not file.filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF files are supported."
            ),
        )

    # Use full UUID for robust validation
    document_id = str(uuid.uuid4())

    file_path = (
        UPLOAD_DIR /
        f"{document_id}.pdf"
    )

    try:

        # ----------------------------------------------
        # Read uploaded file
        # ----------------------------------------------

        content = await file.read()

        if not content:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        if len(content) > MAX_FILE_SIZE:

            raise HTTPException(
                status_code=413,
                detail=(
                    "PDF must be smaller than "
                    "50 MB."
                ),
            )

        # ----------------------------------------------
        # Basic PDF validation
        # ----------------------------------------------

        if not content.startswith(b"%PDF-"):

            raise HTTPException(
                status_code=400,
                detail="Invalid PDF file.",
            )

        # ----------------------------------------------
        # Save uploaded PDF
        # ----------------------------------------------

        file_path.write_bytes(content)

        # ----------------------------------------------
        # Extract pages and tables
        # ----------------------------------------------

        pages = extract_pdf(
            str(file_path)
        )

        if not pages:

            raise HTTPException(
                status_code=422,
                detail=(
                    "No pages found in PDF."
                ),
            )

        # ----------------------------------------------
        # Calculate extraction statistics
        # ----------------------------------------------

        total_text_pages = sum(
            bool(
                page.get(
                    "has_text",
                    False,
                )
            )
            for page in pages
        )

        total_tables = sum(
            len(
                page.get(
                    "tables",
                    [],
                )
            )
            for page in pages
        )

        scanned_pages = (
            len(pages) - total_text_pages
        )

        # Do NOT crash simply because some pages
        # have no extractable text.
        #
        # This allows mixed scanned/text PDFs.
        # If the entire report has no searchable
        # content, indexing will safely fail later.

        # ----------------------------------------------
        # Save extracted document
        # ----------------------------------------------

        output_path = (
            PROCESSED_DIR /
            f"{document_id}.json"
        )

        output_path.write_text(
            json.dumps(
                pages,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "success": True,
            "document_id": document_id,
            "filename": file.filename,
            "page_count": len(pages),
            "text_pages": total_text_pages,
            "scanned_pages": scanned_pages,
            "table_count": total_tables,
            "file_size_bytes": len(content),
            "indexed": False,
            "message": (
                "PDF uploaded and processed "
                "successfully."
            ),
        }

    except HTTPException:

        file_path.unlink(
            missing_ok=True
        )

        raise

    except Exception as error:

        file_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=422,
            detail=(
                "PDF processing failed: "
                f"{str(error)}"
            ),
        )

    finally:

        await file.close()


# ==================================================
# 10. GET COMPLETE DOCUMENT
# ==================================================

@app.get(
    "/api/documents/{document_id}"
)
def get_document(
    document_id: str,
):

    pages = get_processed_document(
        document_id
    )

    return {
        "success": True,
        "document_id": document_id,
        "page_count": len(pages),
        "pages": pages,
    }


# ==================================================
# 11. GET SPECIFIC PDF PAGE
# ==================================================

@app.get(
    "/api/documents/"
    "{document_id}/pages/{page_number}"
)
def get_document_page(
    document_id: str,
    page_number: int,
):

    pages = get_processed_document(
        document_id
    )

    if (
        page_number < 1
        or page_number > len(pages)
    ):

        raise HTTPException(
            status_code=404,
            detail="Page not found.",
        )

    return {
        "success": True,
        "document_id": document_id,
        **pages[page_number - 1],
    }


# ==================================================
# 12. BUILD HYBRID RAG INDEX
# ==================================================

@app.post(
    "/api/documents/{document_id}/index"
)
def index_document(
    document_id: str,
):

    try:

        # ----------------------------------------------
        # Load processed document
        # ----------------------------------------------

        pages = get_processed_document(
            document_id
        )

        # ----------------------------------------------
        # Check whether there is searchable content
        # ----------------------------------------------

        has_searchable_content = any(
            page.get("text", "").strip()
            or page.get("tables")
            for page in pages
        )

        if not has_searchable_content:

            raise HTTPException(
                status_code=422,
                detail=(
                    "No searchable text or tables "
                    "were found in this PDF."
                ),
            )

        # ----------------------------------------------
        # Clear old retriever
        # ----------------------------------------------

        clear_retriever_cache(
            document_id
        )

        # ----------------------------------------------
        # Build FAISS index
        # ----------------------------------------------

        result = build_document_index(
            document_id=document_id,
            pages=pages,
        )

        return {
            "success": True,
            "message": (
                "Hybrid RAG index created "
                "successfully."
            ),
            "document_id": document_id,
            "total_chunks": result[
                "total_chunks"
            ],
            "text_chunks": result[
                "text_chunks"
            ],
            "table_chunks": result[
                "table_chunks"
            ],
            "index_type": [
                "FAISS",
                "BM25",
                "RRF",
            ],
        }

    except HTTPException:
        raise

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Indexing failed: "
                f"{str(error)}"
            ),
        )


# ==================================================
# 13. HYBRID RAG EVIDENCE QUERY
# ==================================================

@app.post("/api/query")
def query_document(
    request: QueryRequest,
):

    try:

        # Validate document exists
        get_processed_document(
            request.document_id
        )

        if not document_index_exists(
            request.document_id
        ):

            raise HTTPException(
                status_code=409,
                detail=(
                    "Document has not been indexed yet. "
                    "Call the index endpoint first."
                ),
            )

        # Load cached retriever
        retriever = get_retriever(
            request.document_id
        )

        # Retrieve evidence
        evidence = retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
        )

        formatted_evidence = (
            format_evidence(evidence)
        )

        return {
            "success": True,
            "document_id": (
                request.document_id
            ),
            "question": request.question,
            "total_results": len(
                formatted_evidence
            ),
            "retrieval_method": (
                "Hybrid RAG "
                "(FAISS + BM25 + RRF)"
            ),
            "evidence": formatted_evidence,
        }

    except HTTPException:
        raise

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Search failed: "
                f"{str(error)}"
            ),
        )


# ==================================================
# 14. ANSWER QUESTION
# ==================================================

@app.post("/api/answer")
async def answer_question(
    request: QueryRequest,
):

    try:

        # Validate document
        get_processed_document(
            request.document_id
        )

        if not document_index_exists(
            request.document_id
        ):

            raise HTTPException(
                status_code=409,
                detail=(
                    "Document has not been indexed yet. "
                    "Call the index endpoint first."
                ),
            )

        # Retrieve evidence
        retriever = get_retriever(
            request.document_id
        )

        evidence = retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
        )

        # Generate grounded answer
        result = await generate_answer(
            question=request.question,
            evidence=evidence,
        )

        return {
            "success": True,
            "document_id": (
                request.document_id
            ),
            "question": request.question,
            "answer": result["answer"],
            "citations": result["citations"],
            "status": result["status"],
            "evidence": format_evidence(
                evidence
            ),
        }

    except HTTPException:
        raise

    except RuntimeError as error:

        raise HTTPException(
            status_code=503,
            detail=str(error),
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Answer generation failed: "
                f"{str(error)}"
            ),
        )


# ==================================================
# 15. APPLICATION STARTUP
# ==================================================

@app.on_event("startup")
async def startup_event():

    print("=" * 60)

    print("REPORT NAVIGATOR BACKEND")

    print("Status: Running")

    print("PDF Processing: Enabled")

    print("Table Extraction: Enabled")

    print("FAISS: Enabled")

    print("BM25: Enabled")

    print("Hybrid Retrieval: Enabled")

    print("Evidence Citations: Enabled")

    print("=" * 60)
