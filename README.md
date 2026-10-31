# Report Navigator

Report Navigator is an AI-powered application for uploading PDF reports,
indexing their contents, and asking natural-language questions about the
information in those reports.

## Problem Statement

Reports are often long and difficult to review manually. Finding a
specific fact, understanding a section, or answering questions across a
document can take significant time. Report Navigator aims to make report
exploration easier by allowing users to upload a PDF and ask questions
about its contents, with answers generated using retrieval-augmented
generation (RAG).

## Solution Overview

The application provides a web interface for document upload and
question answering. The backend extracts text from PDFs, divides the
extracted content into chunks, creates vector embeddings, and combines
semantic retrieval with keyword retrieval. Retrieved context is then
used to generate an answer through Google's Gemini API.

## Architecture

``` text
+-----------------------------+
| React + TypeScript + Vite   |
| Frontend                    |
+--------------+--------------+
               |
               | HTTP requests
               | VITE_API_BASE_URL
               v
+-----------------------------+
| FastAPI Backend             |
| PDF upload and processing  |
| Document indexing          |
| Retrieval and answer API   |
+------+----------------------+
       |
       +------------------------------+
       |                              |
       v                              v
+----------------------+    +-------------------------+
| PDF text extraction  |    | Google Gemini API       |
| PyMuPDF / pdfplumber |    | Embeddings and answers   |
+----------------------+    +-------------------------+
       |
       v
+-----------------------------------------------+
| Retrieval pipeline                            |
| Chunking -> Gemini embeddings -> FAISS search  |
|                         + BM25 keyword search  |
|                         -> Reciprocal Rank     |
|                            Fusion              |
+-----------------------------------------------+
       |
       v
+-----------------------------+
| Relevant document context   |
| passed to answer generation |
+-----------------------------+
```

### Main Technologies

-   **Frontend:** React, TypeScript, Vite, React Router, Lucide React.
-   **Backend:** Python, FastAPI, Uvicorn.
-   **PDF processing:** PyMuPDF and pdfplumber.
-   **Embeddings:** Google Gemini embedding API
    (`gemini-embedding-001`).
-   **Retrieval:** FAISS vector search, BM25 keyword search, and
    Reciprocal Rank Fusion.
-   **Answer generation:** Google Gemini API.
-   **Deployment:** Render.

### Request Flow

1.  The user uploads a PDF through the frontend.
2.  The frontend sends the file to the backend upload endpoint.
3.  The backend extracts text and creates document chunks.
4.  Gemini generates embeddings for the chunks.
5.  The backend builds a FAISS index from the vectors and retains the
    document chunks.
6.  When the user asks a question, the backend embeds the query and
    performs semantic and keyword retrieval.
7.  Reciprocal Rank Fusion combines the retrieval results.
8.  Relevant context is sent to Gemini to generate an answer.

## Deployed Application and API

-   **Backend base URL:** https://report-navigator.onrender.com
-   **Swagger / OpenAPI documentation:**
    https://report-navigator.onrender.com/docs
-   **Backend status endpoint:** https://report-navigator.onrender.com

**Frontend URL:** Add the Render Static Site URL here after frontend
deployment is complete.

## API Endpoints

The frontend currently calls these backend routes:

  -------------------------------------------------------------------------------------
  Method                  Endpoint                              Purpose
  ----------------------- ------------------------------------- -----------------------
  `POST`                  `/api/documents/upload`               Upload a PDF document.

  `POST`                  `/api/documents/{documentId}/index`   Trigger or access
                                                                document indexing.

  `POST`                  `/api/answer`                         Submit a question and
                                                                receive an answer.

  `GET`                   `/api/health`                         Check backend health.
  -------------------------------------------------------------------------------------

For authoritative request schemas, response formats, and interactive
testing, see the [Swagger API
documentation](https://report-navigator.onrender.com/docs).

## Local Setup

### Prerequisites

-   Python 3.12 or a compatible Python version
-   Node.js and npm
-   A Google Gemini API key

### 1. Clone the Repository

``` bash
git clone https://github.com/tanishka2402/Report_Navigator.git
cd Report_Navigator
```

### 2. Configure and Run the Backend

``` bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file inside `backend`:

``` env
GEMINI_API_KEY=your_gemini_api_key
```

Start the API:

``` bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Backend URL: `http://localhost:8000`\
Swagger UI: `http://localhost:8000/docs`

### 3. Configure and Run the Frontend

In a second terminal:

``` bash
cd frontend
npm install
```

Create a `.env` file inside `frontend`:

``` env
VITE_API_BASE_URL=http://localhost:8000
```

Start the development server:

``` bash
npm run dev
```

Create a production build:

``` bash
npm run build
```

Vite writes the production build to `frontend/dist`.

## Deployment Configuration

The backend is deployed as a Render Web Service. Its start command is:

``` bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

The frontend can be deployed as a Render Static Site with:

  -----------------------------------------------------------------------------------------------
  Setting                             Value
  ----------------------------------- -----------------------------------------------------------
  Root Directory                      `frontend`

  Build Command                       `npm install && npm run build`

  Publish Directory                   `dist`

  Environment Variable                `VITE_API_BASE_URL=https://report-navigator.onrender.com`
  -----------------------------------------------------------------------------------------------

Set the Gemini API key as a secret environment variable in the backend
service. Do not commit `.env` files or API keys.

## Notes

-   The backend root endpoint returns a JSON service-status response.
    Interactive API documentation is at `/docs`.
-   The frontend deployment URL is left as a placeholder until the
    Render Static Site is created.
-   Gemini API calls require a valid API key and are subject to
    applicable quotas and usage terms.
-   If the hosting environment uses an ephemeral filesystem, uploaded
    documents and generated indexes may not persist across restarts or
    redeployments unless persistent storage is configured.

## Repository

[GitHub:
tanishka2402/Report_Navigator](https://github.com/tanishka2402/Report_Navigator)
