
import re
import fitz
import pdfplumber


def clean_text(text: str) -> str:
    """Clean extracted PDF text without removing meaningful content."""

    text = text.replace("\x00", " ")

    # Normalize multiple spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf(pdf_path: str) -> list:
    """
    Extract page-wise text and tables from a PDF.

    Physical PDF page numbers are preserved using
    1-based indexing.
    """

    document_data = []

    # Extract text using PyMuPDF
    with fitz.open(pdf_path) as pdf:

        if pdf.is_encrypted:
            raise ValueError(
                "Encrypted PDFs are not supported yet."
            )

        for page_index, page in enumerate(pdf):

            text = page.get_text("text")
            text = clean_text(text)

            document_data.append({
                "page": page_index + 1,
                "text": text,
                "tables": [],
                "has_text": bool(text)
            })

    # Extract tables using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:

        for page_index, page in enumerate(pdf.pages):

            extracted_tables = page.extract_tables() or []

            for table_index, table in enumerate(extracted_tables):

                if not table:
                    continue

                rows = []

                for row in table:

                    if not row:
                        continue

                    cleaned_row = [
                        str(cell or "").strip()
                        for cell in row
                    ]

                    if any(cleaned_row):
                        rows.append(cleaned_row)

                if rows:
                    document_data[page_index]["tables"].append({
                        "table_number": table_index + 1,
                        "rows": rows
                    })

    return document_data


if __name__ == "__main__":

    import json
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python -m app.pdf_processor <pdf_path>"
        )
        raise SystemExit(1)

    result = extract_pdf(sys.argv[1])

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )
