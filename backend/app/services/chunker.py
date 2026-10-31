import re
from typing import Any, Dict, List


def split_into_chunks(
    text: str,
    chunk_size: int = 220,
    overlap: int = 40
) -> List[str]:
    """
    Split text into overlapping word-based chunks.
    """

    words = text.split()

    if not words:
        return []

    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def normalize_cell(cell: Any) -> str:
    """
    Normalize a table cell while preserving meaningful content.
    """

    if cell is None:
        return ""

    text = str(cell)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def create_row_mapping(
    headers: List[str],
    row: List[str]
) -> Dict[str, str]:
    """
    Create structured header-to-value mapping.
    """

    row_data = {}

    for index, value in enumerate(row):

        if index < len(headers) and headers[index]:
            header = headers[index]
        else:
            header = f"Column {index + 1}"

        row_data[header] = value

    return row_data


def create_document_chunks(
    pages: List[Dict],
    document_id: str,
    text_chunk_size: int = 220,
    text_overlap: int = 40,
    table_chunk_size: int = 100
) -> List[Dict]:
    """
    Convert extracted PDF pages into searchable chunks.

    Creates:
    1. Page-aware text chunks
    2. Table row chunks with header context
    3. Table context chunks for multi-row questions

    Every chunk preserves its original PDF page number.
    """

    all_chunks = []
    chunk_number = 0

    for page_data in pages:

        page_number = page_data["page"]

        # ==================================================
        # 1. NORMAL TEXT
        # ==================================================

        page_text = page_data.get("text", "").strip()

        if page_text:

            text_parts = re.split(
                r"\n\s*\n+",
                page_text
            )

            for part in text_parts:

                part = part.strip()

                if not part:
                    continue

                text_chunks = split_into_chunks(
                    part,
                    chunk_size=text_chunk_size,
                    overlap=text_overlap
                )

                for content in text_chunks:

                    all_chunks.append({
                        "chunk_id": f"chunk_{chunk_number}",
                        "document_id": document_id,
                        "page": page_number,
                        "content": content,
                        "source_type": "text"
                    })

                    chunk_number += 1

        # ==================================================
        # 2. TABLES
        # ==================================================

        tables = page_data.get("tables", [])

        for table in tables:

            rows = table.get("rows", [])

            if not rows:
                continue

            cleaned_rows = []

            for row in rows:

                if not row:
                    continue

                cleaned_row = [
                    normalize_cell(cell)
                    for cell in row
                ]

                if any(cleaned_row):
                    cleaned_rows.append(cleaned_row)

            if len(cleaned_rows) < 2:
                continue

            headers = cleaned_rows[0]
            table_number = table.get("table_number", 1)

            # ----------------------------------------------
            # 2A. ROW-LEVEL CHUNKS
            # ----------------------------------------------

            table_context_rows = []

            for row_number, row in enumerate(
                cleaned_rows[1:],
                start=1
            ):

                if not any(row):
                    continue

                row_data = create_row_mapping(
                    headers,
                    row
                )

                row_values = [
                    f"{header}: {value}"
                    for header, value in row_data.items()
                ]

                row_content = (
                    f"Table {table_number}. "
                    f"Page {page_number}. "
                    + " | ".join(row_values)
                )

                all_chunks.append({
                    "chunk_id": f"chunk_{chunk_number}",
                    "document_id": document_id,
                    "page": page_number,
                    "content": row_content,
                    "source_type": "table_row",
                    "table_number": table_number,
                    "row_number": row_number,
                    "headers": headers,
                    "row_data": row_data
                })

                chunk_number += 1

                table_context_rows.append(
                    " | ".join(row_values)
                )

            # ----------------------------------------------
            # 2B. TABLE CONTEXT CHUNKS
            # ----------------------------------------------

            if table_context_rows:

                for start in range(
                    0,
                    len(table_context_rows),
                    table_chunk_size
                ):

                    batch = table_context_rows[
                        start:start + table_chunk_size
                    ]

                    context_content = (
                        f"Table {table_number}. "
                        f"Page {page_number}. "
                        f"Column headers: {' | '.join(headers)}.\n"
                        + "\n".join(batch)
                    )

                    all_chunks.append({
                        "chunk_id": f"chunk_{chunk_number}",
                        "document_id": document_id,
                        "page": page_number,
                        "content": context_content,
                        "source_type": "table_context",
                        "table_number": table_number,
                        "row_number": 0,
                        "headers": headers
                    })

                    chunk_number += 1

    return all_chunks