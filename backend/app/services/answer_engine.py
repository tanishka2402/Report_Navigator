import httpx

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"


async def generate_answer(question: str, evidence: list):
    if not evidence:
        return {
            "answer": "Not found in report.",
            "citations": [],
            "status": "not_found"
        }

    context_parts = []
    valid_pages = set()

    for item in evidence:
        page = item.get("page", "Unknown")
        content = item.get("content", "")

        if page != "Unknown":
            valid_pages.add(int(page))

        context_parts.append(
            f"[Page {page}]\n{content}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are Report Navigator, an assistant that answers questions
using only the supplied report evidence.

Rules:
1. Use only the evidence below.
2. Do not use outside knowledge or invent facts.
3. If the evidence does not answer the question, reply exactly:
   NOT_FOUND
4. Include page citations for factual claims, formatted as [Page N].
5. For numerical answers, preserve the exact numbers and units.
6. Be concise and clear.

REPORT EVIDENCE:
{context}

QUESTION:
{question}

ANSWER:
"""

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False
                }
            )

        response.raise_for_status()
        answer = response.json()["message"]["content"].strip()

        if answer == "NOT_FOUND":
            return {
                "answer": "Not found in report.",
                "citations": [],
                "status": "not_found"
            }

        # Accept citations only when they refer to retrieved pages.
        import re

        cited_pages = sorted({
            int(page)
            for page in re.findall(r"\[Page\s+(\d+)\]", answer)
            if int(page) in valid_pages
        })

        # Abstain if the model cites pages that were not retrieved,
        # or fails to provide any valid page citation.
        all_citations = [
            int(page)
            for page in re.findall(r"\[Page\s+(\d+)\]", answer)
        ]

        if not cited_pages or len(cited_pages) != len(set(all_citations)):
            return {
                "answer": "I couldn't verify an answer from the retrieved report evidence.",
                "citations": [],
                "status": "not_found"
            }

        return {
            "answer": answer,
            "citations": cited_pages,
            "status": "answered"
        }

    except httpx.HTTPError as exc:
        raise RuntimeError(f"Could not connect to Ollama: {exc}")