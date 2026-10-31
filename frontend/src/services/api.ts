const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

async function handleResponse(response: Response) {
  const data = await response.json().catch(() => ({}))

  if (!response.ok) {
    const message =
      data.detail ||
      data.error ||
      `Request failed with status ${response.status}`

    throw new Error(
      typeof message === 'string' ? message : JSON.stringify(message)
    )
  }

  return data
}

export async function uploadDocument(file: File) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/api/documents/upload`, {
    method: 'POST',
    body: formData,
  })

  return handleResponse(response)
}

export async function indexDocument(documentId: string) {
  const response = await fetch(
    `${API_BASE_URL}/api/documents/${documentId}/index`,
    { method: 'POST' }
  )

  return handleResponse(response)
}

export async function askQuestion(
  documentId: string,
  question: string,
  topK = 5
) {
  const response = await fetch(`${API_BASE_URL}/api/answer`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      document_id: documentId,
      question,
      top_k: topK,
    }),
  })

  return handleResponse(response)
}

export async function getHealth() {
  const response = await fetch(`${API_BASE_URL}/api/health`)
  return handleResponse(response)
}