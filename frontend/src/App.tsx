import { useEffect, useRef, useState } from 'react'
import {
  BookOpen, FileText, Plus, Search, Settings, HelpCircle,
  MessageSquare, Upload, Sparkles, Send, Paperclip,
  CheckCircle2, Loader2, AlertCircle, FileUp, X,
} from 'lucide-react'

import {
  uploadDocument,
  indexDocument,
  askQuestion,
  getHealth,
} from './services/api'

type DocumentItem = {
  document_id: string
  name: string
  status: 'uploaded' | 'indexing' | 'indexed' | 'error'
}

type AnswerResult = {
  success?: boolean
  answer?: string
  status?: string
  citations?: number[]
  evidence?: Array<{
    chunk_id?: string
    page?: number
    source_type?: string
    text?: string
    content?: string
  }>
}

function App() {
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [documents, setDocuments] = useState<DocumentItem[]>([])
  const [activeDocumentId, setActiveDocumentId] = useState('')
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState<AnswerResult | null>(null)

  const [uploading, setUploading] = useState(false)
  const [asking, setAsking] = useState(false)
  const [backendOnline, setBackendOnline] = useState(false)
  const [error, setError] = useState('')

  const activeDocument = documents.find(
    (doc) => doc.document_id === activeDocumentId
  )

  useEffect(() => {
    getHealth()
      .then(() => setBackendOnline(true))
      .catch(() => setBackendOnline(false))
  }, [])

  async function handleUpload(file?: File) {
    if (!file) return

    if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
      setError('Please select a PDF file.')
      return
    }

    setError('')
    setAnswer(null)
    setUploading(true)

    try {
      const uploaded = await uploadDocument(file)

      if (!uploaded.document_id) {
        throw new Error(
          'The upload response did not contain a document_id. Please check the backend upload response.'
        )
      }

      const newDocument: DocumentItem = {
        document_id: uploaded.document_id,
        name: uploaded.filename || file.name,
        status: 'indexing',
      }

      setDocuments((previous) => [newDocument, ...previous])
      setActiveDocumentId(uploaded.document_id)

      await indexDocument(uploaded.document_id)

      setDocuments((previous) =>
        previous.map((doc) =>
          doc.document_id === uploaded.document_id
            ? { ...doc, status: 'indexed' }
            : doc
        )
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed.')

      setDocuments((previous) =>
        previous.map((doc) =>
          doc.document_id === activeDocumentId
            ? { ...doc, status: 'error' }
            : doc
        )
      )
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  async function handleAsk() {
    if (!activeDocumentId || !question.trim() || asking) return

    setError('')
    setAnswer(null)
    setAsking(true)

    try {
      const result = await askQuestion(activeDocumentId, question.trim(), 5)
      setAnswer(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not get an answer.')
    } finally {
      setAsking(false)
    }
  }

  return (
    <div className="flex min-h-screen bg-[#f6f7fb] text-slate-800">
      <aside className="hidden w-72 shrink-0 flex-col border-r border-slate-200 bg-white p-5 md:flex">
        <div className="mb-9 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-600 text-white">
            <BookOpen size={21} />
          </div>
          <div>
            <h1 className="text-lg font-bold">Report Navigator</h1>
            <p className="text-xs text-slate-500">Intelligence from documents</p>
          </div>
        </div>

        <button
          onClick={() => fileInputRef.current?.click()}
          className="mb-7 flex items-center justify-center gap-2 rounded-xl bg-indigo-600 px-4 py-3 text-sm font-semibold text-white hover:bg-indigo-700"
        >
          <Plus size={18} /> Upload document
        </button>

        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          className="hidden"
          onChange={(event) => handleUpload(event.target.files?.[0])}
        />

        <div className="mb-3 flex items-center justify-between">
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Your library
          </p>
          <Search size={15} className="text-slate-400" />
        </div>

        <div className="space-y-2">
          {documents.length === 0 && (
            <p className="px-2 py-3 text-xs leading-5 text-slate-400">
              No documents yet. Upload a PDF to get started.
            </p>
          )}

          {documents.map((doc) => (
            <button
              key={doc.document_id}
              onClick={() => {
                setActiveDocumentId(doc.document_id)
                setAnswer(null)
                setError('')
              }}
              className={`w-full rounded-xl p-3 text-left ${
                activeDocumentId === doc.document_id
                  ? 'bg-indigo-50 ring-1 ring-indigo-100'
                  : 'hover:bg-slate-50'
              }`}
            >
              <div className="flex items-start gap-3">
                <FileText size={19} className="mt-0.5 shrink-0 text-indigo-500" />
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">{doc.name}</p>
                  <p className="mt-1 text-xs text-slate-400">
                    {doc.status === 'indexed' && 'Indexed'}
                    {doc.status === 'indexing' && 'Indexing…'}
                    {doc.status === 'uploaded' && 'Uploaded'}
                    {doc.status === 'error' && 'Processing failed'}
                  </p>
                </div>
              </div>
            </button>
          ))}
        </div>

        <div className="mt-auto space-y-1 border-t border-slate-100 pt-4">
          <button className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm text-slate-500 hover:bg-slate-50">
            <Settings size={17} /> Settings
          </button>
          <button className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm text-slate-500 hover:bg-slate-50">
            <HelpCircle size={17} /> Help & support
          </button>
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-[72px] items-center justify-between border-b border-slate-200 bg-white px-5 md:px-8">
          <div>
            <p className="text-xs text-slate-400">Workspace / Documents</p>
            <h2 className="max-w-[260px] truncate text-sm font-semibold">
              {activeDocument?.name || 'Your workspace'}
            </h2>
          </div>

          <div className="flex items-center gap-2">
            <span className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium ${
              backendOnline
                ? 'bg-emerald-50 text-emerald-700'
                : 'bg-red-50 text-red-700'
            }`}>
              <span className={`h-2 w-2 rounded-full ${
                backendOnline ? 'bg-emerald-500' : 'bg-red-500'
              }`} />
              {backendOnline ? 'Backend connected' : 'Backend offline'}
            </span>
          </div>
        </header>

        <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-5 py-10 md:px-10">
          <div className="mb-8">
            <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-indigo-100 bg-white px-3 py-1.5 text-xs font-medium text-indigo-600">
              <Sparkles size={14} /> AI-powered report intelligence
            </div>
            <h1 className="text-3xl font-bold tracking-tight text-slate-900 md:text-4xl">
              Understand your reports.
              <br />
              <span className="text-indigo-600">Find answers faster.</span>
            </h1>
            <p className="mt-3 max-w-xl text-sm leading-6 text-slate-500">
              Ask questions about statistical reports and receive evidence-backed answers with page-level citations.
            </p>
          </div>

          <section className="mb-7 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold">Ask your document</h3>
                <p className="mt-1 text-xs text-slate-400">
                  Answers are grounded in your uploaded report.
                </p>
              </div>
              <MessageSquare size={19} className="text-indigo-500" />
            </div>

            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  handleAsk()
                }
              }}
              placeholder="e.g. Which district has the highest literacy rate?"
              className="min-h-28 w-full resize-none rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100"
            />

            <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
              <span className="flex items-center gap-1.5 text-xs text-slate-400">
                <Paperclip size={14} /> Sources are cited automatically
              </span>
              <button
                onClick={handleAsk}
                disabled={!question.trim() || !activeDocumentId || activeDocument?.status !== 'indexed' || asking}
                className="flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {asking ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />}
                {asking ? 'Thinking…' : 'Ask question'}
              </button>
            </div>

            {!activeDocument && (
              <p className="mt-3 text-xs text-amber-700">
                Upload a PDF before asking a question.
              </p>
            )}
            {activeDocument?.status === 'indexing' && (
              <p className="mt-3 text-xs text-indigo-600">
                Your document is being indexed. Questions will be enabled when it finishes.
              </p>
            )}
          </section>

          {uploading && (
            <div className="mb-5 flex items-center gap-3 rounded-xl border border-indigo-100 bg-indigo-50 p-4 text-sm text-indigo-700">
              <Loader2 size={18} className="animate-spin" />
              Uploading and indexing your PDF…
            </div>
          )}

          {error && (
            <div className="mb-5 flex items-start gap-2 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
              <AlertCircle size={18} className="mt-0.5 shrink-0" />
              <span>{error}</span>
              <button className="ml-auto" onClick={() => setError('')}>
                <X size={16} />
              </button>
            </div>
          )}

          {answer && (
            <section className="mb-7 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="mb-4 flex items-center justify-between">
                <h3 className="text-sm font-semibold">Answer</h3>
                <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${
                  answer.status === 'not_found'
                    ? 'bg-amber-50 text-amber-700'
                    : 'bg-emerald-50 text-emerald-700'
                }`}>
                  {answer.status === 'not_found' ? 'Not found' : 'Response received'}
                </span>
              </div>

              <p className="whitespace-pre-wrap text-sm leading-7 text-slate-700">
                {answer.answer || 'The backend did not return an answer.'}
              </p>

              {!!answer.citations?.length && (
                <div className="mt-5 border-t border-slate-100 pt-4">
                  <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Page citations
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {answer.citations.map((page) => (
                      <span
                        key={page}
                        className="rounded-lg bg-indigo-50 px-3 py-1.5 text-xs font-medium text-indigo-700"
                      >
                        Page {page}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {!!answer.evidence?.length && (
                <div className="mt-5 border-t border-slate-100 pt-4">
                  <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Retrieved evidence
                  </p>
                  <div className="space-y-3">
                    {answer.evidence.map((item, index) => (
                      <div
                        key={item.chunk_id || index}
                        className="rounded-xl bg-slate-50 p-3"
                      >
                        <div className="mb-2 flex items-center gap-2">
                          <span className="text-xs font-semibold text-indigo-600">
                            Page {item.page ?? '—'}
                          </span>
                          <span className="text-xs text-slate-400">
                            {item.source_type || 'Evidence'}
                          </span>
                        </div>
                        <p className="whitespace-pre-wrap text-xs leading-6 text-slate-600">
                          {item.text || item.content || 'Evidence text was not included in the response.'}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>
          )}

          {!answer && (
            <div className="rounded-2xl border border-dashed border-slate-300 bg-white/70 px-6 py-10 text-center">
              <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-indigo-50 text-indigo-600">
                <FileUp size={22} />
              </div>
              <h3 className="text-sm font-semibold">Your report workspace</h3>
              <p className="mx-auto mt-2 max-w-sm text-xs leading-5 text-slate-400">
                Upload a PDF from the sidebar. Once indexing finishes, ask a question to see its answer and supporting evidence here.
              </p>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="mt-4 inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium hover:bg-slate-50"
              >
                <Upload size={15} /> Choose PDF
              </button>
            </div>
          )}
        </div>

        <footer className="border-t border-slate-200 bg-white px-5 py-3 text-center text-xs text-slate-400">
          Report Navigator · Evidence-backed document intelligence
        </footer>
      </main>
    </div>
  )
}

export default App