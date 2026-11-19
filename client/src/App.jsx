import { useEffect, useRef, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'
const HEALTH_INTERVAL_MS = 5000

const STATUS_LABELS = {
  checking: 'Checking server…',
  online: 'Server online',
  degraded: 'Server online, Ollama unreachable',
  offline: 'Server offline',
}

function useServerHealth() {
  const [status, setStatus] = useState('checking')

  useEffect(() => {
    let timer
    let controller

    async function check() {
      controller?.abort()
      const current = (controller = new AbortController())
      try {
        const res = await fetch(`${API_BASE}/health`, {
          signal: AbortSignal.any([current.signal, AbortSignal.timeout(3000)]),
          cache: 'no-store',
        })
        if (!res.ok) throw new Error()
        const data = await res.json()
        setStatus(data.ollama ? 'online' : 'degraded')
      } catch {
        // Ignore requests we cancelled ourselves (superseded or tab hidden).
        if (!current.signal.aborted) setStatus('offline')
      }
    }

    function start() {
      check()
      timer = setInterval(check, HEALTH_INTERVAL_MS)
    }

    function stop() {
      clearInterval(timer)
      controller?.abort()
    }

    // Skip polling while the tab is in the background.
    function onVisibility() {
      if (document.hidden) stop()
      else start()
    }

    if (!document.hidden) start()
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      stop()
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [])

  return status
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const abortRef = useRef(null)
  const listEndRef = useRef(null)
  const formRef = useRef(null)
  const health = useServerHealth()
  useEffect(() => {
    listEndRef.current?.scrollIntoView({ block: 'end' })
  }, [messages])

  const updateLast = (content) =>
    setMessages((prev) => [...prev.slice(0, -1), { ...prev.at(-1), content }])

  async function send(e) {
    e.preventDefault()
    const text = input.trim()
    if (!text || busy) return

    const history = [...messages, { role: 'user', content: text }]
    setMessages([...history, { role: 'assistant', content: '' }])
    setInput('')
    setError('')
    setBusy(true)
    const controller = new AbortController()
    abortRef.current = controller

    try {
      const res = await fetch(`${API_BASE}/${streaming ? 'chat-stream' : 'chat'}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: history }),
        signal: controller.signal,
      })
      if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)

      if (streaming) {
        const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
        let acc = ''
        for (;;) {
          const { value, done } = await reader.read()
          if (done) break
          acc += value
          updateLast(acc)
        }
      } else {
        const data = await res.json()
        updateLast(data.message.content)
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message)
        setMessages((prev) => (prev.at(-1)?.content ? prev : prev.slice(0, -1)))
      }
    } finally {
      setBusy(false)
      abortRef.current = null
    }
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      // Don't submit while an IME is composing (keyCode 229 covers Safari).
      if (e.nativeEvent.isComposing || e.keyCode === 229) return
      formRef.current.requestSubmit()
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>
          Ollama Chat
          <span
            className={`status-dot ${health}`}
            role="status"
            title={STATUS_LABELS[health]}
          >
            <span className="visually-hidden">{STATUS_LABELS[health]}</span>
          </span>
        </h1>
        <div className="controls">
          <label className="toggle">
            <input
              type="checkbox"
              checked={streaming}
              onChange={(e) => setStreaming(e.target.checked)}
              disabled={busy}
            />
            Stream
          </label>
          <button
            type="button"
            className="secondary"
            onClick={() => {
              setMessages([])
              setError('')
            }}
            disabled={busy || messages.length === 0}
          >
            New chat
          </button>
        </div>
      </header>

      <main className="messages" aria-live="polite">
        {messages.length === 0 && <p className="empty">Ask me anything to get started.</p>}
        {messages.map((m, i) => (
          <div key={i} className={`bubble ${m.role}`}>
            {m.content || (busy && i === messages.length - 1 ? <span className="typing">…</span> : '')}
          </div>
        ))}
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        <div ref={listEndRef} />
      </main>

      <form ref={formRef} className="composer" onSubmit={send}>
        <label htmlFor="chat-input" className="visually-hidden">
          Message
        </label>
        <textarea
          id="chat-input"
          rows={1}
          placeholder="Type a message… (Shift+Enter for newline)"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          autoFocus
        />
        {busy ? (
          <button type="button" onClick={() => abortRef.current?.abort()}>
            Stop
          </button>
        ) : (
          <button type="submit" disabled={!input.trim()}>
            Send
          </button>
        )}
      </form>
    </div>
  )
}
