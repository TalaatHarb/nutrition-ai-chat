import { useEffect, useRef, useState } from 'react'
import AssistantMarkdown from './AssistantMarkdown.jsx'

function apiUrl(path) {
  const base = window._env_?.API_URL || (import.meta.env.VITE_API_BASE ?? '/api')
  return `${base.replace(/\/+$/, '')}/${path}`
}
const HEALTH_INTERVAL_MS = 5000
const STREAM_ERROR_PREFIX = '[error] '

const STATUS_LABELS = {
  checking: 'Checking server…',
  online: 'Server online',
  degraded: 'Server online, Ollama unreachable',
  offline: 'Server offline',
}

const STARTERS = [
  ['Plan my meals', 'Help me plan a week of simple, balanced meals.'],
  ['Scale a recipe', 'How much of each ingredient do I need to scale a recipe from 2 to 5 servings?'],
  ['Set meal times', 'Help me build a meal schedule around my workday.'],
]

function useServerHealth() {
  const [status, setStatus] = useState('checking')

  useEffect(() => {
    let timer
    let controller

    async function check() {
      controller?.abort()
      const current = (controller = new AbortController())
      try {
        const res = await fetch(apiUrl('health'), {
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
  const [generatingIndex, setGeneratingIndex] = useState(null)
  const [error, setError] = useState('')
  const [copyFeedback, setCopyFeedback] = useState(null)
  const busy = generatingIndex !== null
  const abortRef = useRef(null)
  const copyTimerRef = useRef(null)
  const listEndRef = useRef(null)
  const formRef = useRef(null)
  const inputRef = useRef(null)
  const health = useServerHealth()
  useEffect(() => {
    if (generatingIndex !== null && generatingIndex === messages.length - 1) {
      listEndRef.current?.scrollIntoView({ block: 'end' })
    }
  }, [messages, generatingIndex])

  useEffect(() => () => clearTimeout(copyTimerRef.current), [])

  function send(e) {
    e.preventDefault()
    const text = input.trim()
    if (!text || busy) return

    const history = [...messages, { role: 'user', content: text }]
    setMessages([...history, { role: 'assistant', content: '' }])
    setInput('')
    generate(history, history.length)
  }

  function regenerate(index) {
    if (busy) return
    const original = messages[index]
    setMessages((prev) => prev.map((m, i) => i === index ? { role: 'assistant', content: '' } : m))
    generate(messages.slice(0, index), index, original)
  }

  async function generate(history, index, original) {
    const sentHistory = history
      .map((message, messageIndex) => ({ message, messageIndex }))
      .filter(({ message }) => !message.filtered && !message.blockedResponse)
    const updateReply = (content) =>
      setMessages((prev) => prev.map((m, i) => i === index ? { ...m, content } : m))

    setError('')
    setGeneratingIndex(index)
    const controller = new AbortController()
    abortRef.current = controller

    try {
      const res = await fetch(apiUrl(streaming ? 'chat-stream' : 'chat'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: sentHistory.map(({ message: { role, content } }) => ({ role, content })),
        }),
        signal: controller.signal,
      })
      const filtered = res.headers.get('X-Filtered-Message-Indices')
      if (filtered) {
        const blocked = new Set(filtered.split(',').map((position) => sentHistory[Number(position)]?.messageIndex))
        setMessages((prev) => prev.map((m, i) => {
          if (blocked.has(i)) return { ...m, filtered: true }
          if (i === index) return { ...m, blockedResponse: true }
          return m
        }))
      }
      if (!res.ok) {
        const detail = await res.text()
        throw new Error(`${res.status} ${detail}`)
      }

      let reply
      if (streaming) {
        const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
        let acc = ''
        try {
          for (;;) {
            const { value, done } = await reader.read()
            if (done) break
            acc += value
            const errorIndex = acc.indexOf(STREAM_ERROR_PREFIX)
            updateReply(errorIndex === -1 ? acc : acc.slice(0, errorIndex))
          }
          const errorIndex = acc.indexOf(STREAM_ERROR_PREFIX)
          if (errorIndex !== -1) throw new Error(acc.slice(errorIndex + STREAM_ERROR_PREFIX.length))
          reply = acc
        } finally {
          reader.releaseLock()
        }
      } else {
        const data = await res.json()
        reply = data.message.content
      }
      if (!reply.trim()) throw new Error('The assistant did not return a reply.')
      updateReply(reply)
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message)
      }
      setMessages((prev) => {
        if (original) return prev.map((m, i) => i === index ? original : m)
        return prev[index].content.trim() ? prev : prev.filter((_, i) => i !== index)
      })
    } finally {
      setGeneratingIndex(null)
      abortRef.current = null
    }
  }

  async function copyMessage(index, content) {
    let text
    try {
      await navigator.clipboard.writeText(content)
      text = 'Copied'
    } catch {
      text = 'Could not copy'
    }
    clearTimeout(copyTimerRef.current)
    setCopyFeedback({ index, text })
    copyTimerRef.current = setTimeout(() => setCopyFeedback(null), 2500)
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
          <img className="brand-icon" src="/favicon.svg" alt="" />
          Nutrition Assistant
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
              setInput('')
              setCopyFeedback(null)
              inputRef.current.focus()
            }}
            disabled={busy || messages.length === 0}
          >
            New chat
          </button>
        </div>
      </header>

      <main className="messages">
        {messages.length === 0 && (
          <section className="welcome">
            <img className="welcome-icon" src="/favicon.svg" alt="" />
            <h2>I am your nutrition assistant. How can I help you?</h2>
            <p>Let&apos;s plan balanced meals that fit your tastes, budget, and routine.</p>
            <div className="starters">
              {STARTERS.map(([label, prompt]) => (
                <button
                  type="button"
                  className="secondary starter"
                  key={label}
                  onClick={() => {
                    setInput(prompt)
                    inputRef.current.focus()
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
          </section>
        )}
        {messages.length > 0 && (
          <div className="transcript" role="log" aria-label="Conversation" aria-busy={busy}>
            {messages.map((m, i) => (
              <div key={i} className={`bubble ${m.role}`}>
                <span className="visually-hidden">{m.role === 'assistant' ? 'Nutrition assistant: ' : 'You: '}</span>
                {m.filtered && <p className="filter-badge">Possible prompt injection — not sent to the model</p>}
                {m.content
                  ? m.role === 'assistant' && !m.filtered
                    ? <AssistantMarkdown content={m.content} />
                    : <div className="message-text">{m.content}</div>
                  : i === generatingIndex && <span className="typing">…</span>}
                {(m.content || i === generatingIndex) && (
                  <div className="message-actions">
                    <button
                      type="button"
                      disabled={i === generatingIndex}
                      aria-label={`Copy message ${i + 1}`}
                      onClick={() => copyMessage(i, m.content)}
                    >
                      Copy
                    </button>
                    {m.role === 'assistant' && !m.blockedResponse && (
                      <button
                        type="button"
                        aria-disabled={busy}
                        aria-label={`Regenerate reply ${i + 1}`}
                        onClick={() => regenerate(i)}
                      >
                        Regenerate
                      </button>
                    )}
                    <span className="copy-feedback" aria-live="polite">
                      {copyFeedback?.index === i ? copyFeedback.text : ''}
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        <div ref={listEndRef} />
      </main>

      <footer className="chat-footer">
        <form ref={formRef} className="composer" onSubmit={send}>
          <label htmlFor="chat-input" className="visually-hidden">
            Message
          </label>
          <textarea
            id="chat-input"
            ref={inputRef}
            rows={1}
            placeholder="Ask about meals, portions, or timing…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
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
        <p>General nutrition guidance, not medical advice.</p>
      </footer>
    </div>
  )
}
