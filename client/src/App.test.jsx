import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App.jsx'

const json = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })

function textStream(chunks) {
  const encoder = new TextEncoder()
  return new Response(
    new ReadableStream({
      start(controller) {
        chunks.forEach((c) => controller.enqueue(encoder.encode(c)))
        controller.close()
      },
    }),
  )
}

// Routes fetch calls by path; `handlers` maps path -> (init) => Response.
function mockFetch(handlers) {
  const fetchMock = vi.fn(async (url, init) => {
    const path = url.replace('/api', '')
    const handler = handlers[path]
    if (!handler) throw new Error(`Unexpected fetch ${url}`)
    return handler(init)
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

const callsTo = (fetchMock, path) => fetchMock.mock.calls.filter(([url]) => url === `/api${path}`)
const healthy = () => json({ status: 'ok', ollama: true })

async function sendMessage(user, text) {
  await user.type(screen.getByLabelText('Message'), text)
  await user.keyboard('{Enter}')
}

describe('health indicator', () => {
  it.each([
    ['online', () => json({ status: 'ok', ollama: true }), 'Server online'],
    ['degraded', () => json({ status: 'ok', ollama: false }), 'Server online, Ollama unreachable'],
    ['offline (HTTP error)', () => json({}, 500), 'Server offline'],
    [
      'offline (network error)',
      () => Promise.reject(new TypeError('Failed to fetch')),
      'Server offline',
    ],
  ])('shows %s', async (_, health, label) => {
    mockFetch({ '/health': health })
    render(<App />)
    const dot = await screen.findByRole('status')
    await waitFor(() => expect(dot).toHaveTextContent(label))
    expect(dot).toHaveAttribute('title', label)
  })

  it('polls /health every 5 seconds', async () => {
    vi.useFakeTimers()
    const fetchMock = mockFetch({ '/health': healthy })
    render(<App />)
    expect(callsTo(fetchMock, '/health')).toHaveLength(1)

    await act(() => vi.advanceTimersByTimeAsync(5000))
    expect(callsTo(fetchMock, '/health')).toHaveLength(2)

    await act(() => vi.advanceTimersByTimeAsync(10000))
    expect(callsTo(fetchMock, '/health')).toHaveLength(4)
  })

  it('pauses polling while the tab is hidden', async () => {
    vi.useFakeTimers()
    const fetchMock = mockFetch({ '/health': healthy })
    let hidden = false
    vi.spyOn(document, 'hidden', 'get').mockImplementation(() => hidden)
    render(<App />)

    hidden = true
    act(() => document.dispatchEvent(new Event('visibilitychange')))
    await act(() => vi.advanceTimersByTimeAsync(20000))
    expect(callsTo(fetchMock, '/health')).toHaveLength(1)

    hidden = false
    act(() => document.dispatchEvent(new Event('visibilitychange')))
    expect(callsTo(fetchMock, '/health')).toHaveLength(2)
  })
})

describe('chat', () => {
  let user
  beforeEach(() => {
    user = userEvent.setup()
  })

  it('streams the assistant reply from /chat-stream by default', async () => {
    const fetchMock = mockFetch({
      '/health': healthy,
      '/chat-stream': () => textStream(['Hel', 'lo', ' there']),
    })
    render(<App />)
    await sendMessage(user, 'Hi')

    expect(await screen.findByText('Hello there')).toBeInTheDocument()
    expect(screen.getByText('Hi')).toHaveClass('bubble', 'user')
    const [, init] = callsTo(fetchMock, '/chat-stream')[0]
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body)).toEqual({ messages: [{ role: 'user', content: 'Hi' }] })
  })

  it('uses /chat with the same payload when streaming is off', async () => {
    const fetchMock = mockFetch({
      '/health': healthy,
      '/chat': () => json({ model: 'm', message: { role: 'assistant', content: 'Full reply' } }),
    })
    render(<App />)
    await user.click(screen.getByLabelText('Stream'))
    await sendMessage(user, 'Hi')

    expect(await screen.findByText('Full reply')).toHaveClass('bubble', 'assistant')
    expect(callsTo(fetchMock, '/chat-stream')).toHaveLength(0)
    expect(JSON.parse(callsTo(fetchMock, '/chat')[0][1].body)).toEqual({
      messages: [{ role: 'user', content: 'Hi' }],
    })
  })

  it('sends the full conversation history on follow-up messages', async () => {
    const replies = ['First answer', 'Second answer']
    const fetchMock = mockFetch({
      '/health': healthy,
      '/chat-stream': () => textStream([replies.shift()]),
    })
    render(<App />)
    await sendMessage(user, 'Q1')
    await screen.findByText('First answer')
    await sendMessage(user, 'Q2')
    await screen.findByText('Second answer')

    expect(JSON.parse(callsTo(fetchMock, '/chat-stream')[1][1].body).messages).toEqual([
      { role: 'user', content: 'Q1' },
      { role: 'assistant', content: 'First answer' },
      { role: 'user', content: 'Q2' },
    ])
  })

  it('inserts a newline on Shift+Enter instead of sending', async () => {
    const fetchMock = mockFetch({ '/health': healthy })
    render(<App />)
    const input = screen.getByLabelText('Message')
    await user.type(input, 'line1{Shift>}{Enter}{/Shift}line2')

    expect(input).toHaveValue('line1\nline2')
    expect(callsTo(fetchMock, '/chat-stream')).toHaveLength(0)
  })

  it('does not send whitespace-only messages', async () => {
    const fetchMock = mockFetch({ '/health': healthy })
    render(<App />)
    await sendMessage(user, '   ')

    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled()
    expect(callsTo(fetchMock, '/chat-stream')).toHaveLength(0)
  })

  it('shows an error and removes the empty assistant bubble on failure', async () => {
    mockFetch({ '/health': healthy, '/chat-stream': () => json({ detail: 'nope' }, 502) })
    const { container } = render(<App />)
    await sendMessage(user, 'Hi')

    expect(await screen.findByRole('alert')).toHaveTextContent('502')
    expect(container.querySelectorAll('.bubble.assistant')).toHaveLength(0)
  })

  it('can stop an in-flight request', async () => {
    mockFetch({
      '/health': healthy,
      '/chat-stream': (init) =>
        new Promise((_, reject) =>
          init.signal.addEventListener('abort', () =>
            reject(new DOMException('Aborted', 'AbortError')),
          ),
        ),
    })
    render(<App />)
    await sendMessage(user, 'Hi')
    await user.click(await screen.findByRole('button', { name: 'Stop' }))

    expect(await screen.findByRole('button', { name: 'Send' })).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('clears the conversation with New chat', async () => {
    mockFetch({ '/health': healthy, '/chat-stream': () => textStream(['Reply']) })
    render(<App />)
    await sendMessage(user, 'Hi')
    await screen.findByText('Reply')
    await user.click(screen.getByRole('button', { name: 'New chat' }))

    expect(screen.queryByText('Reply')).not.toBeInTheDocument()
    expect(screen.getByText('Ask me anything to get started.')).toBeInTheDocument()
  })
})
