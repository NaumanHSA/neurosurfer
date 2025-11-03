// api.ts
import { TITLE_SYSTEM_PROMPT } from './constants'
import { ChatMessage, ModelInfo } from './types'
import type { ChatCompletionChunk } from './types'

// You can set VITE_BACKEND_API_URL to "http://localhost:8081"
// const API_BASE = import.meta.env.VITE_BACKEND_API_URL ?? ''
const API_BASE = import.meta.env.VITE_BACKEND_URL || `${window.location.protocol}//${window.location.hostname}:8081`;
  
// -------------------------------------------------------------
// Auth strategy
// - Primary: HttpOnly cookie (set by /auth/login or /auth/register)
// - Optional: API key / Bearer for service calls (opt-in)
// -------------------------------------------------------------
let _apiKey: string | null = null          // optional service API key
export function setApiKey(key?: string) {
  _apiKey = key || null
}

function baseHeaders(hasBody: boolean, extra?: HeadersInit) {
  // Do NOT set Content-Type when sending FormData
  const h: Record<string, string> = hasBody ? { 'Content-Type': 'application/json' } : {}
  if (_apiKey) h['Authorization'] = `Bearer ${_apiKey}`
  return { ...h, ...(extra || {}) }
}

// lib/api.ts (or wherever req is)
type ReqOpts = { timeoutMs?: number }

export async function req<T = any>(
  path: string,
  init: RequestInit = {},
  opts: ReqOpts = {}
): Promise<T> {
  const { timeoutMs = 10000 } = opts
  const hasBody = !!init.body && !(init.body instanceof FormData)

  const ac = new AbortController()
  const timer = setTimeout(() => ac.abort(), timeoutMs)

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...init,
      credentials: 'include',
      headers: {
        ...(init.headers || {}),
        ...baseHeaders(hasBody),
      },
      signal: ac.signal,
    })

    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(text || res.statusText)
    }

    // 204: no body
    if (res.status === 204) return undefined as any

    const ct = (res.headers.get('content-type') || '').toLowerCase()
    const text = await res.text().catch(() => '')

    if (ct.includes('application/json')) {
      return text ? JSON.parse(text) : (undefined as any)
    }
    return text as any
  } finally {
    clearTimeout(timer)
  }
}


// -------------------------------------------------------------
// Auth endpoints (cookie-based)
// -------------------------------------------------------------
export type UserSummary = { id: string; name: string; email?: string | null }

export async function register(name: string, email: string, password: string): Promise<UserSummary> {
  // Backend expects full_name
  return req<UserSummary>(
    '/v1/auth/register',
    {
    method: 'POST',
    body: JSON.stringify({ full_name: name, email, password }),
  },
)
}

export async function login(email: string, password: string): Promise<{ token: string; user: UserSummary }> {
  // Backend returns { token, user } AND sets HttpOnly cookie.
  // You do NOT need to store token on the frontend (cookie is enough).
  return req('/v1/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  })
}

export async function logout(): Promise<void> {
  await req('/v1/auth/logout', { method: 'POST' })
  // If you were using API key override, clear it if desired:
  // _apiKey = null
}

export async function me(): Promise<UserSummary> {
  return req('/v1/auth/me')
}

export async function deleteAccount(password: string): Promise<void> {
  // Backend can map this to your preferred route.
  // Suggested FastAPI endpoint: POST /v1/auth/delete_account { password }
  await req('/v1/auth/delete_account', {
    method: 'POST',
    body: JSON.stringify({ password }),
  })
}

// lib/api.ts
export async function health(): Promise<boolean> {
  // Ask nicely for text but accept JSON too
  const res = await req<any>('/health', { headers: { Accept: 'text/plain,application/json' } }, { timeoutMs: 2500 })

  // If server returned JSON (rare), treat existence as healthy
  if (res && typeof res === 'object') return true

  // If server returned text, consider common “OK” shapes
  const s = String(res || '').trim().toLowerCase()
  return s === 'ok' || s === 'healthy' || s.includes('ok') || s.includes('alive') || s.includes('ready')
}

// -------------------------------------------------------------
// Models / Info / Files
// -------------------------------------------------------------
export async function fetchModels(): Promise<ModelInfo[]> {
  const data = await req<any>('/v1/models')
  return Array.isArray(data) ? data : (data.data ?? [])
}

// -------------------------------------------------------------
// Chat threads
// -------------------------------------------------------------
export type ChatThreadWire = {
  id: string; title: string; createdAt: number; updatedAt: number; messagesCount: number
}
export type ChatMessageWire = { id: number; role: string; content: string; createdAt: number }

export async function listChats(): Promise<ChatThreadWire[]> {
  return req('/v1/chats')
}
export async function createChat(title = 'New Chat'): Promise<ChatThreadWire> {
  return req('/v1/chats', { method: 'POST', body: JSON.stringify({ title }) })
}
export async function listChatMessages(chatId: string|number): Promise<ChatMessageWire[]> {
  return req(`/v1/chats/${chatId}/messages`)
}
export async function appendChatMessage(chatId: string|number, msg: { role: string; content: string }) {
  return req(`/v1/chats/${chatId}/messages`, { method: 'POST', body: JSON.stringify(msg) })
}
export async function deleteChat(chatId: string|number): Promise<void> {
  await req(`/v1/chats/${chatId}`, { method: 'DELETE' })
}
export async function updateChatTitle(chatId: string|number, title: string): Promise<void> {
  await req(`/v1/chats/${chatId}`, { method: 'PUT', body: JSON.stringify({ title }) })
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Sidebar helper: prefer /chats; fallback to /info if you have it
export async function fetchInfo(): Promise<{ conversations: { id: string; title: string; updatedAt: string }[] }> {
  try {
    const arr = await req<any[]>('/v1/chats')
    return {
      conversations: (arr || []).map((c: any) => ({
        id: c.id,
        title: c.title || 'Chat',
        updatedAt: new Date((c.createdAt || Math.floor(Date.now() / 1000)) * 1000).toISOString(),
      })),
    }
  } catch {}
  try {
    return await req('/v1/info')
  } catch {}
  return { conversations: [] }
}

export async function uploadFile(query: string, file: File): Promise<{ file_id: string, message?: string }> {
  const form = new FormData()
  form.append('query', query)
  form.append('file', file)
  return req('/v1/files', { method: 'POST', body: form }) // no Content-Type header for FormData
}

// -------------------------------------------------------------
// Streaming completions (SSE text/event-stream compatible)
// -------------------------------------------------------------
export type StreamOpts = {
  file?: File | null
  threadId?: string | number | null
}

// robust base64 using FileReader -> data URL, then strip the header
async function fileToBase64(file: File): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const r = new FileReader()
    r.onerror = () => reject(r.error)
    r.onload = () => {
      const res = String(r.result || '')
      const comma = res.indexOf(',')
      resolve(comma >= 0 ? res.slice(comma + 1) : res) // strip "data:...;base64,"
    }
    r.readAsDataURL(file)
  })
}

// -------------------------------------------------------------
// Streaming completions (SSE text/event-stream compatible)
// JSON-only; if opts.file is provided, we embed it as base64 in body.files[]
// -------------------------------------------------------------
export async function* streamCompletions(
  body: any,
  controller: AbortController,
  opts: StreamOpts = {}
): AsyncGenerator<ChatCompletionChunk> {
  const { file = null, threadId = null } = opts
  // put thread id into the JSON body (backend expects this now)
  if (threadId != null) body.thread_id = Number(threadId)

  // if a file is present, convert -> base64 and attach to body.files (JSON)
  if (file) {
    try {
      const b64 = await fileToBase64(file)
      const fileEntry = {
        name: file.name,
        content: b64,          // base64 payload
        type: file.type || undefined,
      }
      // preserve any existing files[] caller added; else create it
      if (Array.isArray(body.files)) body.files.push(fileEntry)
      else body.files = [fileEntry]
    } catch (e) {
      console.warn('Failed to base64-encode file, sending without files[]:', e)
    }
  }

  const res = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: 'POST',
    credentials: 'include',
    headers: {
      ...baseHeaders(true, {}), // IMPORTANT: JSON headers
    },
    body: JSON.stringify(body),
    signal: controller.signal,
  })
  if (!res.ok) throw new Error(await res.text())

  const reader = res.body!.getReader()
  const decoder = new TextDecoder('utf-8')
  let buf = ''

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })

    const lines = buf.split('\n')
    for (let i = 0; i < lines.length - 1; i++) {
      const line = lines[i].trim()
      if (!line.startsWith('data:')) continue
      const payload = line.slice(5).trim()
      if (payload === '[DONE]') {
        yield { finish_reason: 'stop' }
        continue
      }
      try {
        const json = JSON.parse(payload)
        if (Array.isArray(json?.choices)) {
          const ch0 = json.choices[0] || {}
          const d = ch0?.delta?.content || ''
          const fr = ch0?.finish_reason
          if (d)  yield { delta: d }
          if (fr) yield { finish_reason: fr }
        } else if (json?.delta || json?.finish_reason) {
          yield json
        } else if (typeof json === 'string') {
          yield { delta: json }
        }
      } catch {
        // ignore malformed line
      }
    }
    buf = lines[lines.length - 1]
  }
}

// -------------------------------------------------------------
// Stop (compat with both /stop and /stop/{op_id})
// -------------------------------------------------------------
export async function stopGenerationAPICall(): Promise<void> {
  try {
    await req('/v1/stop', { method: 'POST' })
    return
  } catch {}
}

// -------------------------------------------------------------
// Optional: Followups via non-stream completion
// -------------------------------------------------------------
export async function fetchFollowups(model: string, messages: { role: string; content: string }[]): Promise<string[]> {
  try {
    const data = await req<any>('/v1/chat/completions', {
      method: 'POST',
      body: JSON.stringify({
        model,
        stream: false,
        temperature: 0.7,
        // messages: [{ role: "system", content: FOLLOWUPS_SYSTEM_PROMPT }, ...messages],
        messages: messages,
        metadata: {"follow_up_questions": true}
      }),
    })
    const content = data?.choices?.[0]?.message?.content || ""
    try {
      const obj = JSON.parse(content)
      return obj?.suggestions || []
    } catch { return [] }
  } catch { return [] }
}

// -------------------------------------------------------------
// Optional: Title via non-stream completion
// -------------------------------------------------------------
// Normalize messages (only what the LLM needs)
function toLLMMessages(messages: { role: string; content: string }[]) {
  // Keep only "user" and "assistant" roles, and trim to last 10 for cost
  return messages
    .filter(m => m && typeof m.content === 'string' && (m.role === 'user' || m.role === 'assistant'))
    .slice(-10)
    .map(m => ({ role: m.role, content: m.content }))
}

// Robust JSON extractor (handles code fences or stray text)
function extractJSON<T = any>(raw: string): T | null {
  if (!raw) return null
  let text = raw.trim()
  // strip ```json ... ``` or ``` ... ```
  if (text.startsWith('```')) {
    const first = text.indexOf('{')
    const last = text.lastIndexOf('}')
    if (first !== -1 && last !== -1 && last > first) {
      text = text.slice(first, last + 1)
    }
  }
  try { return JSON.parse(text) as T } catch { return null }
}

// NEW: Title suggestion via /v1/chat/completions (no backend change)
export async function fetchTitle(
  model: string,
  messages: { role: string; content: string }[],
): Promise<string> {
  try {
    const data = await req<any>('/v1/chat/completions', {
      method: 'POST',
      body: JSON.stringify({
        model,
        stream: false,
        temperature: 0.7,
        messages: [{ role: 'system', content: TITLE_SYSTEM_PROMPT }, ...toLLMMessages(messages)],
      }),
    })
    const content = data?.choices?.[0]?.message?.content || ''
    const obj = extractJSON<{ title?: string }>(content)
    const title = (obj?.title || '').trim()
    // sanity: collapse whitespace, remove trailing punctuation
    return title.replace(/\s+/g, ' ').replace(/[.?!:\-—\s]+$/, '')
  } catch {
    return ''
  }
}