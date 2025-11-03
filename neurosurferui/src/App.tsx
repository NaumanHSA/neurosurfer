import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import TopBar from './components/TopBar'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import AuthModal from './components/AuthModal'
import SettingsDialog from './components/SettingsDialog'
import { Spinner } from './components/ui/spinner'
import type { ConversationSummary } from './lib/types'

import {
    health, me, logout, listChats, deleteChat as apiDeleteChat,
    createChat, fetchTitle,
    updateChatTitle
} from './lib/api'

import { exportChatAsPDF } from './lib/exportPdf'
import { useChat } from './hooks/useChat'
import type { Theme } from './components/SettingsDialog'

// Utility to create a temp id for optimistic rows
const tempId = () => `temp-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

// Force update hook
function useForceUpdate() {
  const [, setValue] = useState(0)
  return () => setValue(value => value + 1)
}

export default function App() {
    // Backend/auth bootstrap
    const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking')
    const [authed, setAuthed] = useState<boolean | null>(null)
    const [authOpen, setAuthOpen] = useState(false)
    const [settingsOpen, setSettingsOpen] = useState(false)
    const [user, setUser] = useState<{ id: string; name: string; email?: string | null } | null>(null)
    const forceUpdate = useForceUpdate()

    // Conversations list
    const [convos, setConvos] = useState<ConversationSummary[]>([])

    // Track a “fresh” chat being created/opened to block duplicates
    const [creatingNew, setCreatingNew] = useState(false)
    const [freshThreadId, setFreshThreadId] = useState<string | null>(null) // thread with zero user msgs

    const handleThemeChange = useCallback((theme: Theme) => {
        // The theme is already being applied by the SettingsDialog's useTheme hook
        // We just need to trigger a re-render to apply the new theme
        forceUpdate()
    }, [forceUpdate])

    // Auto-new-on-refresh guard
    const autoNewDoneRef = useRef(false)
    const isMountedRef = useRef(false)
    const convosLoadedRef = useRef(false)
    const currentUserIdRef = useRef<string | null>(null)


    // For title auto-suggest after N messages (SIMPLE policy)
    const TITLE_AFTER_N = 3

    // Whether we've already fetched a title for this thread (simple policy)
    const [titled, setTitled] = useState<Record<string, boolean>>({})

    // Chat hook (focuses on the current thread+messages UI)
    const chatEnabled = backendStatus === 'online' && authed === true
    const { model, messages, send, stopGeneration, loading, followups, openChat, threadId, reset: resetChat } = useChat({ enabled: chatEnabled })

    // App-level selected/open thread (mirrors the hook)
    const [activeThreadId, setActiveThreadId] = useState<string | null>(null)
    const [chatResetKey, setChatResetKey] = useState(0)   // force ChatWindow remount when clearing

    // Keep App in sync with the hook's current thread
    useEffect(() => {
        setActiveThreadId(threadId ?? null)
    }, [threadId])

    // Mark when component is mounted
    useEffect(() => {
        isMountedRef.current = true
    }, [])

    // ---- Bootstrap health & auth ----
    useEffect(() => {
        (async () => {
            try { setBackendStatus((await health()) ? 'online' : 'offline') }
            catch { setBackendStatus('offline') }
        })()
    }, [])

    useEffect(() => {
        if (backendStatus !== 'online') return
        (async () => {
            try {
                const u = await me()
                setUser(u)
                setAuthed(true)
            } catch {
                setAuthed(false)
                setAuthOpen(true)
            }
        })()
    }, [backendStatus])

    // ---- Load convos (prefer server titles over local "New Chat") ----
    const loadConvos = useCallback(async () => {
        try {
            convosLoadedRef.current = false
            const rows = await listChats()
            setConvos(prevExisting => {
                const prevMap = new Map(prevExisting.map(c => [c.id, c]))
                return rows.map(r => {
                    const id = String(r.id)
                    const prev = prevMap.get(id)
                    const serverTitle = r.title || 'New Chat'

                    // Only preserve local title if it's not "New Chat" and server also has "New Chat"
                    const shouldPreserveLocal = prev?.title &&
                        prev.title.trim() !== '' &&
                        prev.title !== 'New Chat' &&
                        serverTitle === 'New Chat'
                    return {
                        id,
                        title: shouldPreserveLocal ? prev.title : serverTitle,
                        updatedAt: new Date(r.updatedAt * 1000).toISOString(),
                        messagesCount: r.messagesCount || 0,
                    }
                })
            })
        } finally {
            convosLoadedRef.current = true
        }
    }, [])

    useEffect(() => { if (authed) loadConvos() }, [authed, loadConvos])

    // Helper: find an existing "fresh" chat (no messages yet) if any
    const findExistingFresh = useCallback(() => {
        // Prefer a chat with messagesCount === 0. As a secondary heuristic, title "New Chat".
        return convos.find(c => (c.messagesCount ?? 0) === 0) || convos.find(c => (c.title || '').trim().toLowerCase() === 'new chat') || null
    }, [convos])

    // ---- Auto-create/select on refresh (do not duplicate if a fresh chat already exists) ----
    useEffect(() => {
        if (!chatEnabled) return
        if (autoNewDoneRef.current) return
        if (!convosLoadedRef.current) return // Wait for convos to finish loading
        if (!isMountedRef.current) return // Only run after component is mounted

        // If a thread already selected (e.g., hot rehydration), we're done
        if (activeThreadId) {
            autoNewDoneRef.current = true
            return
        }

        // Create one
        if (convosLoadedRef.current) {
            autoNewDoneRef.current = true
            handleNewChat()
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [chatEnabled, convosLoadedRef.current])

    const handleNewChat = useCallback(async () => {
        if (freshThreadId) { openChat(freshThreadId); return }
        if (creatingNew) return // in-flight guard

        // Also: if a fresh chat already exists in convos, just select it (extra safety)
        const existingFresh = findExistingFresh()
        if (existingFresh) { openChat(existingFresh.id); return }

        // Check if the first chat is already a "New Chat" - don't create another one
        if (convos.length > 0) {
            const firstChat = convos[0]
            if (firstChat && (firstChat.title === 'New Chat' || !firstChat.title || firstChat.title.trim() === '')) {
                openChat(firstChat.id)
                return
            }
        }
        setCreatingNew(true)
        // 1) optimistic row
        const tid = tempId()
        const optimistic: ConversationSummary = {
            id: tid,
            title: 'New Chat',
            updatedAt: new Date().toISOString(),
            messagesCount: 0,
        }
        setConvos(prev => [optimistic, ...prev])
        setFreshThreadId(tid)
        setActiveThreadId(tid) // select it immediately
        try {
            // 2) actually create on backend
            const created = await createChat('') // if your server wants body, pass it
            // 3) replace optimistic row with the real one
            const realId = String(created.id)
            setConvos(prev => {
                const idx = prev.findIndex(c => c.id === tid)
                if (idx === -1) {
                    // optimistic row missing (unlikely), just prepend
                    return [{
                        id: realId,
                        title: created.title || 'New Chat',
                        updatedAt: new Date(created.updatedAt * 1000).toISOString(),
                        messagesCount: created.messagesCount || 0,
                    }, ...prev]
                }
                const next = [...prev]
                next[idx] = {
                    id: realId,
                    title: created.title || 'New Chat',
                    updatedAt: new Date(created.updatedAt * 1000).toISOString(),
                    messagesCount: created.messagesCount || 0,
                }
                return next
            })

            setFreshThreadId(realId)
            setActiveThreadId(realId)
            openChat(realId)
        } catch (e) {
            // roll back optimistic if failed
            setConvos(prev => prev.filter(c => c.id !== tid))
            setFreshThreadId(null)
            setActiveThreadId(null)
            console.error('Failed to create chat', e)
        } finally {
            setCreatingNew(false)
        }
    }, [creatingNew, freshThreadId, openChat, findExistingFresh, convos])

    // ---- Seed counts & simple-title policy when selecting/opening a thread ----
    useEffect(() => {
        if (!activeThreadId) return
        const row = convos.find(c => c.id === activeThreadId)
        if (!row) return

        // SIMPLE rule: if initial count already >= N, assume title was handled -> mark as titled
        setTitled(prev => {
            if (prev[activeThreadId] != null) return prev
            const c = row.messagesCount ?? 0
            return { ...prev, [activeThreadId]: c >= TITLE_AFTER_N }
        })
    }, [activeThreadId, convos])

    // ---- When total reaches N for the first time on SHORT threads, fetch a title via /v1/chat/completions ----
    useEffect(() => {
        (async () => {
            if (!activeThreadId) return
            if (loading) return
            if (!messages.length) return
            const last = messages[messages.length - 1]
            if (last.role !== 'assistant') return
            // Once assistant first replies, no longer "fresh"
            if (freshThreadId === activeThreadId) setFreshThreadId(null)

            const row = convos.find(c => c.id === activeThreadId)
            const total = row?.messagesCount ?? 0
            const alreadyTitled = !!titled[activeThreadId]

            // Only fire once for threads that started below N and just crossed threshold
            if (alreadyTitled) return
            if (total < TITLE_AFTER_N) return

            // Build compact message list for the LLM (like followups)
            const llmMessages = messages
                .filter(m => m.role === 'user' || m.role === 'assistant')
                .slice(-10)
                .map(m => ({ role: m.role, content: m.content }))
            try {
                const title = await fetchTitle(model!, llmMessages)
                if (title) {
                    // Update the sidebar immediately with new timestamp to trigger re-render
                    setConvos(prev => prev.map(c =>
                        c.id === activeThreadId ? { ...c, title, updatedAt: new Date().toISOString() } : c
                    ))
                    // Update the title on the server
                    await updateChatTitle(activeThreadId, title)
                }
            } catch {
                // ignore errors
            } finally {
                // Mark as done so we don't re-fetch for this simple policy
                setTitled(prev => ({ ...prev, [activeThreadId]: true }))
            }
        })()
    }, [activeThreadId, freshThreadId, convos, titled, messages, model, loading])

    // ---- Delete & Share from sidebar ----
    const handleDeleteChat = useCallback(async (id: string) => {
        await apiDeleteChat(id)

        // Compute the next chat to select BEFORE mutating state, using current order
        let nextToSelect: string | null = null
        if (activeThreadId === id) {
            const currentOrder = convos.map(c => c.id)
            const idx = currentOrder.findIndex(cid => cid === id)
            if (idx !== -1) {
                // Try the next item (becomes the new top if we deleted the first)
                nextToSelect = currentOrder[idx + 1] || currentOrder[idx - 1] || null
            }
        }
        // remove from list
        setConvos(prev => prev.filter(c => c.id !== id))

        if (activeThreadId === id) {
            setFreshThreadId(null)
            try { await stopGeneration() } catch { }
            setChatResetKey(k => k + 1)

            if (nextToSelect) {
                setActiveThreadId(nextToSelect)
                openChat(nextToSelect)
            } else {
                // No chats left — optionally create a new one automatically
                setActiveThreadId(null)
                await handleNewChat() // enable if you want auto-create when list becomes empty
            }
        }
    }, [activeThreadId, convos, stopGeneration, openChat])

    const handleShareChat = useCallback(async (id: string) => {
        const chat = convos.find(c => c.id === id)
        const title = chat?.title || 'Chat Export'
        await exportChatAsPDF(id, title)
    }, [convos])

    const handleDeletedAccount = useCallback(() => {
        setSettingsOpen(false)
        setUser(null); setAuthed(false); setAuthOpen(true)
    }, [])

    // Add this effect to detect user changes and clear state
    useEffect(() => {
        if (user?.id && currentUserIdRef.current && currentUserIdRef.current !== user.id) {
        // User changed - clear all state
        setConvos([])
        setActiveThreadId(null)
        setFreshThreadId(null)
        setChatResetKey(k => k + 1)
        autoNewDoneRef.current = false
        convosLoadedRef.current = false
        resetChat() // Reset chat hook state
        }
        
        if (user?.id) {
        currentUserIdRef.current = user.id
        }
    }, [user?.id])
    
    // Update handleLogout to clear all state
    const handleLogout = useCallback(async () => {
        try {
        await stopGeneration()
        } catch (e) {
        console.error('Error stopping chat:', e)
        }
        
        // Clear all state
        setConvos([])
        setActiveThreadId(null)
        setFreshThreadId(null)
        setChatResetKey(k => k + 1)
        resetChat() // Reset chat hook
        
        // Reset refs
        autoNewDoneRef.current = false
        convosLoadedRef.current = false
        currentUserIdRef.current = null
        
        // Set auth state
        setUser(null)
        setAuthed(false)
        setAuthOpen(true)
        
        // Call logout API
        await logout()
    }, [stopGeneration, resetChat])

    // ---- Disable “New chat” button when we must block clicks ----
    const newDisabled = useMemo(() => {
        return !chatEnabled || creatingNew || !!freshThreadId
    }, [chatEnabled, creatingNew, freshThreadId])

    // ---- Chats loading: only when chatEnabled; prefer server titles over local "New Chat" ----
    const refreshChats = React.useCallback(async () => {
        if (!chatEnabled) return
        try {
            const rows = await listChats()
            setConvos(prevExisting => {
                const prevMap = new Map(prevExisting.map(c => [c.id, c]))
                return rows.map(r => {
                    const id = String(r.id)
                    const prev = prevMap.get(id)
                    const serverTitle = r.title || 'New Chat'
                    // Only preserve local title if it's not "New Chat" and server also has "New Chat"
                    const shouldPreserveLocal = prev?.title &&
                        prev.title.trim() !== '' &&
                        prev.title !== 'New Chat' &&
                        serverTitle === 'New Chat'
                    return {
                        id,
                        title: shouldPreserveLocal ? prev.title : serverTitle,
                        updatedAt: new Date(r.updatedAt * 1000).toISOString(),
                        messagesCount: r.messagesCount || 0,
                    }
                })
            })
        } catch (err) {
            console.error('Failed to refresh chats:', err)
        }
    }, [chatEnabled])

    const currentChatTitle = useMemo(() => {
        if (!activeThreadId) return 'Chat Export'
        const chat = convos.find(c => c.id === activeThreadId)
        return chat?.title || 'Chat Export'
    }, [activeThreadId, convos])

    

    // ---- Render ----
    if (backendStatus === 'checking') {
        return <div className="h-screen flex items-center justify-center"><Spinner /></div>
    }
    if (backendStatus === 'offline') {
        return (
            <div className="h-screen flex items-center justify-center text-center">
                <div>
                    <div className="text-2xl font-semibold mb-2">Backend offline</div>
                    <div className="text-muted-foreground">Please start the server and refresh.</div>
                </div>
            </div>
        )
    }

    return (
        <div className="h-screen w-screen flex">
            {authOpen ? (
                <AuthModal
                    onClose={() => setAuthOpen(false)}
                    onAuthed={async () => {
                        setAuthed(true)
                        setUser(await me())
                        setAuthOpen(false)
                        loadConvos()
                    }}
                />
            ) :
                <>
                    <Sidebar
                        convos={convos}
                        onNew={handleNewChat}
                        onOpen={openChat}
                        onDelete={handleDeleteChat}
                        onShare={handleShareChat}
                        newDisabled={newDisabled}
                        user={user}
                        onLogout={handleLogout}
                        onOpenSettings={() => setSettingsOpen(true)}
                        selectedId={activeThreadId || undefined}
                    />
                    <div className="flex-1 min-w-0 flex flex-col">
                        <TopBar />
                        <div className="flex-1 overflow-hidden">
                            <ChatWindow
                                key={chatResetKey}
                                messages={activeThreadId ? messages : []}
                                followups={activeThreadId ? followups : []}
                                onPickFollowup={(q) => activeThreadId ? send(q).then(refreshChats).catch(() => { }) : Promise.resolve()}
                                onSend={(t, f) => activeThreadId ? send(t, f).then(refreshChats).catch(() => { }) : Promise.resolve()}
                                onStop={stopGeneration}
                                loading={!!activeThreadId && loading}
                                chatId={activeThreadId}
                                chatTitle={currentChatTitle}
                                user={user}
                            />
                        </div>
                    </div>
                    <SettingsDialog
                        open={settingsOpen}
                        onClose={() => setSettingsOpen(false)}
                        onDeleted={handleDeletedAccount}
                        onThemeChange={handleThemeChange}
                        user={user}
                    />
                </>
            }
        </div>
    )
}
