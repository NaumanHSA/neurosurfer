import { useRef, useState, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { ChatMessage } from '../lib/types';
import {
    createChat,
    appendChatMessage,
    listChatMessages,
    streamCompletions,
    stopGenerationAPICall,
    fetchFollowups
} from '../lib/api';
import { loadSettings, type ChatSettings } from '../components/SettingsDialog';
import { extractThinking } from '../lib/thinkParser';
import { useModels } from './useModels';
import { getRandomFollowUps } from '../lib/constants';

type UseChatOpts = {
    enabled?: boolean
}

export function useChat(opts: UseChatOpts = {}) {
    const { enabled = true } = opts
    const [messages, setMessages] = useState<ChatMessage[]>([])
    const [followups, setFollowups] = useState<string[]>([])
    const [loading, setLoading] = useState(false)
    const [opId, setOpId] = useState<string | null>(null)
    const [threadId, setThreadId] = useState<string | null>(null)
    const assistantMessageRef = useRef<ChatMessage | null>(null)
    const userMessageRef = useRef<ChatMessage | null>(null)
    const controllerRef = useRef<AbortController | null>(null)
    const { model } = useModels()

    // Open an existing chat and hydrate history
    async function openChat(id: string) {
        if (!enabled) return
        setThreadId(id)
        const rows = await listChatMessages(id)
        const mapped: ChatMessage[] = rows.map(r => ({
            id: uuidv4(),                   // UI id
            role: r.role as any,
            content: r.content,
            createdAt: new Date(r.createdAt * 1000).toISOString(),
        }))
        setMessages(mapped)
        // if mapped is empty, get from random followups
        if (mapped.length === 0) {
            setFollowups(getRandomFollowUps(3))
        } else {
            try {
                if (model && mapped.length > 0) {
                    const history = mapped
                        .slice(-3)
                        .map(m => ({ role: m.role, content: m.content }))
                    const s = await fetchFollowups(model, history)
                    setFollowups(s)
                    assistantMessageRef.current = null
                }
            } catch (error) {
                console.error('Error fetching follow-ups:', error)
                // Fallback to random follow-ups if there's an error
                setFollowups(getRandomFollowUps(3))
            }
        }
    }

    // start a new chat and return the server thread
    const newChat = useCallback(async () => {
        if (!enabled) return
        // clear current UI and create a server thread
        setMessages([])
        assistantMessageRef.current = null
        userMessageRef.current = null
        controllerRef.current?.abort()
        setOpId(null)
        setLoading(false)

        const th = await createChat('New Chat')
        setThreadId(th.id)
        setFollowups(getRandomFollowUps(3))
        return th  // <-- return created thread (id, title, createdAt)
    }, [enabled])

    function isFreshThread() {
        // We consider "fresh" = we already have a thread id, and no messages were sent yet
        return !!threadId && messages.length === 0
    }

    // Add a reset function
    const reset = useCallback(() => {
        setMessages([])
        setLoading(false)
        setFollowups([])
        setThreadId(null)
    }, [])

    // hooks/useChat.ts  (only the changed bits inside send())
    // Load settings from localStorage
    const getChatSettings = (overrides: Partial<ChatSettings> = {}) => {
        const settings = loadSettings();
        return { ...settings, ...overrides };
    };

    async function send(text: string, file?: File) {
        if (!enabled) return;
        setFollowups([]);
        if (!model) throw new Error('Select a model first');

        // Get settings from localStorage
        const settings = getChatSettings();

        // Ensure we have a thread
        let tid = threadId
        if (!tid) {
            const th = await createChat('New Chat')
            tid = th.id
            setThreadId(th.id)
        }

        // UI + persist user message (unchanged)
        const userMsg: ChatMessage = { id: uuidv4(), role: 'user', content: text, createdAt: new Date().toISOString() }
        userMessageRef.current = userMsg
        setMessages(prev => [...prev, userMsg])
        try { await appendChatMessage(tid!, { role: 'user', content: text }) } catch { }

        // Get message history based on messageHistoryLimit
        const recentMessages = messages.slice(-settings.messageHistoryLimit);

        // Build request payload with settings
        const body: any = {
            model,
            stream: true,
            messages: [
                { role: 'system', content: settings.systemPrompt },
                ...recentMessages.map(m => ({ role: m.role, content: m.content })),
                { role: 'user', content: text }
            ],
            temperature: settings.temperature,
            top_p: settings.topP,
        };

        // Only include max_tokens if it's set and valid
        if (settings.maxTokens && settings.maxTokens > 0) {
            body.max_tokens = settings.maxTokens;
        }

        // placeholder assistant (UI)
        const assistantMsg: ChatMessage = {
            id: uuidv4(),
            role: 'assistant',
            content: '',
            thinking: '',
            createdAt: new Date().toISOString(),
        }
        assistantMessageRef.current = assistantMsg
        setMessages(prev => [...prev, assistantMsg])

        setLoading(true)
        const controller = new AbortController()
        controllerRef.current = controller
        const op_id = uuidv4()
        setOpId(op_id)
        body.op_id = op_id

        let accum = ''
        try {
            // 🟢 NEW: pass file + thread id; the helper switches to multipart automatically if file is present
            for await (const chunk of streamCompletions(body, controller, { file, threadId: tid! })) {
                if (chunk.error) throw new Error(chunk.error)
                if (chunk.delta) {
                    accum += chunk.delta
                    const { thinking, visible } = extractThinking(accum)
                    const updated = { ...assistantMsg, content: visible, thinking }
                    assistantMessageRef.current = updated
                    setMessages(prev => prev.map(m => (m.id === assistantMsg.id ? updated : m)))
                }
                if (chunk.finish_reason) break
            }
        } finally {
            setLoading(false)
            setOpId(null)
            const finalAssistant = assistantMessageRef.current
            if (finalAssistant) {
                try { await appendChatMessage(tid!, { role: 'assistant', content: finalAssistant.content }) } catch { }
            }
            // followups (unchanged)
            try {
                if (model) {
                    const history = [...messages, userMsg, finalAssistant!].map(m => ({ role: m.role, content: m.content }))
                    const recentChats = history.slice(-3)
                    const s = await fetchFollowups(model, recentChats)
                    setFollowups(s)
                    assistantMessageRef.current = null
                }
            } catch { }
        }
    }

    async function stopGeneration() {
        try {
            controllerRef.current?.abort()
            await stopGenerationAPICall()
        } finally {
            setLoading(false)
            setOpId(null)
        }
    }

    return {
        model, messages, send, stopGeneration, loading, followups, openChat, newChat, threadId, isFreshThread: isFreshThread(), reset
    }
}
