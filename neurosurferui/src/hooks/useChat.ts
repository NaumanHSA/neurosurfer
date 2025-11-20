import { useRef, useState, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import type { ChatMessage, ChatMessageWire, ChatFileWire, ChatFile } from '../lib/types';
import {
    createChat,
    appendChatMessage,
    streamCompletions,
    stopGenerationAPICall,
    fetchFollowups
} from '../lib/api';
import { loadSettings, type ChatSettings } from '../components/SettingsDialog';
import { extractThinking } from '../lib/thinkParser';
import { useModels } from './useModels';
import { getRandomFollowUps } from '../lib/constants';

type UseChatOpts = {
    enabled?: boolean;
};

export function useChat(opts: UseChatOpts = {}) {
    const { enabled = true } = opts;

    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [followups, setFollowups] = useState<string[]>([]);
    const [loading, setLoading] = useState(false);
    const [opId, setOpId] = useState<string | null>(null);
    const [threadId, setThreadId] = useState<string | null>(null);

    const assistantMessageRef = useRef<ChatMessage | null>(null);
    const userMessageRef = useRef<ChatMessage | null>(null);
    const controllerRef = useRef<AbortController | null>(null);
    const { model } = useModels();

    // ---------------- helpers ----------------

    const getChatSettings = (overrides: Partial<ChatSettings> = {}) => {
        const settings = loadSettings();
        return { ...settings, ...overrides };
    };

    function mapWireFile(f: ChatFileWire): ChatFile {
        return {
            id: f.id,
            name: f.filename,
            size: f.size,
            type: f.mime,
            url: f.downloadUrl,
            previewUrl: f.mime?.startsWith('image/') ? f.downloadUrl : undefined,
        };
    }

    function mapWireMessage(m: ChatMessageWire): ChatMessage {
        return {
            id: String(m.id),
            role: m.role,
            content: m.content,
            createdAt: new Date(m.createdAt * 1000).toISOString(),
            files: (m.files ?? []).map(mapWireFile),
        };
    }

    // ---------------- open / new / reset ----------------
    async function openChat(id: string) {
        if (!enabled) return;
        setThreadId(id);

        const res = await fetch(`/api/chats/${id}/messages`);
        const data: ChatMessageWire[] = await res.json();
        const mapped = data.map(mapWireMessage);
        setMessages(mapped);

        if (data.length === 0) {
            setFollowups(getRandomFollowUps(3));
        } else {
            try {
                if (model && data.length > 0) {
                    const history = data
                        .slice(-3)
                        .map(m => ({ role: m.role, content: m.content }));
                    const s = await fetchFollowups(model, history);
                    setFollowups(s);
                    assistantMessageRef.current = null;
                }
            } catch (error) {
                console.error('Error fetching follow-ups:', error);
                setFollowups(getRandomFollowUps(3));
            }
        }
    }

    const newChat = useCallback(async () => {
        if (!enabled) return;
        setMessages([]);
        assistantMessageRef.current = null;
        userMessageRef.current = null;
        controllerRef.current?.abort();
        setOpId(null);
        setLoading(false);

        const th = await createChat('New Chat');
        setThreadId(th.id);
        setFollowups(getRandomFollowUps(3));
        return th;
    }, [enabled]);

    function isFreshThread() {
        return !!threadId && messages.length === 0;
    }

    const reset = useCallback(() => {
        setMessages([]);
        setLoading(false);
        setFollowups([]);
        setThreadId(null);
    }, []);

    // ---------------- send ----------------

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


    async function send(text: string, file?: File) {
        if (!text && !file) return;
        if (!enabled) return;
        if (!model) throw new Error('Select a model first');

        setFollowups([]);

        const settings = getChatSettings();
        const currentMessagesSnapshot = messages; // local snapshot for history

        // Ensure we have a thread
        let tid = threadId;
        if (!tid) {
            const th = await createChat('New Chat');
            tid = th.id;
            setThreadId(th.id);
        }

        // Prepare files payload for backend (message + files)
        let filesPayload: {
            name: string;
            mime?: string;
            size?: number;
            base64: string;
        }[] = []

        if (file) {
            const base64 = await fileToBase64(file)
            filesPayload.push({
                name: file.name,
                mime: file.type,
                size: file.size,
                base64,
            })
        }

        // 1) Persist user message (text + files) via chats API
        //    appendChatMessage should POST /chats/{tid}/messages and return ChatMessageWire
        const savedWire: ChatMessageWire = await appendChatMessage(tid!, {
            role: 'user',
            content: text,
            files: filesPayload,
        })

        const savedMsg: ChatMessage = mapWireMessage(savedWire);
        userMessageRef.current = savedMsg;
        setMessages(prev => [...prev, savedMsg]);

        // 2) Build LLM history INCLUDING the new message
        const baseHistory = [...currentMessagesSnapshot, savedMsg];
        const recentMessages = baseHistory.slice(-settings.messageHistoryLimit);

        const llmMessages = [
            { role: 'system' as const, content: settings.systemPrompt },
            ...recentMessages.map(m => ({ role: m.role, content: m.content })),
        ];

        // 3) Build completion body (IDs, flags + normal OpenAI-style stuff)
        const messageIdNum = savedWire.id; // ChatMessageWire.id is number
        const hasFiles = (savedWire.files?.length ?? 0) > 0;

        const completionBody: any = {
            model,
            stream: true,
            messages: llmMessages,
            temperature: settings.temperature,
            top_p: settings.topP,
            // RAG / server context
            thread_id: Number(tid),
            message_id: messageIdNum,
            has_files: hasFiles,
        };

        if (settings.maxTokens && settings.maxTokens > 0) {
            completionBody.max_tokens = settings.maxTokens;
        }

        // 4) Placeholder assistant in UI
        const assistantMsg: ChatMessage = {
            id: uuidv4(),
            role: 'assistant',
            content: '',
            thinking: '',
            createdAt: new Date().toISOString(),
        };
        assistantMessageRef.current = assistantMsg;
        setMessages(prev => [...prev, assistantMsg]);

        // 5) Stream completions
        setLoading(true);
        const controller = new AbortController();
        controllerRef.current = controller;
        const op_id = uuidv4();
        setOpId(op_id);
        completionBody.op_id = op_id;

        let accum = '';
        try {
            // NOTE: streamCompletions now only takes (body, controller)
            for await (const chunk of streamCompletions(completionBody, controller)) {
                if (chunk.error) throw new Error(chunk.error);
                if (chunk.delta) {
                    accum += chunk.delta;
                    const { thinking, visible } = extractThinking(accum);
                    const updated: ChatMessage = {
                        ...assistantMsg,
                        content: visible,
                        thinking,
                    };
                    assistantMessageRef.current = updated;
                    setMessages(prev =>
                        prev.map(m => (m.id === assistantMsg.id ? updated : m)),
                    );
                }
                if (chunk.finish_reason) break;
            }
        } finally {
            setLoading(false);
            setOpId(null);

            const finalAssistant = assistantMessageRef.current;
            if (finalAssistant) {
                // Persist assistant message (no files)
                try {
                    await appendChatMessage(tid!, {
                        role: 'assistant',
                        content: finalAssistant.content,
                        files: [],
                    });
                } catch {
                    // ignore
                }
            }

            // 6) Followups
            try {
                if (model && finalAssistant) {
                    const fullHistory = [
                        ...currentMessagesSnapshot,
                        savedMsg,
                        finalAssistant,
                    ].map(m => ({ role: m.role, content: m.content }));

                    const recentChats = fullHistory.slice(-3);
                    const s = await fetchFollowups(model, recentChats);
                    setFollowups(s);
                    assistantMessageRef.current = null;
                }
            } catch {
                // ignore, we can fall back to random later if needed
            }
        }
    }

    // ---------------- stop ----------------

    async function stopGeneration() {
        try {
            controllerRef.current?.abort();
            await stopGenerationAPICall();
        } finally {
            setLoading(false);
            setOpId(null);
        }
    }

    return {
        model, messages, send, stopGeneration, loading, followups, openChat, newChat, threadId, isFreshThread: isFreshThread(), reset,
    };
}
