import React, { useEffect, useState, useRef } from 'react'
import { ChatMessage } from '../lib/types'
import MessageBubble from './MessageBubble'
import ChatInput from './ChatInput'
import Followups from './Followups'
import ChatActions from './ChatActions'
import { ArrowDown } from 'lucide-react'


export default function ChatWindow({
  messages,
  followups,
  onPickFollowup,
  onSend,
  onStop,
  loading,
  chatId,
  chatTitle,
  user,
}: {
  messages: ChatMessage[],
  followups: string[],
  onPickFollowup: (q: string) => void,
  onSend: (text: string, file?: File) => void,
  onStop: () => void,
  loading: boolean,
  chatId?: string | null,
  chatTitle?: string,
  user?: { id: string; name: string; email?: string | null } | null,
}) {
  const hasMessages = messages.length > 0
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const lastScrollTop = useRef(0);

  // Detect scroll direction and handle attach/detach
  useEffect(() => {
    const container = scrollRef.current
    if (!container) return

    const handleScroll = () => {
      const currentScrollTop = container.scrollTop
      const scrollHeight = container.scrollHeight
      const clientHeight = container.clientHeight

      // Detect scroll direction
      const scrollingUp = currentScrollTop < lastScrollTop.current
      const scrollingDown = currentScrollTop > lastScrollTop.current

      // If scrolling up, immediately detach
      if (scrollingUp) {
        setAutoScroll(false)
      }
      // If scrolling down, check if near bottom and attach
      // else if (scrollingDown) {
      //   const isNearBottom = scrollHeight - currentScrollTop - clientHeight < 50
      //   if (isNearBottom) {
      //     setAutoScroll(true)
      //   }
      // }

      lastScrollTop.current = currentScrollTop
    }

    container.addEventListener("scroll", handleScroll)
    return () => container.removeEventListener("scroll", handleScroll)
  }, [])

  // Auto-scroll when new messages arrive or when autoScroll is true
  useEffect(() => {
    if (autoScroll) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, autoScroll]);

  // Reset autoScroll to true when sending new message
  const handleSend = (text: string, file?: File) => {
    setAutoScroll(true)
    onSend(text, file)
  }

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Main scrollable area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        {!hasMessages ? (
          <div className="h-full flex flex-col items-center justify-center p-4">
            <div className="space-y-6 max-w-4xl w-full flex flex-col justify-center text-start">
              <h1 className="text-5xl font-bold">Good to see you {user?.name}! </h1>
              <p className="text-2xl p-2 text-muted-foreground">How can I help you today?</p>
            </div>

            {/* Followups when no messages */}
            <div className="mt-12 max-w-4xl w-full">
              <Followups
                items={followups}
                onPick={onPickFollowup}
              />
            </div>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto w-full px-4 py-4">
            {messages.map(m => (
              <MessageBubble
                key={m.id}
                role={m.role === 'system' ? 'assistant' : (m.role as 'user' | 'assistant')}
                content={m.content}
                thinking={m.thinking}
              />
            ))}

            {/* Followups after assistant */}
            <div className="py-2">
              <Followups
                items={
                  messages[messages.length - 1]?.role === 'assistant'
                    ? followups
                    : []
                }
                onPick={onPickFollowup}
              />
            </div>
            <div ref={bottomRef} />
          </div>
        )}
        {!loading && (
          <ChatActions
            messages={messages}
            chatId={chatId || ''}
            chatTitle={chatTitle || ''}
            onRewrite={() => {
              // Regenerate last response
              const lastUserMsg = messages
                .slice()
                .reverse()
                .find(m => m.role === 'user')
              if (lastUserMsg?.content) {
                onSend(lastUserMsg.content)
              }
            }}
          />
        )}

      </div>
      <div className="bg-background">

        <div className="max-w-4xl mx-auto w-full px-4 py-2">
          <ChatInput
            onSend={handleSend}
            onStop={onStop}
            loading={loading}
            hasMessages={hasMessages}
          />
        </div>

        {/* Jump-to-bottom button - only show when not in auto-scroll mode */}
        {!autoScroll && (
          <button
            onClick={() => {
              bottomRef.current?.scrollIntoView({ behavior: "smooth" })
              setAutoScroll(true)
            }}
            className="fixed bg-muted text-muted-foreground p-3 rounded-full shadow-lg hover:scale-110 transition-all hover:bg-accent z-50"
            style={{
              left: 'calc(50% + 120px)',
              transform: 'translateX(-50%)',
              bottom: '140px'
            }}
          >
            <ArrowDown size={28} />
          </button>
        )}
        <div className="max-w-4xl mx-auto w-full px-4 pb-2 text-center text-xs text-muted-foreground">
          Neurosurfer models may generate content that is not accurate or suitable for all purposes. Use with caution.
        </div>
      </div>
    </div>
  )
}