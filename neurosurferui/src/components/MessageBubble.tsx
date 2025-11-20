import React from 'react'
import { Bot, User } from './Icons'
import ThinkingAccordion from './ThinkingAccordion'
import Markdown from './Markdown'
import type { ChatMessage } from '../lib/types'
import FileAttachmentList from './FileAttachmentList'

type Props = {
  message: ChatMessage & { role: 'user' | 'assistant' }
}

export default function MessageBubble({ message }: Props) {
  const { role, content, thinking, files } = message
  const isUser = role === 'user'

  return (
    <div className={`flex gap-3 my-2 w-full ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="flex-shrink-0 w-9 h-9 rounded-full bg-primary/10 text-primary flex items-center justify-center">
          <Bot size={18} />
        </div>
      )}

      <div className={`${isUser ? 'max-w-[65%]' : 'max-w-[80%]'} flex flex-col gap-1`}>
        {/* Thinking trace (assistant only) */}
        {!isUser && (
          <div className="w-full">
            <ThinkingAccordion content={thinking || ''} />
          </div>
        )}

        {/* Attached files (user or assistant) */}
        {files && files.length > 0 && (
          <FileAttachmentList files={files} />
        )}

        {/* Text content */}
        {content && (
          <div
            className={
              isUser
                ? 'bg-muted px-3 py-2 rounded-2xl whitespace-pre-wrap text-sm'
                : 'px-1 py-1 rounded-2xl text-sm'
            }
          >
            {isUser ? (
              content
            ) : (
              <Markdown source={content || '…'} />
            )}
          </div>
        )}
      </div>

      {isUser && (
        <div className="flex-shrink-0 w-9 h-9 rounded-full bg-secondary/20 text-secondary-foreground flex items-center justify-center">
          <User size={18} />
        </div>
      )}
    </div>
  )
}
