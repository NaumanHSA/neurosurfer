import React, { useRef, useState } from 'react'
import { Send, Square, Plus } from './Icons'

export default function ChatInput({
  onSend,
  onStop,
  loading,
  hasMessages
}: { 
  onSend: (text: string, file?: File) => void, 
  onStop: () => void, 
  loading: boolean,
  hasMessages: boolean
}) {
  const [text, setText] = useState('')
  const [file, setFile] = useState<File | undefined>(undefined)
  const fileRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Add auto-resize function
  const handleTextareaResize = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const textarea = e.target
    textarea.style.height = 'auto'
    textarea.style.height = textarea.scrollHeight + 'px'
  }

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value)
    handleTextareaResize(e)
  }

  const handleSend = () => {
    if (!text.trim() && !file) return
    onSend(text, file)
    setText('')
    setFile(undefined)
    if (fileRef.current) fileRef.current.value = ''
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  return (
    <div className={`w-full ${!hasMessages ? 'max-w-4xl mx-0' : 'w-full'}`}>
      <div 
        className={`relative bg-card border border-border shadow-lg transition-all w-full focus-within:ring-2 focus-within:ring-ring`} 
        style={{ borderRadius: '2rem' }}
      >
        <div className="flex items-end px-2 py-2">
          <button
            className="p-2 rounded-full hover:bg-muted transition-colors mr-2"
            title="Upload a file"
            onClick={() => fileRef.current?.click()}
            type="button"
          >
            <Plus size={22} className="text-muted-foreground"/>
          </button>
          <input
            ref={fileRef}
            type="file"
            className="hidden"
            onChange={e => setFile(e.target.files?.[0])}
          />
          {/* text more than 20 lines becomes scrollable */}
          <textarea
            ref={textareaRef}
            value={text}
            onChange={handleTextChange}
            placeholder="Ask Anything"
            className="flex-1 px-2 py-2 bg-transparent border-none focus:outline-none focus:ring-0 text-foreground placeholder:text-muted-foreground resize-none overflow-hidden"
            rows={1}
            style={{
              overflowY: 'scroll',
              maxHeight: '500px'
            }}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
          />
          <div className="w-10 flex items-center justify-center ml-2">
            {!loading ? (
              <button 
                className={`p-2 rounded-full ${(text.trim() || file) ? 'bg-primary text-white' : 'text-muted-foreground'} hover:cursor-pointer transition-colors`} 
                onClick={handleSend} 
                title="Send"
                disabled={!text.trim() && !file}
                type="button"
              >
                <Send size={24}/>
              </button>
            ) : (
              <button 
                className="p-2 rounded-full bg-destructive text-white hover:cursor-pointer transition-colors" 
                onClick={onStop} 
                title="Stop"
                type="button"
              >
                <Square size={22}/>
              </button>
            )}
          </div>
        </div>
        {file && (
          <div className="px-4 pb-3 -mt-1">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 text-sm rounded-full bg-muted text-muted-foreground">
              {file.name}
              <button 
                onClick={() => {
                  setFile(undefined);
                  if (fileRef.current) fileRef.current.value = '';
                }}
                className="text-muted-foreground hover:text-foreground"
                type="button"
              >
                ×
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
