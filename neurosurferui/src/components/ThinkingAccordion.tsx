import React, { useState } from 'react'

export default function ThinkingAccordion({ content }: { content: string }) {
  const [open, setOpen] = useState(false)
  if (!content) return null
  return (
    <div className="mb-2" style={{width: '60%'}}>
      <button
        className="w-full text-left px-3 py-2 rounded-t-lg text-sm font-semibold hover:font-bold transition-colors"
        onClick={() => setOpen(!open)}
      >
        {open ? '▼' : '►'} Thinking
      </button>
      {open && (
        <div className="border border-border rounded-b-lg p-3 text-sm whitespace-pre-wrap break-words max-h-60 overflow-y-auto">
          {content}
        </div>
      )}
    </div>
  )
}
