// src/components/Markdown.tsx
import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { Copy } from 'lucide-react'

function Code({
  inline,
  className,
  children,
  ...props
}: React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement> & { inline?: boolean }) {
  const isLang = /(^|\s)language-\w+/.test(className || '')
  const text = String(children ?? '')
  const isInline = !!(inline || (!isLang && !text.includes('\n')))

  if (isInline) {
    // Inline code chip
    return (
      <code
        className="px-1 py-0.5 rounded bg-muted border border-border text-[0.95em] font-mono"
        {...props}
      >
        {children}
      </code>
    )
  }

  // Fenced block (with optional language)
  const lang = (className || '').match(/language-(\w+)/)?.[1] ?? 'text'

  return (
    <div className="code-block not-prose border border-border rounded-xl overflow-hidden my-3">
      <div className="code-block__header">
        <span className="opacity-80 lowercase">{lang}</span>
        <button className="bg-transparent" type="button" onClick={() => navigator.clipboard.writeText(text)}>
          <Copy size={18} className="inline border-none bg-transparent hover:scale-110" aria-label="Copy code to clipboard" />
        </button>
      </div>

      {/* Important bits:
          - whitespace-pre preserves indentation
          - p-2 gives small padding
          - we DO NOT add padding on <code>; we’ll strip hljs padding via CSS override below
      */}
      <div className="code-block__body">
        <code className={className} {...props}>
          {children}
        </code>
      </div>
    </div>
  )
}

export default function Markdown({ source, className = '' }: { source: string; className?: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]}
      components={{ code: Code }}
      className={`prose prose-invert max-w-none ${className}`}
    >
      {source}
    </ReactMarkdown>
  )
}

