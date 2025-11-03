import React from 'react'
import { FileDown, Share, FileText, RefreshCw, Copy, Check } from 'lucide-react'
import { ChatMessage } from '../lib/types'

interface ChatActionsProps {
  messages: ChatMessage[]
  chatId: string | null
  chatTitle: string
  onRewrite: () => void
}

export default function ChatActions({ 
  messages, 
  chatId, 
  chatTitle,
  onRewrite 
}: ChatActionsProps) {
  const [copied, setCopied] = React.useState(false)
  const [exporting, setExporting] = React.useState(false)

  // Get the last assistant message
  const lastAssistantMessage = React.useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'assistant') {
        return messages[i]
      }
    }
    return null
  }, [messages])

  // Export as PDF
  const handleExportPDF = async () => {
    if (!chatId) return
    setExporting(true)
    try {
      const { exportChatAsPDF } = await import('../lib/exportPdf')
      await exportChatAsPDF(chatId, chatTitle)
    } catch (error) {
      console.error('PDF export failed:', error)
      alert('Failed to export PDF. Please try again.')
    } finally {
      setExporting(false)
    }
  }

  // Export as Markdown
  const handleExportMarkdown = () => {
    if (!messages.length) return
    
    let markdown = `# ${chatTitle}\n\n`
    markdown += `*Generated: ${new Date().toLocaleString()}*\n\n---\n\n`
    
    messages.forEach(msg => {
      const role = msg.role === 'user' ? 'User' : 'Assistant'
      const timestamp = new Date(msg.createdAt).toLocaleString()
      markdown += `## ${role}\n`
      markdown += `*${timestamp}*\n\n`
      markdown += `${msg.content}\n\n`
      markdown += `---\n\n`
    })
    
    // Create and download file
    const blob = new Blob([markdown], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${chatTitle.replace(/[^\w\-]+/g, '_')}_${Date.now()}.md`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  // Copy last assistant message to clipboard
  const handleCopy = async () => {
    if (!lastAssistantMessage?.content) return
    
    try {
      await navigator.clipboard.writeText(lastAssistantMessage.content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (error) {
      console.error('Copy failed:', error)
    }
  }

  if (!messages.length || !lastAssistantMessage) {
    return null
  }

  return (
    <div className="bg-background px-4 py-4">
      <div className="max-w-3xl mx-auto flex items-center gap-5">
        {/* Left side buttons */}
        <button
          onClick={handleExportPDF}
          disabled={exporting}
          className="flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-xl hover:bg-muted transition-colors disabled:opacity-50"
          style={{color: 'grey'}}
          title="Export as PDF"
        >
          <Share className="h-4 w-4" />
          <span>{exporting ? 'Exporting...' : 'Export PDF'}</span>
        </button>

        <button
          onClick={handleExportMarkdown}
          className="flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-xl hover:bg-muted transition-colors"
          style={{color: 'grey'}}
          title="Export as Markdown"
        >
          <FileText className="h-4 w-4" />
          <span>Export Markdown</span>
        </button>

        <button
          onClick={onRewrite}
          className="flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-xl hover:bg-muted transition-colors"
          style={{color: 'grey'}}
          title="Regenerate response"
        >
          <RefreshCw className="h-4 w-4" />
          <span>Rewrite</span>
        </button>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Right side button */}
        <button
          onClick={handleCopy}
          className="flex items-center gap-2 px-2 py-2 text-sm font-medium rounded-xl hover:bg-muted transition-colors"
          style={{color: 'grey'}}
          title="Copy response to clipboard"
        >
          {copied ? (
            <>
              <Check className="h-4 w-4 text-green-500" />
              <span className="text-green-500">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="h-4 w-4" />
            </>
          )}
        </button>
      </div>
    </div>
  )
}
