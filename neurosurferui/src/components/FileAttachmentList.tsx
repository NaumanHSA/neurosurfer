import React from 'react'
import { Download, FileText } from 'lucide-react'
import type { ChatFile } from '../lib/types'
import { downloadFile } from '../lib/api'

function isImage(file: ChatFile): boolean {
  const t = file.type || ''
  return t.startsWith('image/')
}

function formatBytes(size?: number): string {
  if (!size && size !== 0) return ''
  const units = ['B', 'KB', 'MB', 'GB']
  let s = size
  let i = 0
  while (s >= 1024 && i < units.length - 1) {
    s /= 1024
    i++
  }
  return `${s.toFixed(s < 10 && i > 0 ? 1 : 0)} ${units[i]}`
}

export default function FileAttachmentList({ files }: { files: ChatFile[] }) {
  if (!files || files.length === 0) return null

  return (
    <div className="mt-2 space-y-2">
      {files.map(file => {
        const fileId = file.id
        const href = file.url || file.previewUrl
        return (
          <div
            key={file.id}
            className="flex items-center gap-3 rounded-lg border border-border/60 bg-muted/60 p-2"
          >
            <div className="flex-shrink-0">
              {isImage(file) && href ? (
                <img
                  src={href}
                  alt={file.name}
                  className="h-12 w-12 rounded-md object-cover border border-border/50"
                />
              ) : (
                <div className="h-10 w-10 flex items-center justify-center rounded-md bg-background border border-border/50">
                  <FileText className="h-5 w-5 text-muted-foreground" />
                </div>
              )}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-xs font-medium truncate">{file.name}</div>
              {file.size != null && (
                <div className="text-[10px] text-muted-foreground">
                  {formatBytes(file.size)}
                </div>
              )}
            </div>
            {fileId && <button
              onClick={() => downloadFile(fileId)}
              className="inline-flex items-center gap-1 text-[11px] rounded-full border px-2 py-1 text-muted-foreground hover:bg-background/60"
            >
              <Download className="h-3 w-3" />
              <span>Download</span>
            </button>
            }
          </div>
        )
      })}
    </div>
  )
}
