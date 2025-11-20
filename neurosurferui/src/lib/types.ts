export type Role = 'user' | 'assistant' | 'system'

export type ChatFile = {
    id: string
    name: string
    size?: number
    type?: string
    /**
     * URL that can be used to download the file again.
     * For freshly-uploaded files this can be a blob: URL; for
     * files coming from the backend it should be an HTTP URL.
     */
    url?: string
    /**
     * Optional preview URL (e.g. image thumbnail). If not provided,
     * `url` will be used for previews when applicable.
     */
    previewUrl?: string
    /**
     * Backend identifier for the file (e.g. database id / vector id).
     * Useful when you later wire up a dedicated download endpoint.
     */
    fileId?: string
}

export type ChatMessage = {
    id: string
    role: Role
    content: string
    thinking?: string
    createdAt: string
    /**
     * Optional files attached to this message. For now only user
     * messages use this, but the UI also supports assistant files.
     */
    files?: ChatFile[]
}

export type ModelInfo = {
    id: string
    name: string
    context_length?: number
    description?: string
}

export type ConversationSummary = {
    id: string
    title: string
    updatedAt: string
    messagesCount: number
}

export type UserInfo = {
    id: string
    email: string
    name: string
}

export type ChatCompletionChunk = {
    op_id?: string
    delta?: string
    finish_reason?: string
    error?: string
}

// purely for API payloads (wire format)
export type ChatFileWire = {
  id: string
  filename: string
  mime?: string
  size?: number
  downloadUrl: string
}

export type ChatMessageWire = {
  id: number
  role: Role
  content: string
  createdAt: number      // unix seconds
  files?: ChatFileWire[]
}

export type UploadedFileIn = {
    name: string;
    mime?: string;
    size?: number;
    base64: string
}