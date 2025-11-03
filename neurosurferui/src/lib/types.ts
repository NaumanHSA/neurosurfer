export type Role = 'user' | 'assistant' | 'system'

export type ChatMessage = {
  id: string
  role: Role
  content: string
  thinking?: string
  createdAt: string
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
