import React, { useEffect, useMemo, useState, useRef } from 'react'
import { Menu, Plus, Search, X } from './Icons'
import { MoreHorizontal, Trash2, Share, LogOut, Settings } from 'lucide-react'
import type { ConversationSummary } from '../lib/types'
import { useClickOutside } from '../hooks/useClickOutside'

type Props = {
  convos: ConversationSummary[]
  onNew: () => void
  onOpen: (id: string) => void
  onDelete: (id: string) => void
  onShare: (id: string) => void
  newDisabled?: boolean
  user?: { id: string; name: string; email?: string | null } | null
  onLogout?: () => void
  onOpenSettings?: () => void
  /** ID of the currently selected conversation for highlight */
  selectedId?: string
}

export default function Sidebar({ convos, onNew, onOpen, onDelete, onShare, newDisabled = false, user, onLogout, onOpenSettings, selectedId}: Props) {
  const [open, setOpen] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [openMenuId, setOpenMenuId] = useState<string | null>(null)
  const [userMenuOpen, setUserMenuOpen] = useState(false)
  
  const chatMenuRef = useRef<HTMLDivElement>(null)
  const userMenuRef = useRef<HTMLDivElement>(null)

  // Close menus when clicking outside
  useClickOutside(chatMenuRef, () => setOpenMenuId(null))
  useClickOutside(userMenuRef, () => setUserMenuOpen(false))

  const filtered = useMemo(() => {
    return convos.filter((c) => 
      (c.title || '').toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [convos, searchQuery]);

  const initials = (name?: string) =>
    (name || '')
      .split(' ')
      .map((s) => s[0])
      .filter(Boolean)
      .slice(0, 2)
      .join('')
      .toUpperCase() || 'U'

  const handleOpen = (id: string) => {
    if (id === selectedId) return // already open; no-op to prevent extra work
    onOpen(id)
  }

  return (
    <div
      className={`h-full border-r border-border ${
        open ? 'w-80' : 'w-14'
      } transition-all duration-200 bg-card flex flex-col`}
    >
      {/* Top controls */}
      <div className="px-3 py-3 flex items-center gap-2 border-b border-border">
        <button
          className="p-2 rounded-lg hover:bg-accent"
          onClick={() => setOpen((v) => !v)}
          title="Toggle sidebar"
        >
          <Menu className="h-5 w-5" />
        </button>
        <div className="flex-1" />
        {open && (
          <div className="flex flex-col gap-2">
            <button
              className="px-3 py-2 rounded-lg bg-primary text-primary-foreground hover:opacity-90 text-sm disabled:opacity-50"
              onClick={() => onNew()}
              disabled={newDisabled}
              title="New chat"
            >
              <div className="flex items-center gap-1">
                <Plus className="h-4 w-4" />
                <span>New chat</span>
              </div>
            </button>
          </div>
        )}
      </div>

      {/* Search */}
      <div className="px-3 py-3 flex items-center gap-2 border-b border-border overflow-hidden">
        <div className="h-10 ml-auto relative w-full">
          <Search className="h-4 w-4 absolute left-2 top-2.5 text-muted-foreground" />
          {open && (
            <input
              className="w-full pl-8 pr-8 py-2 rounded-lg bg-muted outline-none focus:ring-2 focus:ring-ring text-sm"
              placeholder="Search chats…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          )}
          {open && searchQuery && (
            <button
              className="absolute right-2 top-2.5 text-muted-foreground"
              onClick={() => setSearchQuery('')}
              title="Clear"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {/* Chats list */}
      <div className="flex-1 overflow-auto">
        {open ? (
          <div className="p-2">
            {filtered.length ? (
              filtered.map((c) => {
                const isSelected = c.id === selectedId
                return (
                  <div key={`${c.id}-${c.updatedAt}`} className="group relative">
                    <button
                      className={`w-full text-left px-3 py-2 rounded-lg hover:bg-accent transition-colors ${
                        isSelected ? 'bg-accent/70 ring-1 ring-border' : ''
                      }`}
                      onClick={() => handleOpen(c.id)}
                      title={c.title}
                      aria-current={isSelected ? 'page' : undefined}
                    >
                      <div className="flex items-center gap-2">
                        {/* Left active indicator */}
                        <span
                          className={`inline-block h-2 w-2 rounded-full ${
                            isSelected ? 'bg-primary' : 'bg-transparent'
                          }`}
                        />
                        <div className="min-w-0 flex-1">
                          <div
                            className={`truncate ${
                              isSelected ? 'font-semibold' : ''
                            }`}
                          >
                            {c.title}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {new Date(c.updatedAt).toLocaleString()}
                          </div>
                        </div>
                      </div>
                    </button>

                    <div className="absolute right-2 top-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        className="p-1 rounded hover:bg-accent"
                        onClick={() => setOpenMenuId(openMenuId === c.id ? null : c.id)}
                        title="More"
                      >
                        <MoreHorizontal className="h-4 w-4" />
                      </button>
                    </div>

                    {openMenuId === c.id && (
                      <div 
                        ref={chatMenuRef}
                        className="absolute right-2 top-8 z-10 bg-popover border border-border rounded-lg shadow-lg py-1 w-44"
                      >
                        <button
                          className="w-full text-left px-3 py-2 hover:bg-accent flex items-center gap-2 text-sm"
                          onClick={() => {
                            setOpenMenuId(null)
                            onShare(c.id)
                          }}
                        >
                          <Share className="h-4 w-4" /> Export PDF
                        </button>
                        <button
                          className="w-full text-left px-3 py-2 hover:bg-accent flex items-center gap-2 text-sm text-red-600"
                          onClick={() => {
                            setOpenMenuId(null)
                            onDelete(c.id)
                          }}
                        >
                          <Trash2 className="h-4 w-4" /> Delete
                        </button>
                      </div>
                    )}
                  </div>
                )
              })
            ) : (
              <div className="flex items-center justify-center h-32 text-muted-foreground text-sm">
                {searchQuery ? 'No matching chats found' : 'No chats yet'}
              </div>
            )}
          </div>
        ) : (
          <div className="p-2 text-muted-foreground text-xs text-center">Chats</div>
        )}
      </div>

      {/* User section */}
      <div className="border-t border-border px-3 py-3">
        <button
          className="w-full flex items-center gap-3 px-2 py-2 rounded-xl hover:bg-accent"
          onClick={() => setUserMenuOpen((v) => !v)}
          title={user?.email || user?.name || 'User'}
        >
          <div className="h-9 w-9 rounded-full bg-primary border border-border flex items-center justify-center text-sm font-semibold">
            {initials(user?.name || user?.email || 'User')}
          </div>
          {open && (
            <div className="flex-1 min-w-0 text-left">
              <div className="truncate text-sm font-medium">
                {user?.name || 'User'}
              </div>
              <div className="truncate text-xs text-muted-foreground">
                {user?.email}
              </div>
            </div>
          )}
        </button>

        {userMenuOpen && (
          <div className="relative">
            <div 
              ref={userMenuRef}
              className="absolute bottom-14 left-2 right-2 z-20 bg-popover border border-border rounded-xl shadow-lg overflow-hidden"
            >
              <button
                className="w-full text-left px-3 py-2 hover:bg-accent flex items-center gap-2 text-sm"
                onClick={() => {
                  setUserMenuOpen(false)
                  onOpenSettings?.()
                }}
              >
                <Settings className="h-4 w-4" /> Settings
              </button>
              <button
                className="w-full text-left px-3 py-2 hover:bg-accent flex items-center gap-2 text-sm text-red-600"
                onClick={() => {
                  setUserMenuOpen(false)
                  onLogout?.()
                }}
              >
                <LogOut className="h-4 w-4" /> Logout
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
