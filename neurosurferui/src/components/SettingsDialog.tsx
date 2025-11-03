// src/components/SettingsDialog.tsx
import { useState, useEffect } from 'react'
import { deleteAccount } from '../lib/api'
import { Check, Palette, MessageSquare, User } from 'lucide-react'

export type Theme = {
  name: string
  primary: string
  primaryForeground: string
  ring: string
}

const THEMES: Theme[] = [
  {
    name: 'Orange',
    primary: 'hsl(24, 100%, 50%)',
    primaryForeground: 'hsl(0, 0%, 100%)',
    ring: 'hsl(24, 100%, 50%)',
  },
  {
    name: 'Blue',
    primary: 'hsl(221, 83%, 53%)',
    primaryForeground: 'hsl(210, 40%, 98%)',
    ring: 'hsl(221, 83%, 53%)',
  },
  {
    name: 'Green',
    primary: 'hsl(142, 71%, 45%)',
    primaryForeground: 'hsl(0, 0%, 100%)',
    ring: 'hsl(142, 71%, 45%)',
  },
  {
    name: 'Purple',
    primary: 'hsl(262, 83%, 58%)',
    primaryForeground: 'hsl(0, 0%, 100%)',
    ring: 'hsl(262, 83%, 58%)',
  },
  {
    name: 'Pink',
    primary: 'hsl(330, 81%, 60%)',
    primaryForeground: 'hsl(0, 0%, 100%)',
    ring: 'hsl(330, 81%, 60%)',
  },
]

export type ChatSettings = {
  systemPrompt: string
  messageHistoryLimit: number
  temperature: number
  topP: number
  maxTokens?: number
}

const DEFAULT_CHAT_SETTINGS: ChatSettings = {
  systemPrompt: 'You are a helpful AI assistant.',
  messageHistoryLimit: 10,
  temperature: 0.7,
  topP: 1.0,
  maxTokens: 4096,
};
const SETTINGS_KEY = 'neurosurfer_chat_settings';

type AccountSettings = {
  name: string
  email: string
}

type SettingsTab = 'appearance' | 'chat' | 'account'

type Props = {
  open: boolean
  onClose: () => void
  onDeleted: () => void
  onThemeChange?: (theme: Theme) => void
  user?: { id: string; name: string; email?: string | null } | null,
}

function useTheme() {
  const [currentTheme, setCurrentTheme] = useState<Theme>(THEMES[0])

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme')
    if (savedTheme) {
      const theme = THEMES.find(t => t.name === savedTheme) || THEMES[0]
      setCurrentTheme(theme)
      applyTheme(theme)
    } else {
      // Apply default theme if no saved theme
      applyTheme(THEMES[0])
    }
  }, [])

  const applyTheme = (theme: Theme) => {
    const root = document.documentElement
    
    // Extract HSL values from theme
    const primaryMatch = theme.primary.match(/hsl\((\d+),\s*([\d.]+)%,\s*([\d.]+)%\)/)
    const primaryFgMatch = theme.primaryForeground.match(/hsl\((\d+),\s*([\d.]+)%,\s*([\d.]+)%\)/)
    const ringMatch = theme.ring.match(/hsl\((\d+),\s*([\d.]+)%,\s*([\d.]+)%\)/)
    
    if (primaryMatch) {
      const [_, h, s, l] = primaryMatch
      root.style.setProperty('--primary', `${h} ${s}% ${l}%`)
      
      // Update secondary and accent colors based on primary
      const secondaryL = Math.max(20, parseFloat(l) - 10)
      root.style.setProperty('--secondary', `${h} ${s}% ${secondaryL}%`)
      root.style.setProperty('--accent', `${h} ${s}% ${l}%`)
    }
    
    if (primaryFgMatch) {
      const [_, h, s, l] = primaryFgMatch
      root.style.setProperty('--primary-foreground', `${h} ${s}% ${l}%`)
    }
    
    if (ringMatch) {
      const [_, h, s, l] = ringMatch
      root.style.setProperty('--ring', `${h} ${s}% ${l}%`)
    }
    
    localStorage.setItem('theme', theme.name)
    // Force a re-render of components that might be using these variables
    document.body.style.setProperty('color-scheme', theme.name.toLowerCase())
  }

  const setTheme = (theme: Theme) => {
    setCurrentTheme(theme)
    applyTheme(theme)
  }

  return { currentTheme, setTheme }
}

export function loadSettings(): ChatSettings {
  if (typeof window === 'undefined') {
    return { ...DEFAULT_CHAT_SETTINGS };
  }

  try {
    const saved = localStorage.getItem(SETTINGS_KEY);
    if (saved) {
      return { ...DEFAULT_CHAT_SETTINGS, ...JSON.parse(saved) };
    }
  } catch (error) {
    console.error('Failed to load settings from localStorage', error);
  }
  
  // Save default settings if none exist
  saveSettings(DEFAULT_CHAT_SETTINGS);
  return { ...DEFAULT_CHAT_SETTINGS };
}

// Save settings to localStorage
export function saveSettings(settings: Partial<ChatSettings>): void {
  if (typeof window === 'undefined') return;
  
  try {
    // Try to get existing settings directly from localStorage
    let current: Partial<ChatSettings> = {};
    const saved = localStorage.getItem(SETTINGS_KEY);
    if (saved) {
      try {
        current = JSON.parse(saved);
      } catch (e) {
        console.error('Failed to parse saved settings', e);
      }
    }
    
    // Merge with new settings, using defaults as fallback
    const newSettings = { ...DEFAULT_CHAT_SETTINGS, ...current, ...settings };
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(newSettings));
  } catch (error) {
    console.error('Failed to save settings to localStorage', error);
  }
}

export default function SettingsDialog({ open, onClose, onDeleted, onThemeChange, user }: Props) {
  const [activeTab, setActiveTab] = useState<SettingsTab>('appearance')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [chatSettings, setChatSettings] = useState<ChatSettings>(() => loadSettings())
  const [accountSettings, setAccountSettings] = useState<AccountSettings>({
    name: user?.name || '',
    email: user?.email || ''
  })
  const { currentTheme, setTheme } = useTheme()
  const maxTokensValues = [0, 512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 42368, 64000, 81920, 102400, 128000];
  useEffect(() => {
    if (user) {
      setAccountSettings({
        name: user.name || '',
        email: user.email || ''
      })
    }
  }, [user])

  const handleThemeChange = (theme: Theme) => {
    setTheme(theme)
    onThemeChange?.(theme)
  }

  const handleChatSettingChange = <K extends keyof ChatSettings>(key: K, value: ChatSettings[K]) => {
    const newSettings = {
      ...chatSettings,
      [key]: value
    };
    setChatSettings(newSettings);
    saveSettings(newSettings);
  }

  const handleAccountSettingChange = <K extends keyof AccountSettings>(key: K, value: AccountSettings[K]) => {
    const newSettings = {
      ...accountSettings,
      [key]: value
    };
    setAccountSettings(newSettings);
    // Optionally save account settings if needed
    // saveAccountSettings(newSettings);
  }

  async function handleDeleteAccount() {
    if (!password) {
      setErr('Please enter your password to confirm.')
      return
    }
    try {
      setBusy(true)
      setErr(null)
      await deleteAccount(password)
      onDeleted()
    } catch (e: any) {
      setErr(e?.message || 'Failed to delete account.')
    } finally {
      setBusy(false)
    }
  }

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <div className="relative z-10 w-full max-w-2xl h-[60vh] flex flex-col rounded-2xl border border-border bg-card shadow-xl">
        <div className="p-6 border-b border-border">
          <h2 className="text-xl font-semibold">Settings</h2>
        </div>

        <div className="flex flex-1 overflow-hidden">
          {/* Sidebar */}
          <div className="w-55 border-r border-border p-4 space-y-1">
            <button
              onClick={() => setActiveTab('appearance')}
              className={`w-full text-left px-3 py-2 rounded-md flex items-center space-x-2 ${
                activeTab === 'appearance' ? 'bg-accent' : 'hover:bg-accent/50'
              }`}
            >
              <Palette className="h-4 w-4" />
              <span>Appearance</span>
            </button>
            <button
              onClick={() => setActiveTab('chat')}
              className={`w-full text-left px-3 py-2 rounded-md flex items-center space-x-2 ${
                activeTab === 'chat' ? 'bg-accent' : 'hover:bg-accent/50'
              }`}
            >
              <MessageSquare className="h-4 w-4" />
              <span>Chat & Messages</span>
            </button>
            <button
              onClick={() => setActiveTab('account')}
              className={`w-full text-left px-3 py-2 rounded-md flex items-center space-x-2 ${
                activeTab === 'account' ? 'bg-accent' : 'hover:bg-accent/50'
              }`}
            >
              <User className="h-4 w-4" />
              <span>Account</span>
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 p-6 overflow-y-auto">
            {activeTab === 'appearance' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium mb-4">Theme</h3>
                  <div className="grid grid-cols-5 gap-3">
                    {THEMES.map((theme) => (
                      <button
                        key={theme.name}
                        onClick={() => handleThemeChange(theme)}
                        className={`w-10 h-10 rounded-full mb-2 ${
                          currentTheme.name === theme.name ? 'w-11 h-11 ring-2 ring-primary' : 'border-border hover:border-primary/50'
                        }`}
                        style={{ backgroundColor: theme.primary }}
                      >
                        <div
                          className="w-10 h-10 rounded-full mb-4"
                          style={{ backgroundColor: theme.primary }}
                        />
                        <span className="text-sm">{theme.name}</span>
                        {currentTheme.name === theme.name && (
                          <Check className="ml-3 h-4 w-4 text-primary mt-1" />
                        )}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}
            {activeTab === 'chat' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium mb-4">Chat Settings</h3>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium mb-1">
                        System Prompt
                      </label>
                      <textarea
                        value={chatSettings.systemPrompt}
                        onChange={(e) => handleChatSettingChange('systemPrompt', e.target.value)}
                        className="w-full p-2 border border-border rounded-md bg-background text-foreground min-h-[100px]"
                        placeholder="You are a helpful AI assistant..."
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium mb-1">
                        Message History Limit: {chatSettings.messageHistoryLimit}
                      </label>
                      <input
                        type="range"
                        min="1"
                        max="50"
                        value={chatSettings.messageHistoryLimit}
                        onChange={(e) => handleChatSettingChange('messageHistoryLimit', Number(e.target.value))}
                        className="w-full"
                      />
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>1</span>
                        <span>50</span>
                      </div>
                    </div>

                    <div className="space-y-4 pt-4 border-t border-border">
                      <h4 className="font-medium">Model Parameters</h4>
                      
                      <div>
                        <label className="block text-sm font-medium mb-1">
                          Temperature: {chatSettings.temperature.toFixed(1)}
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="2"
                          step="0.1"
                          value={chatSettings.temperature}
                          onChange={(e) => handleChatSettingChange('temperature', parseFloat(e.target.value))}
                          className="w-full"
                        />
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>Precise</span>
                          <span>Balanced</span>
                          <span>Creative</span>
                        </div>
                      </div>

                      <div>
                        <label className="block text-sm font-medium mb-1">
                          Top P: {chatSettings.topP.toFixed(1)}
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.1"
                          value={chatSettings.topP}
                          onChange={(e) => handleChatSettingChange('topP', parseFloat(e.target.value))}
                          className="w-full"
                        />
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>Focused</span>
                          <span>Balanced</span>
                          <span>Diverse</span>
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between items-center mb-1">
                          <label className="block text-sm font-medium">
                            Max Tokens: {chatSettings.maxTokens ? (chatSettings.maxTokens === -1 ? 'Unlimited' : chatSettings.maxTokens.toLocaleString()) : '4096'}
                          </label>
                          <button
                            onClick={() => handleChatSettingChange('maxTokens', 4096)}
                            className="text-xs text-muted-foreground hover:text-foreground"
                          >
                            Reset to 4K
                          </button>
                        </div>
                        <input
                          type="range"
                          min="0"
                          max="14"
                          step="1"
                          value={chatSettings.maxTokens ? maxTokensValues.indexOf(chatSettings.maxTokens) : 4}
                          onChange={(e) => {
                            const value = parseInt(e.target.value);
                            const maxTokens = value === 0 ? -1 : maxTokensValues[value];
                            handleChatSettingChange('maxTokens', maxTokens);
                          }}
                          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                        />
                        <div className="flex justify-between text-xs text-muted-foreground mt-1">
                          <span>Unlimited</span>
                          <span>4K</span>
                          <span>12K</span>
                          <span>32K</span>
                          <span>64K</span>
                          <span>128K</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'account' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium mb-4">Profile Information</h3>
                  
                  <div className="space-y-4">
                    <div className="space-y-1">
                      <div>
                        <span className="text-sm font-medium">Name</span>
                        <p className="text-sm text-muted-foreground">{accountSettings.name || 'Not provided'}</p>
                      </div>
                      <div>
                        <span className="text-sm font-medium">Email</span>
                        <p className="text-sm text-muted-foreground">{accountSettings.email || 'Not provided'}</p>
                      </div>
                      <p className="text-xs text-muted-foreground pt-2">
                        Contact support to update your account information.
                      </p>
                    </div>

                    <div className="pt-4 border-t border-border">
                      <h4 className="font-medium text-destructive mb-2">Danger Zone</h4>
                      <div className="space-y-3">
                        <input
                          type="password"
                          value={password}
                          onChange={(e) => setPassword(e.target.value)}
                          placeholder="Enter your password to confirm"
                          className="w-full p-2 border border-border rounded-md bg-background text-foreground"
                          disabled={busy}
                        />
                        <button
                          onClick={handleDeleteAccount}
                          disabled={busy}
                          className="px-4 py-2 bg-destructive/10 text-destructive rounded-md hover:bg-destructive/20 transition-colors disabled:opacity-50"
                        >
                          {busy ? 'Deleting...' : 'Delete Account'}
                        </button>
                        {err && <p className="text-sm text-destructive">{err}</p>}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="flex justify-end space-x-3 p-4 border-t border-border">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium rounded-md border border-border hover:bg-accent/50"
          >
            Cancel
          </button>
          <button
            onClick={() => {
              // Save settings logic here
              onClose()
            }}
            className="px-4 py-2 text-sm font-medium rounded-md bg-primary text-primary-foreground hover:bg-primary/90"
          >
            Save Changes
          </button>
        </div>
      </div>
    </div>
  )
}
