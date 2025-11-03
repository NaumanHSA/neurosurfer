import { useState } from 'react'
import { login, register } from '../lib/api'

export default function AuthModal({ onClose, onAuthed }: { onClose: () => void, onAuthed: () => void }) {
  const [mode, setMode] = useState<'login'|'register'>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [name, setName] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const [validationErrors, setValidationErrors] = useState<{
    email?: string
    password?: string
    confirmPassword?: string
    name?: string
  }>({})

  // Email validation regex
  const validateEmail = (email: string): boolean => {
    const emailRegex = /^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$/i
    return emailRegex.test(email)
  }

  // Password validation (min 8 chars, 1 special char, 1 number, 1 uppercase)
  const validatePassword = (password: string): boolean => {
    const passwordRegex = /^(?=.*[A-Z])(?=.*[0-9])(?=.*[_!@#$%^&*])[A-Za-z\d_!@#$%^&*]{8,}$/
    return passwordRegex.test(password)
  }

  const validate = (): boolean => {
    const errors: typeof validationErrors = {}

    if (!email.trim()) {
      errors.email = 'Email is required'
    } else if (!validateEmail(email)) {
      errors.email = 'Please enter a valid email address'
    }

    if (!password) {
      errors.password = 'Password is required'
    } else if (mode === 'register' && !validatePassword(password)) {
      errors.password = 'Password must be at least 8 characters with 1 uppercase, 1 number, and 1 special character'
    }

    if (mode === 'register') {
      if (!name.trim()) {
        errors.name = 'Name is required'
      }
      if (!confirmPassword) {
        errors.confirmPassword = 'Please confirm your password'
      } else if (password !== confirmPassword) {
        errors.confirmPassword = 'Passwords do not match'
      }
    }

    setValidationErrors(errors)
    return Object.keys(errors).length === 0
  }

  async function submit() {
    setErr(null)
    if (!validate()) return

    try {
      if (mode === 'register') {
        await register(name, email, password)
      }
      await login(email, password)
      onAuthed()
      onClose()
    } catch (e: any) {
      // Handle specific error cases for better user experience
      const errorMessage = e.message || 'Invalid email or password. Please try again.'
      
      if (errorMessage.toLowerCase().includes('user not found')) {
        setErr('User not found. Please check your credentials and try again.')
      }
      else if (errorMessage.toLowerCase().includes('email already registered')) {
        setErr('This email is already registered. Please use a different email or try signing in instead.')
      } else if (errorMessage.toLowerCase().includes('invalid email or password')) {
        setErr('Invalid email or password. Please check your credentials and try again.')
      } else if (errorMessage.toLowerCase().includes('network') || errorMessage.toLowerCase().includes('fetch')) {
        setErr('Network error. Please check your connection and try again.')
      } else {
        setErr(errorMessage)
      }
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      submit()
    }
  }

  return (
    <div className="fixed inset-0 bg-background/95 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-card border border-border rounded-2xl w-full max-w-md p-8 space-y-4 shadow-2xl">
        <div className="text-center space-y-2">
          <h2 className="text-2xl font-bold">{mode === 'login' ? 'Welcome Back' : 'Create Account'}</h2>
          <p className="text-sm text-muted-foreground">
            {mode === 'login' ? 'Sign in to continue to NeuroChat' : 'Sign up to get started with NeuroChat'}
          </p>
        </div>

        <div className="space-y-4">
          {mode === 'register' && (
            <div>
              <label className="block text-sm font-medium mb-1.5">Name *</label>
              <input 
                className={`w-full bg-muted rounded-lg px-3 py-2.5 outline-none focus:ring-2 focus:ring-primary ${
                  validationErrors.name ? 'ring-2 ring-red-500' : ''
                }`}
                placeholder="John Doe" 
                value={name} 
                onChange={e => {
                  setName(e.target.value)
                  setValidationErrors(prev => ({ ...prev, name: undefined }))
                }}
                onKeyPress={handleKeyPress}
              />
              {validationErrors.name && (
                <p className="text-red-500 text-xs mt-1">{validationErrors.name}</p>
              )}
            </div>
          )}

          <div>
            <label className="block text-sm font-medium mt-6 mb-1.5">Email *</label>
            <input 
              className={`w-full bg-muted border border-border rounded-lg px-3 py-2.5 outline-none focus:ring-2 focus:ring-primary ${
                validationErrors.email ? 'ring-2 ring-red-500' : ''
              }`}
              placeholder="Your email" 
              type="email"
              value={email} 
              onChange={e => {
                setEmail(e.target.value)
                setValidationErrors(prev => ({ ...prev, email: undefined }))
              }}
              onKeyDown={handleKeyPress}
            />
            {validationErrors.email && (
              <p className="text-red-500 text-xs mt-1">{validationErrors.email}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium mt-6 mb-1.5">Password *</label>
            <input 
              className={`w-full bg-muted border border-border rounded-lg px-3 py-2.5 outline-none focus:ring-2 focus:ring-primary ${
                validationErrors.password ? 'ring-2 ring-red-500' : ''
              }`}
              placeholder={mode === 'register' ? 'Min. 8 characters' : 'Enter your password'}
              type="password" 
              value={password} 
              onChange={e => {
                setPassword(e.target.value)
                setValidationErrors(prev => ({ ...prev, password: undefined }))
              }}
              onKeyDown={handleKeyPress}
            />
            {validationErrors.password && (
              <p className="text-red-500 text-xs mt-1">{validationErrors.password}</p>
            )}
          </div>

          {mode === 'register' && (
            <div>
              <label className="block text-sm font-medium mt-4 mb-1.5">Confirm Password *</label>
              <input 
                className={`w-full bg-muted border border-border rounded-lg px-3 py-2.5 outline-none focus:ring-2 focus:ring-primary ${
                  validationErrors.confirmPassword ? 'ring-2 ring-red-500' : ''
                }`}
                placeholder="Re-enter your password" 
                type="password" 
                value={confirmPassword} 
                onChange={e => {
                  setConfirmPassword(e.target.value)
                  setValidationErrors(prev => ({ ...prev, confirmPassword: undefined }))
                }}
                onKeyDown={handleKeyPress}
              />
              {validationErrors.confirmPassword && (
                <p className="text-red-500 text-xs mt-1">{validationErrors.confirmPassword}</p>
              )}
            </div>
          )}
        </div>

        {err && (
          <div className="bg-red-500/10 border border-red-500 rounded-lg px-3 py-2 text-red-500 text-sm">
            {err}
          </div>
        )}

        <div className="space-y-3 pt-2">
          <button 
            className="w-full px-4 py-2.5 bg-primary text-white rounded-lg font-medium hover:opacity-90 transition-opacity" 
            onClick={submit}
          >
            {mode === 'login' ? 'Sign In' : 'Create Account'}
          </button>
          
          <div className="text-center text-sm">
            <span className="text-muted-foreground">
              {mode === 'login' ? "Don't have an account? " : 'Already have an account? '}
            </span>
            <button 
              className="text-primary hover:underline font-medium"
              onClick={() => {
                setMode(mode === 'login' ? 'register' : 'login')
                setValidationErrors({})
                setErr(null)
              }}
            >
              {mode === 'login' ? 'Sign up' : 'Sign in'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
