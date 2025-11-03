import { useEffect, useState } from 'react'
import { loadTheme, saveTheme } from '../lib/storage'

export function useTheme() {
  const [theme, setTheme] = useState<'dark'|'light'>(loadTheme())

  useEffect(() => {
    const el = document.documentElement
    if (theme === 'dark') el.classList.add('dark'); else el.classList.remove('dark')
    saveTheme(theme)
  }, [theme])

  return { theme, setTheme }
}
