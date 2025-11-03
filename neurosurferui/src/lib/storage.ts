export function saveModel(id: string) {
  localStorage.setItem('model', id)
}
export function loadModel(): string | null {
  return localStorage.getItem('model')
}
export function saveTheme(mode: 'dark'|'light') {
  localStorage.setItem('theme', mode)
}
export function loadTheme(): 'dark'|'light' {
  return (localStorage.getItem('theme') as any) || 'dark'
}
