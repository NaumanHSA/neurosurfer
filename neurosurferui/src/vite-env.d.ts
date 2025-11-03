
interface ImportMetaEnv {
  readonly VITE_BACKEND_URL?: string; // add more VITE_* here as needed
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}