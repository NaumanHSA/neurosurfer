/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_NEUROSURFER_TOKEN?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
