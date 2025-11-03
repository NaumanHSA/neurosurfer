# NeuroChat UI (React + Vite + Tailwind)

- Set backend URL in `.env` -> `VITE_BACKEND_API_URL=http://localhost:8000`
- Dev:
  ```bash
  npm install
  npm run dev
  ```
- Build:
  ```bash
  npm run build
  npm run preview
  ```

## Features
- Sidebar with chat history + New Chat
- Top bar with Model dropdown + theme toggle
- Chat area with welcome tiles until first message
- File upload + Send/Stop single button
- Streaming parsing supports `<think>...</think>` as collapsible "Reasoning"
