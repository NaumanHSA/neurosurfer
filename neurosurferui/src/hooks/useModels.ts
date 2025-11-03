import { useEffect, useState } from 'react'
import { fetchModels } from '../lib/api'
import { ModelInfo } from '../lib/types'


// useModels.ts
export function useModels() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState<string | null>(null);

  async function refreshModels() {
    try {
      const ms = await fetchModels();
      setModels(ms);
      if (ms.length && !model) {
        setModel(ms[0].id);
      }
    } catch {
      setModels([]);
    }
  }

  useEffect(() => { refreshModels(); }, []);

  return { models, model, setModel, refreshModels };
}