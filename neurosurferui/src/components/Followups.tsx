import React from 'react';
import { ArrowDown } from 'lucide-react';

export default function Followups({ items, onPick }: { items: string[]; onPick: (q: string) => void }) {
  if (!items || items.length === 0) return null;
  return (
      <div className="w-full px-0 py-2 gap-2">
        <p className="text-muted-foreground text-sm opacity-80 flex items-center gap-2 mb-2">
          <ArrowDown size={22} className="text-muted-foreground animate-bounce" />
          Suggestions
        </p>
        {items.map((s, i) => (
          <button
            key={i}
            className="w-full py-1 bg-transparent text-md text-muted-foreground hover:text-white transition-all text-left"
            onClick={() => onPick(s)}
            title={s}
          >
            {s}
          </button>
        ))}
      </div>
  );
}