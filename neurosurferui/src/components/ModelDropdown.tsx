import React, { useState, useRef, useEffect } from 'react';
import { ChevronDown, Check } from 'lucide-react';
import { useModels } from '../hooks/useModels';
import clsx from 'clsx';

export default function ModelDropdown() {
  const { models, model: selectedModel, setModel } = useModels();
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const selectedModelData = models.find(m => m.id === selectedModel) || models[0];

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={clsx(
          'flex items-center justify-between w-full min-w-[220px] px-4 py-2 text-sm font-medium',
          'bg-muted hover:bg-muted/80 rounded-lg transition-colors',
          'border border-border focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-background focus:ring-primary/50',
          'text-foreground/90 hover:text-foreground'
        )}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-label="Select model"
      >
        <span className="truncate">{selectedModelData?.name || selectedModelData?.id || 'Select model'}</span>
        <ChevronDown className={clsx(
          'ml-2 h-4 w-4 transition-transform duration-200',
          isOpen ? 'transform rotate-180' : ''
        )} />
      </button>

      {isOpen && (
        <ul
          className="absolute z-50 mt-1 w-full bg-card border border-border rounded-lg shadow-lg py-1 max-h-60 overflow-auto focus:outline-none"
          role="listbox"
          tabIndex={-1}
        >
          {models.map((model) => (
            <li
              key={model.id}
              className={clsx(
                'px-4 py-2 text-sm cursor-pointer flex items-center justify-between',
                'hover:bg-muted/50 transition-colors',
                model.id === selectedModel ? 'bg-primary/10 text-primary' : 'text-foreground'
              )}
              role="option"
              aria-selected={model.id === selectedModel}
              onClick={() => {
                setModel(model.id);
                setIsOpen(false);
              }}
            >
              <span className="truncate">{model.name || model.id}</span>
              {model.id === selectedModel && (
                <Check className="h-4 w-4 ml-2 text-primary flex-shrink-0" />
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
