import React from 'react'
import ModelDropdown from './ModelDropdown'

export default function TopBar() {
  return (
    <div className="w-full bg-card/50 backdrop-blur mb-2 top-0 z-20">
      <div className="w-full px-4 py-3 flex items-center justify-between">
        <div className="text-xl font-semibold">Neurosurfer UI</div>
        <div className="flex items-center gap-3">
          <ModelDropdown />
        </div>
      </div>
    </div>
  )
}
