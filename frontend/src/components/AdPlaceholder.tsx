/**
 * Placeholder for future ad integration (AdSense/Mediavine)
 */
export function AdPlaceholder({ className = '' }: { className?: string }) {
  return (
    <div
      className={`
        w-full max-w-sm mx-auto aspect-[4/3]
        bg-peach-200/30 rounded-lg
        flex items-center justify-center
        border-2 border-dashed border-white/30
        ${className}
      `}
    >
      <span className="text-white/50 font-medium text-lg">AD</span>
    </div>
  )
}


