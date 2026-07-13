/**
 * App logo - the same chef hat mark used for the favicon and the PWA icons.
 *
 * The asset's background is the app peach (#DB7256) exactly, so it sits on the
 * app background with no visible edge and reads as a floating white mark.
 */
export function ChefIcon({ className = 'w-16 h-16' }: { className?: string }) {
  return (
    <img
      src="/assets/makerai.png"
      alt="MakerAI"
      className={className}
      draggable={false}
    />
  )
}
