// 0.5 -> "30 seconds", 1 -> "1 minute", 2.5 -> "2 minutes 30 seconds".
// Returns null for missing/zero/invalid durations so callers can skip rendering.
export function formatDuration(minutes: number | undefined | null): string | null {
  if (minutes == null || !Number.isFinite(minutes) || minutes <= 0) return null
  const totalSeconds = Math.round(minutes * 60)
  if (totalSeconds === 0) return null
  const mins = Math.floor(totalSeconds / 60)
  const secs = totalSeconds % 60
  const plural = (n: number, unit: string) => `${n} ${unit}${n === 1 ? '' : 's'}`
  if (mins === 0) return plural(secs, 'second')
  if (secs === 0) return plural(mins, 'minute')
  return `${plural(mins, 'minute')} ${plural(secs, 'second')}`
}
