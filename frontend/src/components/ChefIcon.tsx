/**
 * Chef hat icon - app logo
 */
export function ChefIcon({ className = 'w-16 h-16' }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* Chef hat */}
      <path
        d="M32 8C24.268 8 18 14.268 18 22C14.134 22 11 25.134 11 29C11 32.866 14.134 36 18 36V48H46V36C49.866 36 53 32.866 53 29C53 25.134 49.866 22 46 22C46 14.268 39.732 8 32 8Z"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Hat band */}
      <path
        d="M18 48H46V52C46 54.209 44.209 56 42 56H22C19.791 56 18 54.209 18 52V48Z"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Pleats */}
      <path
        d="M26 36V48"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M32 36V48"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M38 36V48"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  )
}


