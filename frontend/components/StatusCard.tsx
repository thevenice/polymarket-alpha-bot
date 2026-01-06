interface StatusCardProps {
  title: string
  value: string
  subtitle: string
  status: 'success' | 'warning' | 'error' | 'info'
  icon?: React.ReactNode
}

const statusConfig = {
  success: {
    bg: 'bg-emerald/5',
    border: 'border-emerald/20',
    glow: 'shadow-[inset_0_0_30px_rgba(16,185,129,0.05)]',
    text: 'text-emerald',
    indicator: 'bg-emerald',
  },
  warning: {
    bg: 'bg-amber/5',
    border: 'border-amber/20',
    glow: 'shadow-[inset_0_0_30px_rgba(255,191,0,0.05)]',
    text: 'text-amber',
    indicator: 'bg-amber',
  },
  error: {
    bg: 'bg-rose/5',
    border: 'border-rose/20',
    glow: 'shadow-[inset_0_0_30px_rgba(244,63,94,0.05)]',
    text: 'text-rose',
    indicator: 'bg-rose',
  },
  info: {
    bg: 'bg-cyan/5',
    border: 'border-cyan/20',
    glow: 'shadow-[inset_0_0_30px_rgba(0,212,255,0.05)]',
    text: 'text-cyan',
    indicator: 'bg-cyan',
  },
}

const defaultIcons = {
  success: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  warning: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
    </svg>
  ),
  error: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
    </svg>
  ),
  info: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" />
    </svg>
  ),
}

export function StatusCard({ title, value, subtitle, status, icon }: StatusCardProps) {
  const config = statusConfig[status]
  const displayIcon = icon || defaultIcons[status]

  return (
    <div
      className={`
        relative overflow-hidden rounded-xl border p-5
        ${config.bg} ${config.border} ${config.glow}
        transition-all duration-300 hover:scale-[1.02]
        group
      `}
    >
      {/* Background gradient accent */}
      <div
        className={`
          absolute top-0 right-0 w-32 h-32 rounded-full blur-3xl opacity-20
          ${config.bg.replace('/5', '/30')}
          -translate-y-1/2 translate-x-1/2
          group-hover:opacity-30 transition-opacity
        `}
      />

      <div className="relative">
        {/* Header with icon */}
        <div className="flex items-center justify-between mb-3">
          <p className="text-[11px] uppercase tracking-[0.15em] text-text-muted font-semibold">
            {title}
          </p>
          <div className={`${config.text} opacity-60`}>
            {displayIcon}
          </div>
        </div>

        {/* Value */}
        <p className={`text-3xl font-display font-bold tracking-tight ${config.text}`}>
          {value}
        </p>

        {/* Subtitle with status indicator */}
        <div className="flex items-center gap-2 mt-2">
          <span className={`w-1.5 h-1.5 rounded-full ${config.indicator} animate-pulse`} />
          <p className="text-sm text-text-secondary">{subtitle}</p>
        </div>
      </div>
    </div>
  )
}
