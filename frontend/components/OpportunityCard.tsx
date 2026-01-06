interface Opportunity {
  id: string
  rank: number
  trigger: {
    event_id: string
    title: string
    price: number
    price_display: string
  }
  consequence: {
    event_id: string
    title: string
    price: number
    price_display: string
  }
  relation: {
    type: string
    type_display: string
    confidence: number
  }
  alpha: {
    signal: number
    signal_display: string
    direction: string
  }
}

interface OpportunityCardProps {
  opportunity: Opportunity
  currentPrice?: number
}

export function OpportunityCard({ opportunity, currentPrice }: OpportunityCardProps) {
  const { trigger, consequence, relation, alpha } = opportunity

  // Calculate price change if we have current price
  const priceChange = currentPrice !== undefined
    ? ((currentPrice - consequence.price) / consequence.price) * 100
    : null

  const alphaValue = parseFloat(alpha.signal_display.replace(/[+%]/g, ''))
  const isHighAlpha = alphaValue > 20
  const isBuy = alpha.direction === 'BUY'

  return (
    <div
      className={`
        relative overflow-hidden rounded-xl border p-5
        bg-surface transition-all duration-300
        ${isHighAlpha
          ? isBuy
            ? 'border-alpha-buy/30 shadow-glow-alpha-buy'
            : 'border-alpha-sell/30 shadow-glow-alpha-sell'
          : 'border-border hover:border-cyan/30'
        }
        group hover:scale-[1.01]
      `}
    >
      {/* Background glow for high alpha */}
      {isHighAlpha && (
        <div
          className={`
            absolute inset-0 opacity-[0.03]
            ${isBuy ? 'bg-alpha-buy' : 'bg-alpha-sell'}
          `}
        />
      )}

      {/* Header */}
      <div className="relative flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-text-muted bg-surface-elevated px-2 py-1 rounded">
            #{opportunity.rank}
          </span>
          <span className="text-[10px] uppercase tracking-wider text-text-muted">
            {opportunity.id.slice(0, 8)}
          </span>
        </div>
        <span
          className={`
            px-3 py-1.5 rounded-lg text-xs font-bold tracking-wide
            ${isBuy
              ? 'bg-alpha-buy/10 text-alpha-buy border border-alpha-buy/20'
              : 'bg-alpha-sell/10 text-alpha-sell border border-alpha-sell/20'
            }
          `}
        >
          {alpha.direction} {alpha.signal_display}
        </span>
      </div>

      {/* Trigger Event */}
      <div className="relative mb-4">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-[10px] font-bold uppercase tracking-[0.15em] text-cyan">
            IF
          </span>
          <div className="flex-1 h-px bg-gradient-to-r from-cyan/30 to-transparent" />
        </div>
        <p className="text-sm font-medium text-text-primary line-clamp-2 leading-relaxed" title={trigger.title}>
          {trigger.title}
        </p>
        <div className="flex items-center gap-2 mt-2">
          <span className="text-xs text-text-muted">Price:</span>
          <span className="text-sm font-mono font-semibold text-text-secondary">
            {trigger.price_display}
          </span>
        </div>
      </div>

      {/* Relation Connector */}
      <div className="flex items-center gap-3 my-4">
        <div className="flex-1 h-px bg-border" />
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 13.5L12 21m0 0l-7.5-7.5M12 21V3" />
          </svg>
          <span className="px-3 py-1.5 text-xs font-semibold uppercase tracking-wider bg-surface-elevated text-text-secondary rounded-lg border border-border">
            {relation.type_display || relation.type}
          </span>
          <svg className="w-4 h-4 text-text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 13.5L12 21m0 0l-7.5-7.5M12 21V3" />
          </svg>
        </div>
        <div className="flex-1 h-px bg-border" />
      </div>

      {/* Consequence Event */}
      <div className="relative">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-[10px] font-bold uppercase tracking-[0.15em] text-amber">
            THEN
          </span>
          <div className="flex-1 h-px bg-gradient-to-r from-amber/30 to-transparent" />
        </div>
        <p className="text-sm font-medium text-text-primary line-clamp-2 leading-relaxed" title={consequence.title}>
          {consequence.title}
        </p>
        <div className="flex items-center gap-3 mt-2">
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-muted">Price:</span>
            <span className="text-sm font-mono font-semibold text-text-secondary">
              {consequence.price_display}
            </span>
          </div>
          {priceChange !== null && (
            <span
              className={`
                text-xs font-mono font-semibold px-2 py-0.5 rounded
                ${priceChange > 0
                  ? 'text-alpha-buy bg-alpha-buy/10'
                  : priceChange < 0
                    ? 'text-alpha-sell bg-alpha-sell/10'
                    : 'text-text-muted bg-surface-elevated'
                }
              `}
            >
              {priceChange > 0 ? '+' : ''}{priceChange.toFixed(1)}%
            </span>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="relative flex items-center justify-between mt-4 pt-4 border-t border-border/50">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] uppercase tracking-wider text-text-muted">Conf</span>
            <div className="flex items-center gap-1">
              {/* Confidence bar */}
              <div className="w-16 h-1.5 bg-surface-elevated rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-cyan to-cyan-dim rounded-full"
                  style={{ width: `${relation.confidence * 100}%` }}
                />
              </div>
              <span className="text-xs font-mono text-text-secondary">
                {(relation.confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
        <button
          onClick={() => window.open(`https://polymarket.com/event/${consequence.event_id}`, '_blank')}
          className="
            flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold
            bg-cyan/10 text-cyan border border-cyan/20 rounded-lg
            hover:bg-cyan/20 hover:border-cyan/30 transition-all
          "
        >
          <span>View</span>
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 6H5.25A2.25 2.25 0 003 8.25v10.5A2.25 2.25 0 005.25 21h10.5A2.25 2.25 0 0018 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25" />
          </svg>
        </button>
      </div>
    </div>
  )
}
