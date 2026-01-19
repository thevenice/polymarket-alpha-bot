'use client'

import { useRef, useCallback, useEffect, useState } from 'react'
import { Portfolio } from '@/hooks/usePortfolioPrices'
import { PriceChangeIndicator } from '@/components/PriceFlash'
import { FavoriteButton } from '@/components/terminal/QuickActions'
import { densityStyles, Density } from '@/components/terminal/DensityToggle'

// =============================================================================
// TYPES
// =============================================================================

interface PriceChange {
  direction: 'up' | 'down' | 'changed' | null
}

interface ColumnHint {
  title: string
  beginner: string
  pro: string
}

// =============================================================================
// COLUMN HINTS - Educational tooltips for each column
// =============================================================================

const COLUMN_HINTS: Record<string, ColumnHint> = {
  favorite: {
    title: 'Watchlist',
    beginner: 'Click the star to track strategies you want to monitor. Pinned items stay at the top.',
    pro: 'Persisted to localStorage. Sorted by coverage within pinned group.',
  },
  target: {
    title: 'Target Bet',
    beginner: 'The primary position you\'re betting on. This is the outcome you expect to win.',
    pro: 'Higher probability event in the covered pair. Price shown is current ask.',
  },
  backup: {
    title: 'Backup Bet',
    beginner: 'Your hedge position. If target loses, this one wins — guaranteeing you get $1 back.',
    pro: 'Covers the target\'s failure case. Combined cost < $1 = arbitrage opportunity.',
  },
  confidence: {
    title: 'LLM Confidence',
    beginner: 'AI-assessed probability that this strategy is logically valid (events truly cover all outcomes).',
    pro: 'Primary sort key. Higher = more likely the coverage logic is sound. 80%+ is high confidence.',
  },
  cost: {
    title: 'Total Cost',
    beginner: 'How much you need to invest to buy both positions. Lower is better.',
    pro: 'Sum of target + backup prices. Cost < $1.00 means guaranteed profit potential.',
  },
  return: {
    title: 'Expected Return',
    beginner: 'Your profit if the strategy works: (Payout - Cost) / Cost. Green = profit!',
    pro: 'Secondary sort key. Formula: (1 - total_cost) / total_cost × 100%.',
  },
}

// =============================================================================
// COLUMN HINT COMPONENT
// =============================================================================

function InfoIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="8" cy="8" r="6.5" />
      <path d="M8 7v4" strokeLinecap="round" />
      <circle cx="8" cy="5" r="0.5" fill="currentColor" stroke="none" />
    </svg>
  )
}

function ColumnHintIcon({
  hint,
  position = 'center'
}: {
  hint: ColumnHint
  position?: 'left' | 'center' | 'right'
}) {
  const positionClass = position === 'left'
    ? 'column-hint-tooltip--left'
    : position === 'right'
      ? 'column-hint-tooltip--right'
      : ''

  return (
    <span className="column-hint">
      <span className="column-hint-icon">
        <InfoIcon />
      </span>
      <span className={`column-hint-tooltip ${positionClass}`}>
        <span className="column-hint-title">{hint.title}</span>
        <span className="column-hint-text">{hint.beginner}</span>
        <span className="column-hint-pro">{hint.pro}</span>
      </span>
    </span>
  )
}

interface PortfolioTableProps {
  portfolios: Portfolio[]
  density: Density
  selectedIndex: number
  changedIds: Set<string>
  priceChanges: Map<string, PriceChange>
  pinnedCount: number
  connected: boolean
  isFavorite: (pairId: string) => boolean
  onSelect: (index: number, portfolio: Portfolio) => void
  onToggleFavorite: (pairId: string, coverage: number) => void
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function PortfolioTable({
  portfolios,
  density,
  selectedIndex,
  changedIds,
  priceChanges,
  pinnedCount,
  connected,
  isFavorite,
  onSelect,
  onToggleFavorite,
}: PortfolioTableProps) {
  const styles = densityStyles[density]
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const rowRefs = useRef<Map<number, HTMLTableRowElement>>(new Map())
  const [scrollState, setScrollState] = useState({ atTop: true, atBottom: true })

  // Track scroll position for shadow indicators
  const handleScroll = useCallback(() => {
    const container = scrollContainerRef.current
    if (!container) return

    const { scrollTop, scrollHeight, clientHeight } = container
    const atTop = scrollTop < 10
    const atBottom = scrollTop + clientHeight >= scrollHeight - 10

    setScrollState((prev) => {
      if (prev.atTop !== atTop || prev.atBottom !== atBottom) {
        return { atTop, atBottom }
      }
      return prev
    })
  }, [])

  // Check scroll state on mount and when portfolios change
  useEffect(() => {
    handleScroll()
  }, [portfolios.length, handleScroll])

  // Auto-scroll to selected row
  useEffect(() => {
    if (selectedIndex >= 0) {
      const row = rowRefs.current.get(selectedIndex)
      if (row) {
        row.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
      }
    }
  }, [selectedIndex])

  return (
    <div className="flex flex-col flex-1 min-h-0 rounded-lg border border-border overflow-hidden bg-surface">
      {/* Scroll container with shadow indicators */}
      <div className="relative flex-1 min-h-0">
        {/* Top scroll shadow */}
        <div
          className={`absolute top-0 left-0 right-0 h-6 bg-gradient-to-b from-surface/90 to-transparent z-20 pointer-events-none transition-opacity duration-150 ${
            scrollState.atTop ? 'opacity-0' : 'opacity-100'
          }`}
          aria-hidden="true"
        />

        {/* Bottom scroll shadow */}
        <div
          className={`absolute bottom-0 left-0 right-0 h-6 bg-gradient-to-t from-surface/90 to-transparent z-20 pointer-events-none transition-opacity duration-150 ${
            scrollState.atBottom ? 'opacity-0' : 'opacity-100'
          }`}
          aria-hidden="true"
        />

        <div
          ref={scrollContainerRef}
          className="h-full overflow-y-auto overflow-x-auto"
          onScroll={handleScroll}
        >
          <table className="w-full table-fixed">
            <thead className="bg-surface-elevated border-b border-border sticky top-0 z-10">
              <tr>
                <th className={`${styles.headerPadding} text-left text-[10px] font-medium uppercase tracking-wider text-text-muted w-10`}>
                  <span className="flex items-center gap-1">
                    ★
                    <ColumnHintIcon hint={COLUMN_HINTS.favorite} position="left" />
                  </span>
                </th>
                <th className={`${styles.headerPadding} text-left text-[10px] font-medium uppercase tracking-wider text-text-muted w-[28%]`}>
                  <span className="flex items-center gap-1">
                    Target Bet
                    <ColumnHintIcon hint={COLUMN_HINTS.target} position="left" />
                  </span>
                </th>
                <th className={`${styles.headerPadding} text-left text-[10px] font-medium uppercase tracking-wider text-text-muted w-[28%]`}>
                  <span className="flex items-center gap-1">
                    Backup Bet
                    <ColumnHintIcon hint={COLUMN_HINTS.backup} />
                  </span>
                </th>
                <th className={`${styles.headerPadding} text-left text-[10px] font-medium uppercase tracking-wider text-text-muted w-20`}>
                  <span className="flex items-center gap-1">
                    LLM Conf.
                    <ColumnHintIcon hint={COLUMN_HINTS.confidence} />
                  </span>
                </th>
                <th className={`${styles.headerPadding} text-left text-[10px] font-medium uppercase tracking-wider text-text-muted w-16`}>
                  <span className="flex items-center gap-1">
                    Cost
                    <ColumnHintIcon hint={COLUMN_HINTS.cost} />
                  </span>
                </th>
                <th className={`${styles.headerPadding} text-left text-[10px] font-medium uppercase tracking-wider text-text-muted w-16`}>
                  <span className="flex items-center gap-1">
                    Return
                    <ColumnHintIcon hint={COLUMN_HINTS.return} position="right" />
                  </span>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {portfolios.map((p, index) => {
                const isProfitable = p.expected_profit > 0.001
                const viabilityPercent = p.viability_score !== undefined ? (p.viability_score * 100).toFixed(0) : null
                const isChanged = changedIds.has(p.pair_id)
                const priceChange = priceChanges.get(p.pair_id)
                const isSelected = index === selectedIndex
                const isPinned = isFavorite(p.pair_id)

                const flashClass = isChanged
                  ? priceChange?.direction === 'up'
                    ? 'animate-flash-up'
                    : priceChange?.direction === 'down'
                      ? 'animate-flash-down'
                      : 'animate-flash'
                  : ''

                return (
                  <tr
                    key={p.pair_id}
                    ref={(el) => {
                      if (el) rowRefs.current.set(index, el)
                    }}
                    className={`
                      group transition-colors cursor-pointer ${flashClass}
                      ${isSelected
                        ? 'bg-cyan/10 ring-1 ring-inset ring-cyan/50'
                        : isPinned
                          ? 'bg-amber/5 hover:bg-amber/10'
                          : 'hover:bg-surface-hover'
                      }
                    `}
                    onClick={() => onSelect(index, p)}
                  >
                    <td className={styles.cellPadding}>
                      <FavoriteButton
                        isFavorite={isPinned}
                        onToggle={() => onToggleFavorite(p.pair_id, p.coverage)}
                      />
                    </td>
                    <td className={styles.cellPadding}>
                      <div className="space-y-0.5">
                        <p
                          className={`${styles.fontSize} text-text-primary truncate`}
                          title={p.target_question}
                        >
                          {p.target_question}
                        </p>
                        <div className="flex items-center gap-1">
                          <p className="text-[10px] text-text-muted">
                            {p.target_position} @ ${p.target_price.toFixed(2)}
                          </p>
                          {isChanged && priceChange && (
                            <PriceChangeIndicator direction={priceChange.direction === 'changed' ? null : priceChange.direction} />
                          )}
                        </div>
                      </div>
                    </td>
                    <td className={styles.cellPadding}>
                      <div className="space-y-0.5">
                        <p
                          className={`${styles.fontSize} text-text-primary truncate`}
                          title={p.cover_question}
                        >
                          {p.cover_question}
                        </p>
                        <div className="flex items-center gap-1">
                          <p className="text-[10px] text-text-muted">
                            {p.cover_position} @ ${p.cover_price.toFixed(2)}
                          </p>
                          {isChanged && priceChange && (
                            <PriceChangeIndicator direction={priceChange.direction === 'changed' ? null : priceChange.direction} />
                          )}
                        </div>
                      </div>
                    </td>
                    <td className={styles.cellPadding}>
                      <div className="space-y-1">
                        <span
                          className={`${styles.fontSize} font-mono ${p.viability_score !== undefined ? (p.viability_score >= 0.8 ? 'text-emerald' : p.viability_score >= 0.6 ? 'text-cyan' : 'text-text-secondary') : 'text-text-muted'}`}
                        >
                          {viabilityPercent !== null ? `${viabilityPercent}%` : '—'}
                        </span>
                        {p.viability_score !== undefined && (
                          <div className="w-16 h-1 bg-surface-elevated rounded-full overflow-hidden">
                            <div
                              className={`h-full transition-all duration-500 ${p.viability_score >= 0.8 ? 'bg-emerald' : p.viability_score >= 0.6 ? 'bg-cyan' : 'bg-amber'}`}
                              style={{ width: `${Math.min(100, p.viability_score * 100)}%` }}
                            />
                          </div>
                        )}
                      </div>
                    </td>
                    <td className={styles.cellPadding}>
                      <span className={`${styles.fontSize} font-mono text-text-secondary`}>
                        ${p.total_cost.toFixed(2)}
                      </span>
                    </td>
                    <td className={styles.cellPadding}>
                      <span
                        className={`${styles.fontSize} font-mono font-medium ${isProfitable ? 'text-emerald' : 'text-rose'}`}
                      >
                        {isProfitable ? '+' : ''}
                        {(p.expected_profit * 100).toFixed(1)}%
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Footer */}
      <div className="px-2.5 py-2 bg-surface-elevated border-t border-border flex items-center justify-between shrink-0">
        <span className="text-[10px] text-text-muted">
          Showing {portfolios.length} strategies
          {pinnedCount > 0 && (
            <span className="ml-2 text-amber">★ {pinnedCount} pinned</span>
          )}
          {connected && <span className="ml-2 text-emerald">● Live prices</span>}
          {selectedIndex >= 0 && (
            <span className="ml-2 text-cyan">• Row {selectedIndex + 1} selected</span>
          )}
        </span>
        <span className="text-[10px] text-text-muted">
          Press <kbd className="px-1 py-0.5 bg-surface border border-border rounded text-[9px]">?</kbd> for shortcuts
        </span>
      </div>
    </div>
  )
}

// Re-export types for consumers
export type { PriceChange }
