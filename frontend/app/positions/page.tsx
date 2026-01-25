'use client'

import { useEffect, useState, useCallback } from 'react'
import { getApiBaseUrl } from '@/config/api-config'
import { PositionsTable } from '@/components/positions/PositionsTable'
import { StatusIndicators } from '@/components/StatusIndicators'

// =============================================================================
// TYPES
// =============================================================================

export interface Position {
  position_id: string
  pair_id: string
  entry_time: string
  entry_amount_per_side: number
  entry_total_cost: number

  target_market_id: string
  target_position: 'YES' | 'NO'
  target_token_id: string
  target_question: string
  target_entry_price: number
  target_group_slug: string
  target_split_tx: string
  target_clob_order_id: string | null
  target_clob_filled: boolean

  cover_market_id: string
  cover_position: 'YES' | 'NO'
  cover_token_id: string
  cover_question: string
  cover_entry_price: number
  cover_group_slug: string
  cover_split_tx: string
  cover_clob_order_id: string | null
  cover_clob_filled: boolean

  notes: string | null

  target_balance: number
  cover_balance: number
  target_current_price: number
  cover_current_price: number
  target_unwanted_balance: number
  cover_unwanted_balance: number

  state: 'active' | 'pending' | 'partial' | 'complete'
  entry_net_cost: number
  current_value: number
  pnl: number
  pnl_pct: number

  // Selling in progress flags (persisted across page refresh)
  selling_target: boolean
  selling_cover: boolean
}

interface PositionsResponse {
  count: number
  active_count: number
  total_pnl: number
  positions: Position[]
}

type FilterState = 'all' | 'active' | 'pending' | 'complete'

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export default function PositionsPage() {
  const [positions, setPositions] = useState<Position[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<FilterState>('all')
  const [stats, setStats] = useState({ count: 0, active_count: 0, total_pnl: 0 })

  const fetchPositions = useCallback(async () => {
    try {
      setError(null)
      const res = await fetch(`${getApiBaseUrl()}/positions`)
      if (!res.ok) throw new Error('Failed to fetch positions')
      const data: PositionsResponse = await res.json()
      setPositions(data.positions || [])
      setStats({
        count: data.count,
        active_count: data.active_count,
        total_pnl: data.total_pnl,
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch positions')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchPositions()
    const interval = setInterval(fetchPositions, 5000)
    return () => clearInterval(interval)
  }, [fetchPositions])

  // Filter and sort
  const filtered = positions.filter((p) => {
    if (filter === 'all') return true
    if (filter === 'active') return p.state === 'active' || p.state === 'pending' || p.state === 'partial'
    if (filter === 'pending') return p.state === 'pending'
    if (filter === 'complete') return p.state === 'complete'
    return true
  })

  const sorted = [...filtered].sort((a, b) =>
    new Date(b.entry_time).getTime() - new Date(a.entry_time).getTime()
  )

  // Count issues
  const issueCount = positions.filter(p =>
    (!p.target_clob_filled && !p.target_clob_order_id && p.target_unwanted_balance > 0.01) ||
    (!p.cover_clob_filled && !p.cover_clob_order_id && p.cover_unwanted_balance > 0.01)
  ).length

  return (
    <div className="flex flex-col h-full gap-4 animate-fade-in">
      {/* Header */}
      <header className="bg-surface border border-border rounded-lg p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div>
              <h1 className="text-lg font-semibold text-text-primary">Positions</h1>
              <p className="text-[10px] text-text-muted">
                Manage your holdings
              </p>
            </div>

            <div className="w-px h-10 bg-border" />

            <div className="flex items-center gap-2">
              <span className="text-2xl font-semibold font-mono text-cyan">{stats.count}</span>
              <div className="text-xs text-text-muted leading-tight">
                <p>positions</p>
                <p className="text-text-muted/70">{stats.active_count} active</p>
              </div>
            </div>

            {issueCount > 0 && (
              <>
                <div className="w-px h-10 bg-border" />
                <div className="flex items-center gap-2">
                  <span className="text-2xl font-semibold font-mono text-rose">{issueCount}</span>
                  <div className="text-xs text-rose leading-tight">
                    <p>need</p>
                    <p>attention</p>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Right: Status indicators */}
          <StatusIndicators />
        </div>
      </header>

      {/* Filters */}
      <div className="flex items-center gap-4">
        <div className="flex gap-1">
          {(['all', 'active', 'pending', 'complete'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1.5 rounded text-sm transition-colors ${
                filter === f
                  ? 'bg-surface-elevated text-text-primary border border-cyan'
                  : 'text-text-secondary hover:text-text-primary border border-transparent'
              }`}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
        <div className="flex-1" />
        <span className="text-xs text-text-muted">
          {sorted.length} position{sorted.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Content */}
      {loading ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="flex items-center gap-3">
            <div className="w-5 h-5 border-2 border-cyan border-t-transparent rounded-full animate-spin" />
            <span className="text-sm text-text-muted">Loading positions...</span>
          </div>
        </div>
      ) : error ? (
        <div className="flex-1 flex flex-col items-center justify-center border border-border rounded-lg bg-surface">
          <p className="text-sm text-rose mb-2">{error}</p>
          <button onClick={fetchPositions} className="text-sm text-cyan hover:underline">
            Try again
          </button>
        </div>
      ) : positions.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center border border-border rounded-lg bg-surface">
          <p className="text-sm text-text-secondary mb-1">No positions yet</p>
          <p className="text-xs text-text-muted">Buy a pair from Terminal to start</p>
        </div>
      ) : sorted.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center border border-border rounded-lg bg-surface">
          <p className="text-sm text-text-secondary mb-1">No positions match filter</p>
          <button onClick={() => setFilter('all')} className="text-xs text-cyan hover:underline">
            Show all
          </button>
        </div>
      ) : (
        <PositionsTable positions={sorted} onRefresh={fetchPositions} />
      )}
    </div>
  )
}
