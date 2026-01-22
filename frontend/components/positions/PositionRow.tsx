'use client'

import { useState } from 'react'
import { getApiBaseUrl } from '@/config/api-config'
import type { Position } from '@/app/positions/page'

interface PositionRowProps {
  position: Position
  onRefresh: () => void
}

// Format relative time
function formatRelativeTime(isoString: string): string {
  const date = new Date(isoString)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMins / 60)
  const diffDays = Math.floor(diffHours / 24)

  if (diffMins < 1) return 'just now'
  if (diffMins < 60) return `${diffMins}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays < 7) return `${diffDays}d ago`
  return date.toLocaleDateString()
}

// Ensure TX hash has 0x prefix for block explorer links
function formatTxHash(hash: string): string {
  if (!hash) return ''
  return hash.startsWith('0x') ? hash : `0x${hash}`
}

// State badge colors and tooltips
const stateConfig: Record<string, { bg: string; text: string; label: string; tooltip: string }> = {
  active: { bg: 'bg-emerald/15', text: 'text-emerald', label: 'ACTIVE', tooltip: 'You own tokens on both sides - position is live' },
  pending: { bg: 'bg-amber/15', text: 'text-amber', label: 'PENDING', tooltip: 'Waiting for sell orders to complete - some USDC will be recovered' },
  partial: { bg: 'bg-cyan/15', text: 'text-cyan', label: 'PARTIAL', tooltip: 'You only have tokens on one side (the other was sold or redeemed)' },
  complete: { bg: 'bg-text-muted/15', text: 'text-text-muted', label: 'CLOSED', tooltip: 'All tokens sold or redeemed - position complete' },
}

export function PositionRow({ position: p, onRefresh }: PositionRowProps) {
  const [expanded, setExpanded] = useState(false)
  const [deleting, setDeleting] = useState(false)

  const state = stateConfig[p.state] || stateConfig.active

  const handleDelete = async () => {
    if (!confirm('Remove this from your list?\n\nThis only hides it from your dashboard. Your tokens and money are NOT affected.')) {
      return
    }

    setDeleting(true)
    try {
      await fetch(`${getApiBaseUrl()}/positions/${p.position_id}`, {
        method: 'DELETE',
      })
      onRefresh()
    } catch (e) {
      console.error('Failed to delete:', e)
    } finally {
      setDeleting(false)
    }
  }

  return (
    <div className="border border-border rounded-lg bg-surface overflow-hidden hover:border-border-glow transition-colors">
      {/* Compact Row */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3 flex items-center gap-4 text-left"
      >
        {/* State badge */}
        <div
          className={`px-2 py-0.5 rounded text-[10px] font-mono font-medium ${state.bg} ${state.text}`}
          title={state.tooltip}
        >
          {state.label}
        </div>

        {/* Pair summary */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
              p.target_position === 'YES' ? 'bg-emerald/15 text-emerald' : 'bg-rose/15 text-rose'
            }`}>
              {p.target_position}
            </span>
            <span className="text-sm text-text-primary truncate">
              {p.target_question.slice(0, 50)}{p.target_question.length > 50 ? '...' : ''}
            </span>
          </div>
          <div className="flex items-center gap-2 mt-0.5">
            <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
              p.cover_position === 'YES' ? 'bg-emerald/15 text-emerald' : 'bg-rose/15 text-rose'
            }`}>
              {p.cover_position}
            </span>
            <span className="text-xs text-text-muted truncate">
              {p.cover_question.slice(0, 50)}{p.cover_question.length > 50 ? '...' : ''}
            </span>
          </div>
        </div>

        {/* Tokens held */}
        <div className="text-right hidden sm:block" title="Tokens you own: target position / cover position">
          <div className="text-xs text-text-muted">Tokens</div>
          <div className="text-sm font-mono text-text-secondary">
            {p.target_balance.toFixed(2)} / {p.cover_balance.toFixed(2)}
          </div>
        </div>

        {/* Entry cost */}
        <div className="text-right hidden md:block" title="What you actually paid (initial spend minus recovered from selling unwanted tokens)">
          <div className="text-xs text-text-muted">Entry Cost</div>
          <div className="text-sm font-mono text-text-secondary">
            ${(p.entry_net_cost ?? p.entry_total_cost).toFixed(2)}
          </div>
        </div>

        {/* Value */}
        <div className="text-right hidden lg:block" title="What your tokens are worth right now if you sold them">
          <div className="text-xs text-text-muted">Value</div>
          <div className="text-sm font-mono text-text-primary">
            ${p.current_value.toFixed(2)}
          </div>
        </div>

        {/* P&L */}
        <div className="text-right min-w-[80px]" title="Your profit or loss (current value minus what you paid)">
          <div className={`text-sm font-mono font-semibold ${p.pnl >= 0 ? 'text-emerald' : 'text-rose'}`}>
            {p.pnl >= 0 ? '+' : ''}${p.pnl.toFixed(2)}
          </div>
          <div className={`text-[10px] font-mono ${p.pnl >= 0 ? 'text-emerald/70' : 'text-rose/70'}`}>
            {p.pnl_pct >= 0 ? '+' : ''}{p.pnl_pct.toFixed(1)}%
          </div>
        </div>

        {/* Time */}
        <div className="text-right hidden sm:block min-w-[60px]" title={`Entry time: ${new Date(p.entry_time).toLocaleString()}`}>
          <div className="text-xs text-text-muted">
            {formatRelativeTime(p.entry_time)}
          </div>
        </div>

        {/* Expand indicator */}
        <svg
          className={`w-4 h-4 text-text-muted transition-transform flex-shrink-0 ${expanded ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Expanded Details */}
      {expanded && (
        <div className="border-t border-border px-4 py-4 space-y-3 bg-surface-elevated">
          {/* Target Position */}
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-semibold text-text-primary">Target</span>
                <span className="text-xs text-text-muted">
                  ${p.target_entry_price.toFixed(3)} → ${p.target_current_price.toFixed(3)}
                </span>
                <span
                className={`text-[10px] font-mono ${p.target_clob_filled ? 'text-emerald' : 'text-amber'}`}
                title={p.target_clob_filled ? 'Unwanted tokens have been sold - you got USDC back' : 'Sell order placed - waiting for a buyer'}
              >
                  {p.target_clob_filled ? 'RECOVERED' : 'SELLING...'}
                </span>
              </div>
              <div className="text-xs text-text-secondary">{p.target_question}</div>
            </div>
            <div className="flex items-center gap-2 text-xs shrink-0">
              <a
                href={`https://polymarket.com/event/${p.target_market_id}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-cyan hover:underline"
                title="See this market on Polymarket website"
                onClick={(e) => e.stopPropagation()}
              >
                View market ↗
              </a>
              <a
                href={`https://polygonscan.com/tx/${formatTxHash(p.target_split_tx)}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-cyan hover:underline"
                title="See the blockchain transaction for this purchase"
                onClick={(e) => e.stopPropagation()}
              >
                View TX ↗
              </a>
            </div>
          </div>

          {/* Cover Position */}
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-semibold text-cyan">Cover</span>
                <span className="text-xs text-text-muted">
                  ${p.cover_entry_price.toFixed(3)} → ${p.cover_current_price.toFixed(3)}
                </span>
                <span
                className={`text-[10px] font-mono ${p.cover_clob_filled ? 'text-emerald' : 'text-amber'}`}
                title={p.cover_clob_filled ? 'Unwanted tokens have been sold - you got USDC back' : 'Sell order placed - waiting for a buyer'}
              >
                  {p.cover_clob_filled ? 'RECOVERED' : 'SELLING...'}
                </span>
              </div>
              <div className="text-xs text-text-secondary">{p.cover_question}</div>
            </div>
            <div className="flex items-center gap-2 text-xs shrink-0">
              <a
                href={`https://polymarket.com/event/${p.cover_market_id}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-cyan hover:underline"
                title="See this market on Polymarket website"
                onClick={(e) => e.stopPropagation()}
              >
                View market ↗
              </a>
              <a
                href={`https://polygonscan.com/tx/${formatTxHash(p.cover_split_tx)}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-cyan hover:underline"
                title="See the blockchain transaction for this purchase"
                onClick={(e) => e.stopPropagation()}
              >
                View TX ↗
              </a>
            </div>
          </div>

          {/* Merge Info (shown for pending positions) */}
          {p.state === 'pending' && ((p.target_unwanted_balance ?? 0) > 0.01 || (p.cover_unwanted_balance ?? 0) > 0.01) && (
            <div className="bg-amber/10 border border-amber/30 rounded p-2 text-xs">
              <span className="text-amber font-medium">Merge available: </span>
              <span className="text-text-secondary">
                {((p.target_unwanted_balance ?? 0) + (p.cover_unwanted_balance ?? 0)).toFixed(2)} tokens → ${((p.target_unwanted_balance ?? 0) + (p.cover_unwanted_balance ?? 0)).toFixed(2)} USDC
              </span>
            </div>
          )}

          {/* Footer */}
          <div className="flex items-center justify-between pt-2 border-t border-border text-xs">
            <span className="text-text-muted" title="When you bought and how much USDC was initially spent (before selling unwanted tokens)">
              {new Date(p.entry_time).toLocaleString()} · Initial spend: ${p.entry_total_cost.toFixed(0)}
            </span>
            <button
              onClick={(e) => {
                e.stopPropagation()
                handleDelete()
              }}
              disabled={deleting}
              className="text-rose hover:underline disabled:opacity-50"
              title="Hide this from your dashboard (your tokens and money are NOT affected)"
            >
              {deleting ? 'Removing...' : 'Remove from list'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
