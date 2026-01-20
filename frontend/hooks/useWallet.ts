'use client'

import { useState, useEffect, useCallback } from 'react'
import { getApiBaseUrl } from '@/config/api-config'

interface WalletBalances {
  pol: number
  usdc_e: number
}

interface WalletStatus {
  exists: boolean
  address: string | null
  unlocked: boolean
  balances: WalletBalances | null
  approvals_set: boolean
}

interface UseWalletReturn {
  status: WalletStatus | null
  loading: boolean
  error: string | null
  refresh: () => Promise<void>
  generate: (password: string) => Promise<string>
  importKey: (privateKey: string, password: string) => Promise<string>
  unlock: (password: string) => Promise<void>
  lock: () => Promise<void>
  approveContracts: () => Promise<void>
}

export function useWallet(): UseWalletReturn {
  const [status, setStatus] = useState<WalletStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const apiBase = getApiBaseUrl()

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/wallet/status`)
      if (!res.ok) throw new Error('Failed to fetch wallet status')
      const data = await res.json()
      setStatus(data)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [apiBase])

  useEffect(() => {
    refresh()
    const interval = setInterval(refresh, 30000) // Refresh every 30s
    return () => clearInterval(interval)
  }, [refresh])

  const generate = useCallback(async (password: string): Promise<string> => {
    const res = await fetch(`${apiBase}/wallet/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password }),
    })
    if (!res.ok) {
      const err = await res.json()
      throw new Error(err.detail || 'Failed to generate wallet')
    }
    const data = await res.json()
    await refresh()
    return data.address
  }, [apiBase, refresh])

  const importKey = useCallback(async (privateKey: string, password: string): Promise<string> => {
    const res = await fetch(`${apiBase}/wallet/import`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ private_key: privateKey, password }),
    })
    if (!res.ok) {
      const err = await res.json()
      throw new Error(err.detail || 'Failed to import wallet')
    }
    const data = await res.json()
    await refresh()
    return data.address
  }, [apiBase, refresh])

  const unlock = useCallback(async (password: string): Promise<void> => {
    const res = await fetch(`${apiBase}/wallet/unlock`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password }),
    })
    if (!res.ok) {
      const err = await res.json()
      throw new Error(err.detail || 'Invalid password')
    }
    await refresh()
  }, [apiBase, refresh])

  const lock = useCallback(async (): Promise<void> => {
    await fetch(`${apiBase}/wallet/lock`, { method: 'POST' })
    await refresh()
  }, [apiBase, refresh])

  const approveContracts = useCallback(async (): Promise<void> => {
    const res = await fetch(`${apiBase}/wallet/approve-contracts`, {
      method: 'POST',
    })
    if (!res.ok) {
      const err = await res.json()
      throw new Error(err.detail || 'Failed to approve contracts')
    }
    await refresh()
  }, [apiBase, refresh])

  return {
    status,
    loading,
    error,
    refresh,
    generate,
    importKey,
    unlock,
    lock,
    approveContracts,
  }
}
