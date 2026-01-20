// frontend/app/settings/page.tsx
'use client'

import { useState } from 'react'
import { useWallet } from '@/hooks/useWallet'

export default function SettingsPage() {
  const { status, loading, error, generate, importKey, unlock, lock, approveContracts, refresh } = useWallet()

  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [privateKey, setPrivateKey] = useState('')
  const [mode, setMode] = useState<'generate' | 'import' | null>(null)
  const [actionLoading, setActionLoading] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  const handleGenerate = async () => {
    if (password !== confirmPassword) {
      setActionError('Passwords do not match')
      return
    }
    if (password.length < 8) {
      setActionError('Password must be at least 8 characters')
      return
    }

    setActionLoading(true)
    setActionError(null)
    try {
      await generate(password)
      setPassword('')
      setConfirmPassword('')
      setMode(null)
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Failed to generate wallet')
    } finally {
      setActionLoading(false)
    }
  }

  const handleImport = async () => {
    if (password.length < 8) {
      setActionError('Password must be at least 8 characters')
      return
    }
    if (!privateKey.startsWith('0x') || privateKey.length !== 66) {
      setActionError('Invalid private key format')
      return
    }

    setActionLoading(true)
    setActionError(null)
    try {
      await importKey(privateKey, password)
      setPassword('')
      setPrivateKey('')
      setMode(null)
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Failed to import wallet')
    } finally {
      setActionLoading(false)
    }
  }

  const handleUnlock = async () => {
    setActionLoading(true)
    setActionError(null)
    try {
      await unlock(password)
      setPassword('')
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Invalid password')
    } finally {
      setActionLoading(false)
    }
  }

  const handleApprove = async () => {
    setActionLoading(true)
    setActionError(null)
    try {
      await approveContracts()
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Failed to approve contracts')
    } finally {
      setActionLoading(false)
    }
  }

  const copyAddress = () => {
    if (status?.address) {
      navigator.clipboard.writeText(status.address)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-void p-6">
        <div className="max-w-xl mx-auto">
          <h1 className="text-2xl font-bold text-text-primary mb-6">Settings</h1>
          <div className="text-text-muted">Loading...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-void p-6">
      <div className="max-w-xl mx-auto">
        <h1 className="text-2xl font-bold text-text-primary mb-6">Settings</h1>

        {/* Wallet Section */}
        <div className="bg-surface border border-border rounded-xl p-5">
          <h2 className="text-lg font-semibold text-text-primary mb-4">Wallet</h2>

          {error && (
            <div className="mb-4 p-3 bg-rose/10 border border-rose/25 rounded-lg text-rose text-sm">
              {error}
            </div>
          )}

          {actionError && (
            <div className="mb-4 p-3 bg-rose/10 border border-rose/25 rounded-lg text-rose text-sm">
              {actionError}
            </div>
          )}

          {!status?.exists ? (
            // No wallet - show setup options
            <div className="space-y-4">
              {!mode && (
                <div className="flex gap-3">
                  <button
                    onClick={() => setMode('generate')}
                    className="flex-1 py-2.5 px-4 bg-emerald/10 hover:bg-emerald/15 border border-emerald/25 rounded-lg text-emerald text-sm font-medium transition-colors"
                  >
                    Generate New Wallet
                  </button>
                  <button
                    onClick={() => setMode('import')}
                    className="flex-1 py-2.5 px-4 bg-cyan/10 hover:bg-cyan/15 border border-cyan/25 rounded-lg text-cyan text-sm font-medium transition-colors"
                  >
                    Import Existing
                  </button>
                </div>
              )}

              {mode === 'generate' && (
                <div className="space-y-3">
                  <input
                    type="password"
                    placeholder="Password (min 8 characters)"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full px-3 py-2 bg-surface-elevated border border-border rounded-lg text-text-primary placeholder-text-muted text-sm focus:outline-none focus:border-cyan"
                  />
                  <input
                    type="password"
                    placeholder="Confirm password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="w-full px-3 py-2 bg-surface-elevated border border-border rounded-lg text-text-primary placeholder-text-muted text-sm focus:outline-none focus:border-cyan"
                  />
                  <div className="flex gap-2">
                    <button
                      onClick={() => { setMode(null); setPassword(''); setConfirmPassword(''); setActionError(null); }}
                      className="px-4 py-2 text-text-muted hover:text-text-primary text-sm"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleGenerate}
                      disabled={actionLoading}
                      className="flex-1 py-2 px-4 bg-emerald hover:bg-emerald/90 rounded-lg text-void text-sm font-medium transition-colors disabled:opacity-50"
                    >
                      {actionLoading ? 'Creating...' : 'Create Wallet'}
                    </button>
                  </div>
                </div>
              )}

              {mode === 'import' && (
                <div className="space-y-3">
                  <input
                    type="password"
                    placeholder="Private key (0x...)"
                    value={privateKey}
                    onChange={(e) => setPrivateKey(e.target.value)}
                    className="w-full px-3 py-2 bg-surface-elevated border border-border rounded-lg text-text-primary placeholder-text-muted text-sm font-mono focus:outline-none focus:border-cyan"
                  />
                  <input
                    type="password"
                    placeholder="Encryption password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full px-3 py-2 bg-surface-elevated border border-border rounded-lg text-text-primary placeholder-text-muted text-sm focus:outline-none focus:border-cyan"
                  />
                  <div className="flex gap-2">
                    <button
                      onClick={() => { setMode(null); setPassword(''); setPrivateKey(''); setActionError(null); }}
                      className="px-4 py-2 text-text-muted hover:text-text-primary text-sm"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleImport}
                      disabled={actionLoading}
                      className="flex-1 py-2 px-4 bg-cyan hover:bg-cyan/90 rounded-lg text-void text-sm font-medium transition-colors disabled:opacity-50"
                    >
                      {actionLoading ? 'Importing...' : 'Import Wallet'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            // Wallet exists - show status
            <div className="space-y-4">
              {/* Status indicator */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${status.unlocked ? 'bg-emerald' : 'bg-amber'}`} />
                  <span className="text-sm text-text-muted">
                    {status.unlocked ? 'Unlocked' : 'Locked'}
                  </span>
                </div>
                {status.unlocked && (
                  <button
                    onClick={lock}
                    className="text-sm text-text-muted hover:text-rose transition-colors"
                  >
                    Lock Wallet
                  </button>
                )}
              </div>

              {/* Address */}
              <div className="flex items-center gap-2">
                <span className="text-sm text-text-muted">Address:</span>
                <code className="text-sm text-text-primary font-mono">
                  {status.address?.slice(0, 10)}...{status.address?.slice(-8)}
                </code>
                <button
                  onClick={copyAddress}
                  className="text-text-muted hover:text-cyan transition-colors"
                >
                  {copied ? '✓' : '📋'}
                </button>
              </div>

              {/* Balances */}
              {status.balances && (
                <div className="bg-surface-elevated rounded-lg p-3 space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-text-muted">POL:</span>
                    <span className="text-text-primary font-mono">{status.balances.pol.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-muted">USDC.e:</span>
                    <span className="text-text-primary font-mono">${status.balances.usdc_e.toFixed(2)}</span>
                  </div>
                </div>
              )}

              {/* Approvals */}
              <div className="flex items-center gap-2">
                <span className="text-sm text-text-muted">Contract Approvals:</span>
                <span className={`text-sm ${status.approvals_set ? 'text-emerald' : 'text-amber'}`}>
                  {status.approvals_set ? '✓ Set' : '⚠ Not Set'}
                </span>
              </div>

              {/* Unlock form */}
              {!status.unlocked && (
                <div className="flex gap-2 pt-2">
                  <input
                    type="password"
                    placeholder="Password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleUnlock()}
                    className="flex-1 px-3 py-2 bg-surface-elevated border border-border rounded-lg text-text-primary placeholder-text-muted text-sm focus:outline-none focus:border-cyan"
                  />
                  <button
                    onClick={handleUnlock}
                    disabled={actionLoading}
                    className="px-4 py-2 bg-cyan hover:bg-cyan/90 rounded-lg text-void text-sm font-medium transition-colors disabled:opacity-50"
                  >
                    {actionLoading ? '...' : 'Unlock'}
                  </button>
                </div>
              )}

              {/* Approve button */}
              {status.unlocked && !status.approvals_set && (
                <button
                  onClick={handleApprove}
                  disabled={actionLoading}
                  className="w-full py-2.5 px-4 bg-amber/10 hover:bg-amber/15 border border-amber/25 rounded-lg text-amber text-sm font-medium transition-colors disabled:opacity-50"
                >
                  {actionLoading ? 'Approving...' : 'Set Contract Approvals'}
                </button>
              )}
            </div>
          )}
        </div>

        {/* Refresh button */}
        <div className="mt-4 text-center">
          <button
            onClick={refresh}
            className="text-sm text-text-muted hover:text-text-primary transition-colors"
          >
            ↻ Refresh
          </button>
        </div>
      </div>
    </div>
  )
}
