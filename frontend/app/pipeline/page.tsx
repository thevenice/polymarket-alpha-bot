'use client'

import { useEffect, useState } from 'react'

interface PipelineStep {
  step: string
  name: string
  description: string
  latest_run: string | null
  has_data: boolean
}

interface PipelineStatus {
  timestamp: string
  running: boolean
  current_step: string | null
  steps: PipelineStep[]
}

export default function PipelinePage() {
  const [status, setStatus] = useState<PipelineStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [runningPipeline, setRunningPipeline] = useState(false)

  async function fetchStatus() {
    try {
      const res = await fetch('http://localhost:8000/pipeline/status')
      if (res.ok) {
        setStatus(await res.json())
      }
    } catch (error) {
      console.error('Failed to fetch status:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 5000)
    return () => clearInterval(interval)
  }, [])

  async function runPipeline(fromStep: string = '01') {
    setRunningPipeline(true)
    try {
      const res = await fetch('http://localhost:8000/pipeline/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ from_step: fromStep, to_step: '06_3' }),
      })
      if (res.ok) {
        // Start polling for updates
        fetchStatus()
      }
    } catch (error) {
      console.error('Failed to run pipeline:', error)
    } finally {
      setRunningPipeline(false)
    }
  }

  const formatTimestamp = (ts: string) => {
    try {
      const date = new Date(
        ts.replace(
          /(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/,
          '$1-$2-$3T$4:$5:$6'
        )
      )
      return date.toLocaleString()
    } catch {
      return ts
    }
  }

  const completedSteps = status?.steps.filter(s => s.has_data).length || 0
  const totalSteps = status?.steps.length || 0
  const progressPercent = totalSteps > 0 ? (completedSteps / totalSteps) * 100 : 0

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="font-display text-4xl font-bold tracking-tight text-text-primary">
            Pipeline
          </h1>
          <p className="text-text-secondary mt-2">
            Manage and monitor the data pipeline
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => runPipeline('01')}
            disabled={runningPipeline || status?.running}
            className="btn-primary text-sm disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none"
          >
            {runningPipeline || status?.running ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Running...
              </span>
            ) : (
              <span className="flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
                </svg>
                Run Full Pipeline
              </span>
            )}
          </button>
          <button
            onClick={() => runPipeline('06_1')}
            disabled={runningPipeline || status?.running}
            className="btn-success text-sm disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none"
          >
            <span className="flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
              </svg>
              Quick Refresh
            </span>
          </button>
        </div>
      </div>

      {/* Progress Overview */}
      <div className="rounded-xl border border-border bg-surface p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="text-[11px] uppercase tracking-[0.15em] text-text-muted font-semibold mb-1">
              Pipeline Progress
            </p>
            <p className="text-2xl font-display font-bold text-text-primary">
              {completedSteps} / {totalSteps} <span className="text-lg text-text-secondary font-normal">steps complete</span>
            </p>
          </div>
          {status?.running && (
            <div className="flex items-center gap-3 px-4 py-2 rounded-lg bg-cyan/5 border border-cyan/20">
              <div className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan opacity-75" />
                <span className="relative inline-flex rounded-full h-3 w-3 bg-cyan" />
              </div>
              <span className="text-sm font-medium text-cyan">
                Running: {status.current_step}
              </span>
            </div>
          )}
        </div>
        {/* Progress bar */}
        <div className="h-2 bg-surface-elevated rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-cyan to-emerald rounded-full transition-all duration-500"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </div>

      {/* Steps Table */}
      {loading ? (
        <div className="flex items-center justify-center py-16">
          <div className="flex items-center gap-3 text-text-muted">
            <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <span>Loading pipeline status...</span>
          </div>
        </div>
      ) : (
        <div className="rounded-xl border border-border overflow-hidden bg-surface">
          <table className="w-full terminal-table">
            <thead className="bg-surface-elevated border-b border-border">
              <tr>
                <th className="px-4 py-3.5 text-left text-[10px] font-bold uppercase tracking-[0.15em] text-text-muted w-20">
                  Step
                </th>
                <th className="px-4 py-3.5 text-left text-[10px] font-bold uppercase tracking-[0.15em] text-text-muted w-40">
                  Name
                </th>
                <th className="px-4 py-3.5 text-left text-[10px] font-bold uppercase tracking-[0.15em] text-text-muted">
                  Description
                </th>
                <th className="px-4 py-3.5 text-left text-[10px] font-bold uppercase tracking-[0.15em] text-text-muted w-44">
                  Latest Run
                </th>
                <th className="px-4 py-3.5 text-left text-[10px] font-bold uppercase tracking-[0.15em] text-text-muted w-28">
                  Status
                </th>
                <th className="px-4 py-3.5 text-left text-[10px] font-bold uppercase tracking-[0.15em] text-text-muted w-32">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/50">
              {status?.steps.map((step, idx) => {
                const isCurrentStep = status.running && status.current_step === step.step

                return (
                  <tr
                    key={step.step}
                    className={`
                      transition-colors animate-fade-in opacity-0
                      ${isCurrentStep
                        ? 'bg-cyan/[0.03]'
                        : 'hover:bg-surface-hover'
                      }
                    `}
                    style={{ animationDelay: `${idx * 0.03}s` }}
                  >
                    <td className="px-4 py-3.5">
                      <span className={`
                        text-xs font-mono px-2.5 py-1 rounded-lg border
                        ${isCurrentStep
                          ? 'bg-cyan/10 text-cyan border-cyan/20'
                          : 'bg-surface-elevated text-text-muted border-border'
                        }
                      `}>
                        {step.step}
                      </span>
                    </td>
                    <td className="px-4 py-3.5">
                      <span className="text-sm font-medium text-text-primary">
                        {step.name}
                      </span>
                    </td>
                    <td className="px-4 py-3.5">
                      <span className="text-sm text-text-secondary line-clamp-1">
                        {step.description}
                      </span>
                    </td>
                    <td className="px-4 py-3.5">
                      <span className="text-xs font-mono text-text-muted">
                        {step.latest_run ? formatTimestamp(step.latest_run) : '-'}
                      </span>
                    </td>
                    <td className="px-4 py-3.5">
                      {isCurrentStep ? (
                        <div className="flex items-center gap-2">
                          <div className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan opacity-75" />
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan" />
                          </div>
                          <span className="text-xs font-semibold text-cyan">Running</span>
                        </div>
                      ) : step.has_data ? (
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 bg-emerald rounded-full" />
                          <span className="text-xs font-semibold text-emerald">Complete</span>
                        </div>
                      ) : (
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 bg-text-muted rounded-full" />
                          <span className="text-xs font-semibold text-text-muted">No data</span>
                        </div>
                      )}
                    </td>
                    <td className="px-4 py-3.5">
                      <button
                        onClick={() => runPipeline(step.step)}
                        disabled={runningPipeline || status?.running}
                        className="
                          px-3 py-1.5 text-xs font-semibold
                          bg-surface-elevated text-text-secondary border border-border rounded-lg
                          hover:bg-surface-hover hover:text-text-primary hover:border-cyan/30
                          disabled:opacity-50 disabled:cursor-not-allowed
                          transition-all
                        "
                      >
                        Run from here
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* CLI Commands Card */}
      <div className="rounded-xl border border-border bg-surface p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-8 h-8 rounded-lg bg-amber/10 flex items-center justify-center">
            <svg className="w-4 h-4 text-amber" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6.75 7.5l3 2.25-3 2.25m4.5 0h3m-9 8.25h13.5A2.25 2.25 0 0021 18V6a2.25 2.25 0 00-2.25-2.25H5.25A2.25 2.25 0 003 6v12a2.25 2.25 0 002.25 2.25z" />
            </svg>
          </div>
          <h3 className="font-display text-lg font-bold text-text-primary">CLI Commands</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {[
            { label: 'Full pipeline', command: 'poly run pipeline' },
            { label: 'From step', command: 'poly run pipeline --from-step 03_1' },
            { label: 'Quick refresh', command: 'poly run quick' },
            { label: 'Status', command: 'poly status' },
          ].map((item) => (
            <div
              key={item.label}
              className="flex items-center gap-4 px-4 py-3 rounded-lg bg-surface-elevated border border-border"
            >
              <span className="text-xs uppercase tracking-wider text-text-muted w-24 shrink-0">
                {item.label}
              </span>
              <code className="text-sm font-mono text-emerald">
                {item.command}
              </code>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
