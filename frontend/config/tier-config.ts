// =============================================================================
// COVERAGE STYLING - Color mapping for LLM confidence scores
// =============================================================================

export const getCoverageColor = (coverage: number): string => {
  if (coverage >= 0.95) return 'text-emerald'
  if (coverage >= 0.90) return 'text-cyan'
  if (coverage >= 0.85) return 'text-amber'
  return 'text-text-muted'
}

export const getCoverageBg = (coverage: number): string => {
  if (coverage >= 0.95) return 'bg-emerald'
  if (coverage >= 0.90) return 'bg-cyan'
  if (coverage >= 0.85) return 'bg-amber'
  return 'bg-text-muted'
}
