'use client'

import { usePriceContext } from '@/components/PriceProvider'

export function usePrices() {
  return usePriceContext()
}
