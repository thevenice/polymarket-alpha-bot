'use client'

import { createContext, useContext, useEffect, useState, useRef, useCallback, ReactNode } from 'react'
import { getPricesWsUrl } from '@/config/api-config'

interface PriceData {
  price: number
  title?: string
  market_id?: string
}

interface PriceContextType {
  prices: Record<string, PriceData>
  connected: boolean
}

const PriceContext = createContext<PriceContextType>({
  prices: {},
  connected: false,
})

export function usePriceContext() {
  return useContext(PriceContext)
}

// Reconnection delay in milliseconds
const RECONNECT_DELAY = 5000

export function PriceProvider({ children }: { children: ReactNode }) {
  const [prices, setPrices] = useState<Record<string, PriceData>>({})
  const [connected, setConnected] = useState(false)

  // Refs to track lifecycle and prevent stale closures
  const mountedRef = useRef(true)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | undefined>(undefined)

  const connect = useCallback(() => {
    // Don't connect if unmounted
    if (!mountedRef.current) {
      return
    }

    // Don't create duplicate connections
    if (wsRef.current?.readyState === WebSocket.OPEN ||
        wsRef.current?.readyState === WebSocket.CONNECTING) {
      return
    }

    // Clean up any existing WebSocket (CLOSING state)
    if (wsRef.current) {
      wsRef.current.onopen = null
      wsRef.current.onclose = null
      wsRef.current.onerror = null
      wsRef.current.onmessage = null
      wsRef.current.close()
      wsRef.current = null
    }

    try {
      const ws = new WebSocket(getPricesWsUrl())
      wsRef.current = ws

      ws.onopen = () => {
        // Ignore if component unmounted or this is a stale WebSocket
        if (!mountedRef.current || wsRef.current !== ws) {
          ws.close()
          return
        }
        setConnected(true)
      }

      ws.onmessage = (event) => {
        // Ignore if component unmounted or this is a stale WebSocket
        if (!mountedRef.current || wsRef.current !== ws) {
          return
        }
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'price_update') {
            setPrices(data.prices)
          }
        } catch (e) {
          console.error('PriceProvider: Failed to parse message:', e)
        }
      }

      ws.onclose = () => {
        // Ignore if this is a stale WebSocket (replaced by new connection)
        if (wsRef.current !== ws) {
          return
        }

        wsRef.current = null

        // Only update state and reconnect if still mounted
        if (mountedRef.current) {
          setConnected(false)
          // Clear any existing reconnect timeout
          if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current)
          }
          reconnectTimeoutRef.current = setTimeout(connect, RECONNECT_DELAY)
        }
      }

      ws.onerror = () => {
        // Ignore if this is a stale WebSocket
        if (wsRef.current !== ws) {
          return
        }
        ws.close()
      }
    } catch (e) {
      console.error('PriceProvider: Failed to connect:', e)
      if (mountedRef.current) {
        setConnected(false)
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current)
        }
        reconnectTimeoutRef.current = setTimeout(connect, RECONNECT_DELAY)
      }
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    connect()

    return () => {
      mountedRef.current = false

      // Clear reconnect timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
        reconnectTimeoutRef.current = undefined
      }

      // Clean up WebSocket
      if (wsRef.current) {
        wsRef.current.onopen = null
        wsRef.current.onclose = null
        wsRef.current.onerror = null
        wsRef.current.onmessage = null
        wsRef.current.close()
        wsRef.current = null
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <PriceContext.Provider value={{ prices, connected }}>
      {children}
    </PriceContext.Provider>
  )
}
