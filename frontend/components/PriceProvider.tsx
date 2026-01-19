'use client'

import { createContext, useContext, useEffect, useState, useRef, useCallback, ReactNode } from 'react'
import { getPricesWsUrl } from '@/config/api-config'

// Connection counter for debugging
let connectionAttemptCounter = 0

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
      console.log('PriceProvider: connect() skipped - not mounted')
      return
    }

    // Don't create duplicate connections
    if (wsRef.current?.readyState === WebSocket.OPEN ||
        wsRef.current?.readyState === WebSocket.CONNECTING) {
      console.log('PriceProvider: connect() skipped - already connected/connecting')
      return
    }

    connectionAttemptCounter++
    const connectionId = connectionAttemptCounter
    console.log(`PriceProvider: connect() #${connectionId} starting...`)

    // Clean up any existing WebSocket
    if (wsRef.current) {
      console.log('PriceProvider: CLOSE PATH A - cleaning up existing WebSocket before new connection')
      wsRef.current.onopen = null
      wsRef.current.onclose = null
      wsRef.current.onerror = null
      wsRef.current.onmessage = null
      wsRef.current.close()
      wsRef.current = null
    }

    try {
      const wsUrl = `${getPricesWsUrl()}?cid=${connectionId}`
      console.log(`PriceProvider: Creating WebSocket #${connectionId} to ${wsUrl}`)
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        // Ignore if component unmounted or this is a stale WebSocket
        if (!mountedRef.current || wsRef.current !== ws) {
          console.log(`PriceProvider: CLOSE PATH B #${connectionId} - onopen guard: mounted=`, mountedRef.current, 'isCurrentWs=', wsRef.current === ws)
          ws.close()
          return
        }
        console.log(`PriceProvider: WebSocket #${connectionId} connected`)
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

      ws.onclose = (event) => {
        // Ignore if this is a stale WebSocket (replaced by new connection)
        if (wsRef.current !== ws) {
          console.log(`PriceProvider: onclose #${connectionId} ignored - stale WebSocket`)
          return
        }

        console.log(`PriceProvider: WebSocket #${connectionId} disconnected, mounted:`, mountedRef.current, 'code:', event.code, 'reason:', event.reason, 'wasClean:', event.wasClean)
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

      ws.onerror = (event) => {
        // Ignore if this is a stale WebSocket
        if (wsRef.current !== ws) {
          return
        }
        console.error('PriceProvider: CLOSE PATH C - onerror triggered', event)
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
    console.log('PriceProvider: mounting')
    mountedRef.current = true

    connect()

    return () => {
      console.log('PriceProvider: cleanup')
      mountedRef.current = false

      // Clear reconnect timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
        reconnectTimeoutRef.current = undefined
      }

      // Clean up WebSocket
      if (wsRef.current) {
        console.log('PriceProvider: CLOSE PATH D - cleanup on unmount')
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
