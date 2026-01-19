"""WebSocket endpoint for real-time portfolio price updates."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from server.portfolio_service import portfolio_service
from server.price_aggregation import price_aggregation

router = APIRouter()


# =============================================================================
# CONNECTION MANAGEMENT
# =============================================================================


@dataclass
class ClientState:
    """Per-client filter state."""

    websocket: WebSocket
    max_tier: int = 3
    profitable_only: bool = False
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PortfolioConnectionManager:
    """Manage portfolio WebSocket connections with per-client filter state."""

    def __init__(self):
        self.clients: dict[WebSocket, ClientState] = {}
        self._broadcast_lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> ClientState:
        """Accept a new connection and create client state."""
        await websocket.accept()
        state = ClientState(websocket=websocket)
        self.clients[websocket] = state
        logger.info(f"Portfolio WS client connected. Total: {len(self.clients)}")
        return state

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected client."""
        if websocket in self.clients:
            del self.clients[websocket]
            logger.info(f"Portfolio WS client disconnected. Total: {len(self.clients)}")

            # Unregister callback when last client disconnects to prevent memory leak
            if (
                len(self.clients) == 0
                and on_price_update in price_aggregation._callbacks
            ):
                price_aggregation.unregister_callback(on_price_update)
                logger.info("Unregistered price callback - no clients connected")

    def update_filters(
        self, websocket: WebSocket, max_tier: int, profitable_only: bool
    ) -> None:
        """Update a client's filter preferences."""
        if websocket in self.clients:
            self.clients[websocket].max_tier = max_tier
            self.clients[websocket].profitable_only = profitable_only
            logger.debug(
                f"Client filters updated: max_tier={max_tier}, "
                f"profitable_only={profitable_only}"
            )

    async def broadcast_delta(
        self, changed: list[dict], tier_changes: list[dict], timestamp: datetime
    ) -> None:
        """Broadcast portfolio changes to all connected clients."""
        if not self.clients:
            return

        # Build set of pair_ids that changed tier (for quick lookup)
        tier_change_map = {tc["pair_id"]: tc for tc in tier_changes}

        async with self._broadcast_lock:
            for websocket, state in list(self.clients.items()):
                try:
                    # Separate changed portfolios into those that match filters and those that don't
                    # Portfolios that no longer match filters should be removed from client display
                    filtered_changed = []
                    removed_ids = []

                    for p in changed:
                        pair_id = p.get("pair_id")
                        tier = p.get("tier", 4)
                        expected_profit = p.get("expected_profit", 0)

                        # Check if this portfolio changed tier
                        tier_change = tier_change_map.get(pair_id)

                        # If tier is now outside filter range
                        if tier > state.max_tier:
                            # Only add to removed if it WAS in range before (tier changed from inside to outside)
                            if (
                                tier_change
                                and tier_change["old_tier"] <= state.max_tier
                            ):
                                removed_ids.append(pair_id)
                            continue

                        # If tier is within range, check profitable filter
                        if state.profitable_only and expected_profit <= 0.001:
                            # Portfolio is in tier range but no longer profitable
                            removed_ids.append(pair_id)
                        else:
                            filtered_changed.append(p)

                    # Only send if there are changes for this client
                    if filtered_changed or removed_ids or tier_changes:
                        await websocket.send_json(
                            {
                                "type": "portfolio_update",
                                "timestamp": timestamp.isoformat(),
                                "changed": filtered_changed,
                                "removed": removed_ids,
                                "tier_changes": tier_changes,
                                "changed_count": len(filtered_changed),
                                "summary": portfolio_service.get_summary(),
                            }
                        )

                except Exception as e:
                    logger.error(f"Error sending to client: {e}")
                    # Don't remove here, let the main loop handle disconnect

    async def broadcast_full_reload(
        self, portfolios: list[dict], summary: dict, timestamp: datetime
    ) -> None:
        """Broadcast full portfolio reload to all connected clients."""
        if not self.clients:
            return

        logger.info(f"Broadcasting full reload to {len(self.clients)} clients")

        async with self._broadcast_lock:
            for websocket, state in list(self.clients.items()):
                try:
                    # Filter portfolios to match client's preferences
                    filtered = [
                        p
                        for p in portfolios
                        if self._matches_filters(
                            p, state.max_tier, state.profitable_only
                        )
                    ]

                    await websocket.send_json(
                        {
                            "type": "full_reload",
                            "timestamp": timestamp.isoformat(),
                            "portfolios": filtered,
                            "summary": summary,
                        }
                    )

                except Exception as e:
                    logger.error(f"Error sending full reload to client: {e}")

    def _matches_filters(
        self, portfolio: dict, max_tier: int, profitable_only: bool
    ) -> bool:
        """Check if a portfolio matches the filter criteria."""
        tier = portfolio.get("tier", 4)
        expected_profit = portfolio.get("expected_profit", 0)

        if tier > max_tier:
            return False

        if profitable_only and expected_profit <= 0.001:
            return False

        return True


manager = PortfolioConnectionManager()


# =============================================================================
# PRICE UPDATE CALLBACK
# =============================================================================


async def on_price_update(market_prices: dict[str, dict]) -> None:
    """Callback triggered when prices update. Calculates and broadcasts deltas."""
    if not manager.clients:
        return  # No clients connected, skip calculation

    # Calculate delta
    delta = portfolio_service.update_prices(market_prices)

    if delta.is_empty():
        return  # No changes, skip broadcast

    # Handle full reload (data was reset/changed significantly)
    if delta.full_reload:
        await manager.broadcast_full_reload(
            portfolios=delta.all_portfolios or [],
            summary=portfolio_service.get_summary(),
            timestamp=delta.timestamp,
        )
        return

    # Broadcast delta to all clients
    await manager.broadcast_delta(
        changed=delta.changed,
        tier_changes=delta.tier_changes,
        timestamp=delta.timestamp,
    )


# =============================================================================
# WEBSOCKET ENDPOINT
# =============================================================================


@router.websocket("/ws")
async def portfolio_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time portfolio price updates.

    Connection flow:
    1. Client connects
    2. Server sends initial portfolios based on default filters
    3. Client can send filter updates: {"type": "filter", "max_tier": 2, "profitable_only": true}
    4. Server pushes portfolio changes as prices update

    Message types (server → client):
    - connected: Initial connection confirmation
    - initial: Initial portfolio data
    - portfolio_update: Changes to portfolios from price updates
    - filter_ack: Acknowledgment of filter change

    Message types (client → server):
    - filter: Update filter preferences
    """
    state = await manager.connect(websocket)

    # Register for price updates (if not already registered)
    # Note: We register once globally, the callback checks if clients exist
    if on_price_update not in price_aggregation._callbacks:
        price_aggregation.register_callback(on_price_update)

    try:
        # Send connection confirmation
        await websocket.send_json(
            {
                "type": "connected",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Wait briefly for client's initial filter message before sending initial data
        # This allows client to specify filters before we send portfolios
        try:
            first_msg = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=0.5,  # Short timeout - client should send immediately
            )
            if first_msg.get("type") == "filter":
                state.max_tier = first_msg.get("max_tier", state.max_tier)
                state.profitable_only = first_msg.get(
                    "profitable_only", state.profitable_only
                )
                manager.update_filters(websocket, state.max_tier, state.profitable_only)
                logger.debug(
                    f"Client filters received before initial: max_tier={state.max_tier}, profitable_only={state.profitable_only}"
                )
        except asyncio.TimeoutError:
            # No filter message received, use defaults
            logger.debug("No initial filter message, using defaults")

        # Load portfolios and send initial data (with client's filters if provided)
        summary = portfolio_service.get_summary()
        initial_portfolios = portfolio_service.get_portfolios(
            max_tier=state.max_tier,
            profitable_only=state.profitable_only,
            # No limit - send all matching portfolios so counts stay consistent
        )

        await websocket.send_json(
            {
                "type": "initial",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "portfolios": initial_portfolios,
                "summary": summary,
                "filters": {
                    "max_tier": state.max_tier,
                    "profitable_only": state.profitable_only,
                },
            }
        )

        # Listen for client messages (filter updates)
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=60,  # Timeout to allow cleanup
                )

                if data.get("type") == "filter":
                    # Update filters
                    new_max_tier = data.get("max_tier", state.max_tier)
                    new_profitable_only = data.get(
                        "profitable_only", state.profitable_only
                    )

                    manager.update_filters(websocket, new_max_tier, new_profitable_only)

                    # Send filtered portfolios with new filters
                    filtered = portfolio_service.get_portfolios(
                        max_tier=new_max_tier,
                        profitable_only=new_profitable_only,
                        # No limit - send all matching portfolios
                    )

                    await websocket.send_json(
                        {
                            "type": "filter_ack",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "portfolios": filtered,
                            "count": len(filtered),
                            "filters": {
                                "max_tier": new_max_tier,
                                "profitable_only": new_profitable_only,
                            },
                        }
                    )

            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                try:
                    await websocket.send_json(
                        {
                            "type": "heartbeat",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                except Exception:
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Portfolio WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)
