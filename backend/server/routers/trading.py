"""Trading API endpoints."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

from server.routers.wallet import get_wallet_manager
from core.trading.executor import TradingExecutor


router = APIRouter()


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class BuyPairRequest(BaseModel):
    pair_id: str
    target_market_id: str
    target_position: str
    cover_market_id: str
    cover_position: str
    amount_per_position: float
    skip_clob_sell: bool = False


class TradeResultModel(BaseModel):
    success: bool
    market_id: str
    position: str
    amount: float
    split_tx: Optional[str]
    clob_order_id: Optional[str]
    clob_filled: bool
    error: Optional[str] = None


class BuyPairResponse(BaseModel):
    success: bool
    pair_id: str
    target: TradeResultModel
    cover: TradeResultModel
    total_spent: float
    final_balances: dict


class EstimateResponse(BaseModel):
    pair_id: str
    total_cost: float
    target_market: dict
    cover_market: dict
    wallet_balance: float
    sufficient_balance: bool


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/buy-pair", response_model=BuyPairResponse)
async def buy_pair(req: BuyPairRequest):
    """Execute a pair purchase (target + cover positions)."""
    wallet_manager = get_wallet_manager()

    if not wallet_manager.is_unlocked:
        raise HTTPException(status_code=401, detail="Unlock wallet first")

    executor = TradingExecutor(wallet_manager)

    try:
        result = await executor.buy_pair(
            pair_id=req.pair_id,
            target_market_id=req.target_market_id,
            target_position=req.target_position,
            cover_market_id=req.cover_market_id,
            cover_position=req.cover_position,
            amount_per_position=req.amount_per_position,
            skip_clob_sell=req.skip_clob_sell,
        )

        return BuyPairResponse(
            success=result.success,
            pair_id=result.pair_id,
            target=TradeResultModel(
                success=result.target.success,
                market_id=result.target.market_id,
                position=result.target.position,
                amount=result.target.amount,
                split_tx=result.target.split_tx,
                clob_order_id=result.target.clob_order_id,
                clob_filled=result.target.clob_filled,
                error=result.target.error,
            ),
            cover=TradeResultModel(
                success=result.cover.success,
                market_id=result.cover.market_id,
                position=result.cover.position,
                amount=result.cover.amount,
                split_tx=result.cover.split_tx,
                clob_order_id=result.cover.clob_order_id,
                clob_filled=result.cover.clob_filled,
                error=result.cover.error,
            ),
            total_spent=result.total_spent,
            final_balances=result.final_balances,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Buy pair failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/buy-pair/estimate", response_model=EstimateResponse)
async def estimate_buy_pair(req: BuyPairRequest):
    """Estimate costs for a pair purchase without executing."""
    wallet_manager = get_wallet_manager()
    executor = TradingExecutor(wallet_manager)

    try:
        target_market = await executor.get_market_info(req.target_market_id)
        cover_market = await executor.get_market_info(req.cover_market_id)

        total_cost = req.amount_per_position * 2
        balances = wallet_manager.get_balances()

        return EstimateResponse(
            pair_id=req.pair_id,
            total_cost=total_cost,
            target_market={
                "question": target_market.question[:60],
                "position": req.target_position,
                "price": target_market.yes_price if req.target_position == "YES" else target_market.no_price,
            },
            cover_market={
                "question": cover_market.question[:60],
                "position": req.cover_position,
                "price": cover_market.yes_price if req.cover_position == "YES" else cover_market.no_price,
            },
            wallet_balance=balances.usdc_e,
            sufficient_balance=balances.usdc_e >= total_cost,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
