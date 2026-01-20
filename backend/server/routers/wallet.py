"""Wallet management API endpoints."""

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

from core.wallet.storage import WalletStorage
from core.wallet.manager import WalletManager


router = APIRouter()

# Singleton wallet manager
_wallet_manager: Optional[WalletManager] = None


def get_wallet_manager() -> WalletManager:
    """Get or create wallet manager singleton."""
    global _wallet_manager
    if _wallet_manager is None:
        wallet_path = Path("data/wallet.enc")
        rpc_url = os.environ.get("CHAINSTACK_NODE", "")
        if not rpc_url:
            raise HTTPException(status_code=500, detail="CHAINSTACK_NODE not configured")

        storage = WalletStorage(wallet_path)
        _wallet_manager = WalletManager(storage, rpc_url)
    return _wallet_manager


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class PasswordRequest(BaseModel):
    password: str


class ImportRequest(BaseModel):
    private_key: str
    password: str


class WalletStatusResponse(BaseModel):
    exists: bool
    address: Optional[str]
    unlocked: bool
    balances: Optional[dict]
    approvals_set: bool


class GenerateResponse(BaseModel):
    address: str
    message: str


class UnlockResponse(BaseModel):
    unlocked: bool
    address: str
    balances: dict


class ApprovalResponse(BaseModel):
    success: bool
    tx_hashes: list[str]


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get("/status", response_model=WalletStatusResponse)
async def get_status():
    """Get wallet status including balances and approval state."""
    manager = get_wallet_manager()
    status = manager.get_status()

    return WalletStatusResponse(
        exists=status.exists,
        address=status.address,
        unlocked=status.unlocked,
        balances={
            "pol": status.balances.pol,
            "usdc_e": status.balances.usdc_e,
        } if status.balances else None,
        approvals_set=status.approvals_set,
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate_wallet(req: PasswordRequest):
    """Generate a new wallet encrypted with password."""
    manager = get_wallet_manager()

    try:
        address = manager.generate(req.password)
        return GenerateResponse(
            address=address,
            message="Wallet created. Fund with POL and USDC.e, then set approvals.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/import", response_model=GenerateResponse)
async def import_wallet(req: ImportRequest):
    """Import existing private key encrypted with password."""
    manager = get_wallet_manager()

    try:
        address = manager.import_key(req.private_key, req.password)
        return GenerateResponse(
            address=address,
            message="Wallet imported. Check balances and set approvals if needed.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/unlock", response_model=UnlockResponse)
async def unlock_wallet(req: PasswordRequest):
    """Unlock wallet for trading (decrypt key into memory)."""
    manager = get_wallet_manager()

    try:
        address = manager.unlock(req.password)
        balances = manager.get_balances()
        return UnlockResponse(
            unlocked=True,
            address=address,
            balances={"pol": balances.pol, "usdc_e": balances.usdc_e},
        )
    except ValueError as e:
        raise HTTPException(status_code=401, detail="Invalid password")


@router.post("/lock")
async def lock_wallet():
    """Lock wallet (clear key from memory)."""
    manager = get_wallet_manager()
    manager.lock()
    return {"locked": True}


@router.post("/approve-contracts", response_model=ApprovalResponse)
async def approve_contracts():
    """Set all Polymarket contract approvals (requires unlocked wallet)."""
    manager = get_wallet_manager()

    if not manager.is_unlocked:
        raise HTTPException(status_code=401, detail="Unlock wallet first")

    try:
        tx_hashes = manager.set_approvals()
        return ApprovalResponse(success=True, tx_hashes=tx_hashes)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
