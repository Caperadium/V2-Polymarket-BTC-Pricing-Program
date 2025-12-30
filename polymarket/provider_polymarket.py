"""
polymarket/provider_polymarket.py

Real Polymarket API provider for the Trade Console.

Uses:
- Data-API (https://data-api.polymarket.com) for closed positions, positions, value
- CLOB API (https://clob.polymarket.com) for trades (if needed)

Modes:
- read_only: Uses Data-API endpoints, no private key required
- trading: Requires private key for signing (L1) and trading operations

Environment Variables:
- POLYMARKET_USER_ADDRESS: User's wallet address (required for read_only)
- POLYMARKET_API_KEY: L2 API key (for CLOB authenticated endpoints)
- POLYMARKET_API_SECRET: L2 secret
- POLYMARKET_PASSPHRASE: L2 passphrase
- POLYMARKET_PRIVATE_KEY: Only needed for trading mode
"""

from __future__ import annotations

import json
import logging
import os
import time
import concurrent.futures
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_UP, ROUND_DOWN
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import requests

from polymarket.accounting import PolymarketProvider

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system env vars

logger = logging.getLogger(__name__)

# API Base URLs
DATA_API_BASE = "https://data-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"

# Polygon blockchain for direct USDC.e balance query
POLYGON_RPC = "https://polygon-rpc.com"
USDC_E_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
USDC_E_DECIMALS = 6

# Rate limiting
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # Exponential backoff: 2^retry seconds


class ProviderMode(Enum):
    READ_ONLY = "read_only"
    TRADING = "trading"


@dataclass
class ClosedPosition:
    """Closed position from Data-API."""
    position_id: str
    condition_id: str
    market_slug: str
    title: str
    outcome: str
    outcome_index: int
    avg_price: float
    size: float
    total_bought: float
    realized_pnl: float
    cur_price: float
    timestamp: str
    end_date: str
    raw_json: str


@dataclass
class Trade:
    """Trade from CLOB API."""
    trade_id: str
    timestamp: str
    market_slug: str
    market_id: str
    condition_id: str
    token_id: str
    side: str
    price: float
    size: float
    notional: float
    fee: float
    maker_taker: str
    order_id: str
    raw_json: str


class RealPolymarketProvider(PolymarketProvider):
    """
    Real Polymarket API provider.
    
    Supports two modes:
    - read_only: Uses Data-API for positions/value/closed-positions (no signing)
    - trading: Requires private key for L1 signing and trade execution
    
    For metrics/monitoring, use read_only mode to avoid private key exposure.
    """
    
    def __init__(
        self,
        mode: str = "read_only",
        user_address: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        passphrase: Optional[str] = None,
    ):
        """
        Initialize the provider.
        
        Args:
            mode: "read_only" or "trading"
            user_address: User's wallet address (0x...)
            api_key: L2 API key (for CLOB endpoints)
            api_secret: L2 secret
            passphrase: L2 passphrase
        """
        self.mode = ProviderMode(mode)
        
        # Load from env if not provided
        self.user_address = user_address or os.getenv("POLYMARKET_USER_ADDRESS", "")
        self.api_key = api_key or os.getenv("POLYMARKET_API_KEY", "")
        self.api_secret = api_secret or os.getenv("POLYMARKET_API_SECRET", "")
        self.passphrase = passphrase or os.getenv("POLYMARKET_PASSPHRASE", "")
        
        if not self.user_address:
            logger.warning("POLYMARKET_USER_ADDRESS not set. Some API calls will fail.")
        
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100)
        self._session.mount('https://', adapter)
        self._session.mount('http://', adapter)
        
        self._session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
    
    def _request_with_retry(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry and exponential backoff.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout in seconds
            
        Returns:
            Parsed JSON response
            
        Raises:
            requests.RequestException: On persistent failure
        """
        last_error = None
        
        for retry in range(MAX_RETRIES):
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                )
                
                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = RETRY_BACKOFF_BASE ** (retry + 1)
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.RequestException as e:
                last_error = e
                if retry < MAX_RETRIES - 1:
                    wait_time = RETRY_BACKOFF_BASE ** retry
                    logger.warning(f"Request failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {MAX_RETRIES} retries: {e}")
        
        raise last_error or Exception("Request failed with no error")
    
    def _get_l2_headers(self) -> Dict[str, str]:
        """
        Get L2 authentication headers for CLOB API.
        
        Returns:
            Dict with POLY_ADDRESS, POLY_API_KEY, POLY_API_SECRET, POLY_PASSPHRASE
        """
        return {
            "POLY_ADDRESS": self.user_address,
            "POLY_API_KEY": self.api_key,
            "POLY_API_SECRET": self.api_secret,
            "POLY_PASSPHRASE": self.passphrase,
        }
    
    def _has_l2_credentials(self) -> bool:
        """Check if L2 credentials are available."""
        return bool(self.api_key and self.api_secret and self.passphrase)
    
    # -------------------------------------------------------------------------
    # CLOB API Endpoints (require L2 auth)
    # -------------------------------------------------------------------------
    
    def fetch_clob_balance(self) -> Dict[str, float]:
        """
        Fetch balance from CLOB API using py-clob-client with private key.
        
        Returns:
            Dict with 'balance' and 'allowance' in USDC
        """
        private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        
        if not private_key:
            logger.warning("POLYMARKET_PRIVATE_KEY not set. Cannot fetch CLOB balance.")
            return {"balance": 0.0, "allowance": 0.0}
        
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            
            # Check for proxy address (required for browser wallet users)
            proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
            
            if proxy_address:
                # Browser wallet mode (Phantom, MetaMask, etc.)
                client = ClobClient(
                    host=CLOB_API_BASE,
                    chain_id=137,
                    key=private_key,
                    signature_type=2,
                    funder=proxy_address,
                )
            else:
                # EOA mode (direct wallet trading)
                client = ClobClient(
                    host=CLOB_API_BASE,
                    chain_id=137,
                    key=private_key,
                )
            
            # Derive API credentials
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)
            
            # Fetch USDC collateral balance
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            result = client.get_balance_allowance(params)
            
            # Parse result - balance is string, allowances is dict
            balance = float(result.get("balance", 0) or 0)
            # Get first allowance value or 0
            allowances = result.get("allowances", {})
            allowance = float(list(allowances.values())[0]) if allowances else 0.0
            
            logger.info(f"CLOB balance: {balance:.2f}, allowance: {allowance:.2f}")
            return {"balance": balance, "allowance": allowance}
            
        except Exception as e:
            logger.error(f"Failed to fetch CLOB balance: {e}")
            return {"balance": 0.0, "allowance": 0.0}
    
    def fetch_usdc_balance(self) -> float:
        """
        Fetch USDC.e balance from Polygon blockchain directly.
        
        This queries the actual USDC.e token contract on Polygon to get
        the proxy wallet's cash balance, which the CLOB API doesn't expose.
        
        Returns:
            USDC.e balance in USD
        """
        if not self.user_address:
            return 0.0
        
        try:
            from web3 import Web3
            
            w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
            
            # ERC20 balanceOf ABI
            abi = [{
                'inputs': [{'name': 'account', 'type': 'address'}],
                'name': 'balanceOf',
                'outputs': [{'name': '', 'type': 'uint256'}],
                'type': 'function'
            }]
            
            contract = w3.eth.contract(
                address=Web3.to_checksum_address(USDC_E_ADDRESS),
                abi=abi
            )
            
            balance_wei = contract.functions.balanceOf(
                Web3.to_checksum_address(self.user_address)
            ).call()
            
            balance = balance_wei / (10 ** USDC_E_DECIMALS)
            logger.info(f"USDC.e balance (on-chain): ${balance:.2f}")
            return balance
            
        except Exception as e:
            logger.error(f"Failed to fetch USDC.e balance: {e}")
            return 0.0
    
    # -------------------------------------------------------------------------
    # Data-API Endpoints (read_only compatible)
    # -------------------------------------------------------------------------
    
    def fetch_closed_positions(
        self,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "TIMESTAMP",
        sort_direction: str = "DESC",
    ) -> List[Dict[str, Any]]:
        """
        Fetch closed positions from Data-API.
        
        This is the primary source for realized PnL.
        
        Args:
            limit: Max positions per request (max 50)
            offset: Pagination offset
            sort_by: REALIZEDPNL, TITLE, PRICE, AVGPRICE, TIMESTAMP
            sort_direction: ASC or DESC
            
        Returns:
            List of closed position dicts with realizedPnl field
        """
        if not self.user_address:
            raise ValueError("user_address required for fetch_closed_positions")
        
        url = f"{DATA_API_BASE}/closed-positions"
        params = {
            "user": self.user_address,
            "limit": min(limit, 50),
            "offset": offset,
            "sortBy": sort_by,
            "sortDirection": sort_direction,
        }
        
        logger.debug(f"Fetching closed positions: offset={offset}, limit={limit}")
        return self._request_with_retry("GET", url, params=params)
    
    def fetch_all_closed_positions(
        self,
        max_positions: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all closed positions with pagination.
        
        Args:
            max_positions: Maximum positions to fetch (safety limit)
            
        Returns:
            List of all closed position dicts
        """
        all_positions = []
        offset = 0
        page_size = 50
        
        while len(all_positions) < max_positions:
            page = self.fetch_closed_positions(limit=page_size, offset=offset)
            
            if not page:
                break
            
            all_positions.extend(page)
            
            if len(page) < page_size:
                # Last page
                break
            
            offset += page_size
        
        logger.info(f"Fetched {len(all_positions)} closed positions")
        return all_positions
    
    def fetch_positions(self) -> List[Dict[str, Any]]:
        """
        Fetch current open positions from Data-API.
        
        Returns:
            List of open position dicts
        """
        if not self.user_address:
            raise ValueError("user_address required for fetch_positions")
        
        url = f"{DATA_API_BASE}/positions"
        params = {"user": self.user_address}
        
        logger.debug("Fetching current positions")
        return self._request_with_retry("GET", url, params=params)
    
    def fetch_holdings_value(self) -> float:
        """
        Fetch total holdings value from Data-API.
        
        Returns:
            Total USD value of holdings
        """
        if not self.user_address:
            raise ValueError("user_address required for fetch_holdings_value")
        
        url = f"{DATA_API_BASE}/value"
        params = {"user": self.user_address}
        
        logger.debug("Fetching holdings value")
        result = self._request_with_retry("GET", url, params=params)
        
        # Response format: [{'user': '...', 'value': X}] or {'value': X}
        value = 0.0
        if isinstance(result, list) and len(result) > 0:
            value = float(result[0].get("value", 0) or 0)
        elif isinstance(result, dict):
            value = float(result.get("value", result.get("totalValue", 0)) or 0)
        elif isinstance(result, (int, float)):
            value = float(result)
        else:
            logger.warning(f"Unexpected value response format: {result}")
        
        return value
    
    def fetch_total_positions_value(self) -> float:
        """
        Fetch total value of all open positions.
        
        Returns:
            Sum of currentValue across all positions
        """
        try:
            positions = self.fetch_positions()
            if not positions:
                return 0.0
            
            total = sum(float(p.get("currentValue", 0) or 0) for p in positions)
            return total
        except Exception as e:
            logger.error(f"Failed to fetch positions value: {e}")
            return 0.0
    
    # -------------------------------------------------------------------------
    # PolymarketProvider Interface (for compatibility with existing code)
    # -------------------------------------------------------------------------
    
    def get_balance_allowance(self) -> Tuple[float, float]:
        """
        Get balance and allowance.
        
        Uses on-chain USDC.e balance (most accurate for proxy wallet cash).
        """
        try:
            # Get USDC.e balance from Polygon blockchain (most accurate)
            usdc_balance = self.fetch_usdc_balance()
            
            # Also get positions value
            positions_value = self.fetch_total_positions_value()
            
            # Total = cash + positions
            total = usdc_balance + positions_value
            
            logger.info(f"Account: USDC.e=${usdc_balance:.2f}, Positions=${positions_value:.2f}, Total=${total:.2f}")
            
            # Return the USDC cash as balance (this is what user sees as "Cash" on Polymarket)
            return usdc_balance, usdc_balance
            
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return 0.0, 0.0
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get open orders. Uses fetch_open_orders for real API call.
        """
        return self.fetch_open_orders()
    
    def fetch_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch the status of a specific order from the CLOB API.
        
        Args:
            order_id: The order ID to check
            
        Returns:
            Dict with order status info, or None if not found
            Keys: order_id, status, size_matched, outcome
        """
        private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        
        if not private_key:
            logger.warning("POLYMARKET_PRIVATE_KEY not set. Cannot fetch order status.")
            return None
        
        try:
            from py_clob_client.client import ClobClient
            
            proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
            
            if proxy_address:
                client = ClobClient(
                    host=CLOB_API_BASE,
                    chain_id=137,
                    key=private_key,
                    signature_type=2,
                    funder=proxy_address,
                )
            else:
                client = ClobClient(
                    host=CLOB_API_BASE,
                    chain_id=137,
                    key=private_key,
                )
            
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)
            
            # Try to get the specific order
            order = client.get_order(order_id)
            
            if order:
                if isinstance(order, dict):
                    return {
                        "order_id": order.get("id"),
                        "status": order.get("status"),
                        "size_matched": float(order.get("size_matched", 0)),
                        "original_size": float(order.get("original_size", 0)),
                        "outcome": order.get("outcome"),
                    }
                else:
                    # Handle object response
                    return {
                        "order_id": getattr(order, "id", order_id),
                        "status": getattr(order, "status", "UNKNOWN"),
                        "size_matched": float(getattr(order, "size_matched", 0)),
                        "original_size": float(getattr(order, "original_size", 0)),
                        "outcome": getattr(order, "outcome", None),
                    }
            return None
            
        except Exception as e:
            logger.warning(f"Failed to fetch order status for {order_id}: {e}")
            return None
    
    def fetch_clob_token_id(self, contract_slug: str, side: str) -> Optional[str]:
        """
        Fetch the actual CLOB token_id for a contract and side.
        
        The CLOB API requires the numeric token_id, not the slug.
        Each market has two token_ids: one for YES, one for NO.
        
        Args:
            contract_slug: The contract in format like:
                "will-the-price-of-bitcoin-be-above-92000-on-december-31_2025-12-31_92000"
            side: "YES" or "NO"
            
        Returns:
            The token_id string, or None if not found
        """
        try:
            import requests
            import json
            import re
            
            # Parse the contract string to extract key info
            # Format: "slug_expiry-date_strike" or variations
            parts = contract_slug.split("_")
            slug_part = parts[0] if parts else contract_slug
            
            # Extract strike price from slug (e.g., "92000" from "...above-92000-on...")
            strike_match = re.search(r'above-(\d+)-on', slug_part)
            target_strike = float(strike_match.group(1)) if strike_match else None
            
            # If strike in parts, use that
            if len(parts) >= 3 and parts[2].replace('.', '').isdigit():
                target_strike = float(parts[2])
            
            # Extract month and day from slug (e.g., "december-31")
            month_match = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)-(\d+)', slug_part, re.IGNORECASE)
            
            if not month_match:
                logger.warning(f"Could not parse month/day from: {slug_part}")
                return None
            
            month = month_match.group(1).lower()
            day = int(month_match.group(2))
            
            # Build the API query slug: "bitcoin-above-on-{month}-{day}"
            api_slug = f"bitcoin-above-on-{month}-{day}"
            logger.info(f"Looking up token for: {api_slug} (strike={target_strike}, side={side})")
            
            # Query the Gamma API (for events/markets data)
            GAMMA_API_URL = "https://gamma-api.polymarket.com/events"
            resp = requests.get(
                GAMMA_API_URL,
                params={"slug": api_slug},
                timeout=30,
            )
            
            if resp.status_code != 200:
                logger.warning(f"Event lookup failed: {resp.status_code}")
                return None
            
            events = resp.json()
            if not events:
                logger.warning(f"No events found for slug: {api_slug}")
                return None
            
            # Find the market that matches our strike
            for event in events:
                markets = event.get("markets", [])
                for market in markets:
                    # Check if strike matches
                    question = market.get("question", "")
                    market_strike_match = re.search(r'\$?(\d+(?:,\d+)?(?:\.\d+)?)(k)?', question, re.IGNORECASE)
                    if market_strike_match:
                        val_str = market_strike_match.group(1).replace(',', '')
                        market_strike = float(val_str)
                        if market_strike_match.group(2) and market_strike_match.group(2).lower() == 'k':
                            market_strike *= 1000
                        
                        # Check if this is our target strike
                        if target_strike and abs(market_strike - target_strike) > 100:
                            continue  # Skip, wrong strike
                    
                    clob_token_ids = market.get("clobTokenIds", "[]")
                    outcomes = market.get("outcomes", "[]")
                    
                    # Parse JSON strings
                    try:
                        token_ids = json.loads(clob_token_ids) if isinstance(clob_token_ids, str) else clob_token_ids
                        outcome_list = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
                    except:
                        continue
                    
                    if not token_ids or not outcome_list:
                        continue
                    
                    # Find the index for YES or NO
                    side_upper = side.upper()
                    for i, outcome in enumerate(outcome_list):
                        if outcome.upper() == side_upper and i < len(token_ids):
                            token_id = token_ids[i]
                            logger.info(f"Found token_id for {side} @ strike {market_strike}: {token_id[:20]}...")
                            return token_id
            
            logger.warning(f"Could not find token_id for {side} in {api_slug}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch token_id: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def place_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> Dict[str, Any]:
        """
        Place an order on the Polymarket CLOB.
        
        Args:
            token_id: The token/condition ID to trade
            side: "BUY" or "SELL"
            price: Limit price (0 < price < 1)
            size: Number of shares
            
        Returns:
            Dict with order_id and response details
        """
        private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        
        if not private_key:
            raise RuntimeError("POLYMARKET_PRIVATE_KEY required for trading")
        
        if self.mode == ProviderMode.READ_ONLY:
            raise RuntimeError("place_order not available in read_only mode")
        
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import OrderArgs, OrderType
            
            # Check for proxy address (required for browser wallet users)
            proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
            
            if proxy_address:
                # Browser wallet mode (Phantom, MetaMask, etc.)
                # signature_type=2 for browser wallets
                logger.info(f"Using browser wallet mode with proxy: {proxy_address[:10]}...")
                client = ClobClient(
                    host=CLOB_API_BASE,
                    chain_id=137,  # Polygon mainnet
                    key=private_key,
                    signature_type=2,  # Browser wallet
                    funder=proxy_address,
                )
            else:
                # EOA mode (direct wallet trading)
                logger.info("Using EOA mode (no proxy address)")
                client = ClobClient(
                    host=CLOB_API_BASE,
                    chain_id=137,  # Polygon mainnet
                    key=private_key,
                )
            
            # Derive API credentials
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)
            
            # Map side to CLOB format
            # For opening positions: YES recommendation = BUY YES token
            #                        NO recommendation = BUY NO token  
            # The token_id already points to the correct token (YES or NO)
            # So we always BUY when opening a new position
            clob_side = "BUY"  # Always BUY when opening positions
            
            # Create order args
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=clob_side,
            )
            
            logger.info(f"Placing order: {clob_side} {size:.2f} @ {price:.4f} on {token_id[:16]}...")
            
            # Create and post the order
            response = client.create_and_post_order(order_args)
            
            logger.info(f"Order placed successfully: {response}")
            
            # Extract order ID from response
            order_id = None
            if isinstance(response, dict):
                order_id = response.get("orderID") or response.get("order_id")
            elif hasattr(response, "orderID"):
                order_id = response.orderID
            
            return {
                "success": True,
                "order_id": order_id,
                "raw_response": response if isinstance(response, dict) else str(response),
            }
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {
                "success": False,
                "order_id": None,
                "error": str(e),
            }

    def fetch_open_orders(self) -> List[Dict[str, Any]]:
        """
        Fetch all open/live orders from the CLOB.
        
        Returns:
            List of order dicts with: token_id, side, outcome, price, size, order_id
        """
        private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        
        if not private_key:
            logger.warning("POLYMARKET_PRIVATE_KEY not set. Cannot fetch open orders.")
            return []
        
        try:
            from py_clob_client.client import ClobClient
            
            # Check for proxy address (browser wallet mode)
            proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
            
            if proxy_address:
                client = ClobClient(
                    host=CLOB_API_BASE,
                    chain_id=137,
                    key=private_key,
                    signature_type=2,
                    funder=proxy_address,
                )
            else:
                client = ClobClient(
                    host=CLOB_API_BASE,
                    chain_id=137,
                    key=private_key,
                )
            
            # Derive API credentials
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)
            
            # Fetch open orders
            orders = client.get_orders()
            
            if not orders:
                return []
            
            # Parse orders into simpler format
            result = []
            for order in orders:
                if isinstance(order, dict):
                    result.append({
                        "order_id": order.get("id"),
                        "token_id": order.get("asset_id"),
                        "side": order.get("side"),
                        "outcome": order.get("outcome"),
                        "price": float(order.get("price", 0)),
                        "size": float(order.get("original_size", 0)),
                        "status": order.get("status"),
                    })
            
            logger.info(f"Fetched {len(result)} open orders from CLOB")
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch open orders: {e}")
            return []
    
    def fetch_live_prices(
        self,
        token_ids: List[str],
        timeout: float = 2.0,
    ) -> Dict[str, Dict[str, Decimal]]:
        """
        Fetch live bid/ask prices from CLOB GET /price endpoint.
        
        Uses GET /price for each token_id (with BUY and SELL sides).
        Returns Decimal for precision.
        
        Args:
            token_ids: List of token IDs to fetch prices for
            timeout: Request timeout in seconds (default 2s)
            
        Returns:
            Dict mapping token_id -> {"bid": Decimal, "ask": Decimal, "tick_size": Decimal}
            Missing tokens will not be in the result dict.
        """
        if not token_ids:
            return {}
        
        result = {}
        
        for token_id in token_ids:
            try:
                # Fetch BUY side price (market makers' bid - what they'll pay to buy from you)
                buy_response = requests.get(
                    f"{CLOB_API_BASE}/price",
                    params={"token_id": token_id, "side": "BUY"},
                    timeout=timeout,
                )
                buy_response.raise_for_status()
                buy_data = buy_response.json()
                
                # Fetch SELL side price (market makers' ask - what they'll sell to you at)
                sell_response = requests.get(
                    f"{CLOB_API_BASE}/price",
                    params={"token_id": token_id, "side": "SELL"},
                    timeout=timeout,
                )
                sell_response.raise_for_status()
                sell_data = sell_response.json()
                
                # Parse prices - API returns {"price": "0.50"}
                # side=BUY returns bid (what market makers pay to buy)
                # side=SELL returns ask (what market makers charge to sell)
                bid_price = buy_data.get("price")   # BUY side = bid
                ask_price = sell_data.get("price")  # SELL side = ask
                
                if ask_price is None and bid_price is None:
                    logger.warning(f"No prices found for token {token_id[:20]}...")
                    continue
                
                result[token_id] = {
                    "ask": Decimal(ask_price) if ask_price else None,
                    "bid": Decimal(bid_price) if bid_price else None,
                    "tick_size": Decimal("0.01"),  # Default tick size
                }
                
                logger.debug(
                    f"Live price for {token_id[:12]}...: "
                    f"bid={bid_price}, ask={ask_price}"
                )
                
            except requests.Timeout:
                logger.warning(f"CLOB /price timeout for {token_id[:20]}...")
                continue
            except requests.RequestException as e:
                logger.warning(f"CLOB /price request failed for {token_id[:20]}...: {e}")
                continue
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse price for {token_id[:20]}...: {e}")
                continue
                
        logger.info(f"Fetched live prices for {len(result)}/{len(token_ids)} tokens")
        return result
    
    def fetch_live_price_single(
        self,
        token_id: str,
        timeout: float = 2.0,
    ) -> Optional[Dict[str, Decimal]]:
        """
        Fetch live price for a single token.
        
        Returns:
            {"bid": Decimal, "ask": Decimal, "tick_size": Decimal} or None
        """
        prices = self.fetch_live_prices([token_id], timeout=timeout)
        return prices.get(token_id)
    
    def fetch_order_book_depth(
        self,
        token_id: str,
        timeout: float = 5.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch order book with depth/volume information from CLOB /book endpoint.
        
        Args:
            token_id: The token ID to fetch order book for
            timeout: Request timeout in seconds
            
        Returns:
            Dict with:
                - best_bid: Decimal (best bid price)
                - best_ask: Decimal (best ask price) 
                - bid_depth: Decimal (total size at best bid)
                - ask_depth: Decimal (total size at best ask)
                - bid_levels: List[Dict] (all bid levels with price and size)
                - ask_levels: List[Dict] (all ask levels with price and size)
                - spread: Decimal (ask - bid)
            Or None if failed
        """
        try:
            # Fetch order book from CLOB API
            response = self._session.get(
                f"{CLOB_API_BASE}/book",
                params={"token_id": token_id},
                timeout=timeout,
            )
            response.raise_for_status()
            book_data = response.json()
            
            # Parse bids and asks
            # Format: {"bids": [{"price": "0.50", "size": "100"}, ...], "asks": [...]}
            bids = book_data.get("bids", [])
            asks = book_data.get("asks", [])
            
            # Parse bid levels
            bid_levels = []
            for bid in bids:
                try:
                    bid_levels.append({
                        "price": Decimal(str(bid.get("price", "0"))),
                        "size": Decimal(str(bid.get("size", "0"))),
                    })
                except:
                    continue
            
            # Parse ask levels
            ask_levels = []
            for ask in asks:
                try:
                    ask_levels.append({
                        "price": Decimal(str(ask.get("price", "0"))),
                        "size": Decimal(str(ask.get("size", "0"))),
                    })
                except:
                    continue
            
            # Sort to get correct best bid/ask
            # Best bid = HIGHEST price (sort descending)
            # Best ask = LOWEST price (sort ascending)
            bid_levels.sort(key=lambda x: x["price"], reverse=True)
            ask_levels.sort(key=lambda x: x["price"], reverse=False)
            
            # Get best bid/ask and their depths
            best_bid = bid_levels[0]["price"] if bid_levels else None
            best_ask = ask_levels[0]["price"] if ask_levels else None
            bid_depth = bid_levels[0]["size"] if bid_levels else Decimal("0")
            ask_depth = ask_levels[0]["size"] if ask_levels else Decimal("0")
            
            # Calculate spread
            spread = None
            if best_bid and best_ask:
                spread = best_ask - best_bid
            
            result = {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
                "bid_levels": bid_levels,
                "ask_levels": ask_levels,
                "spread": spread,
            }
            
            logger.debug(
                f"Order book for {token_id[:12]}...: "
                f"bid={best_bid}({bid_depth}), ask={best_ask}({ask_depth}), spread={spread}"
            )
            
            return result
            
        except requests.Timeout:
            logger.warning(f"CLOB /book timeout for {token_id[:20]}...")
            return None
        except requests.RequestException as e:
            logger.warning(f"CLOB /book request failed for {token_id[:20]}...: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to parse order book for {token_id[:20]}...: {e}")
            return None
    
    def fetch_live_prices_with_depth(
        self,
        token_ids: List[str],
        order_dollars: float = 10.0,
        timeout: float = 5.0,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch live prices with depth check from order book.
        
        Checks if there's sufficient liquidity to fill an order of the given dollar size.
        Uses the actual best_ask price to calculate required shares.
        
        Args:
            token_ids: List of token IDs to fetch
            order_dollars: Expected order size in dollars
            timeout: Request timeout in seconds
            
        Returns:
            Dict mapping token_id -> {
                "ask": Decimal,
                "bid": Decimal,
                "ask_depth": Decimal,
                "bid_depth": Decimal,
                "spread": Decimal,
                "has_sufficient_depth": bool,
                "required_shares": Decimal,
            }
        """
        result = {}
        order_dollars_decimal = Decimal(str(order_dollars))
        
        # Parallel execution using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            # Create a future for each token
            future_to_token = {
                executor.submit(self.fetch_order_book_depth, token_id, timeout): token_id
                for token_id in token_ids
            }
            
            for future in concurrent.futures.as_completed(future_to_token):
                token_id = future_to_token[future]
                try:
                    book = future.result()
                    
                    if book and book.get("best_ask"):
                        best_ask = book["best_ask"]
                        ask_depth = book["ask_depth"]
                        
                        # Calculate required shares using ACTUAL price
                        if best_ask > 0:
                            required_shares = order_dollars_decimal / best_ask
                        else:
                            required_shares = Decimal("0")
                        
                        has_depth = ask_depth >= required_shares
                        
                        result[token_id] = {
                            "ask": best_ask,
                            "bid": book["best_bid"],
                            "ask_depth": ask_depth,
                            "bid_depth": book["bid_depth"],
                            "spread": book["spread"],
                            "has_sufficient_depth": has_depth,
                            "required_shares": required_shares,
                            "tick_size": Decimal("0.01"),
                        }
                        
                        if not has_depth:
                            logger.warning(
                                f"Insufficient depth for {token_id[:16]}...: "
                                f"ask_depth={ask_depth:.0f} < required={required_shares:.0f} shares "
                                f"(${order_dollars} @ ${best_ask})"
                            )
                except Exception as e:
                    logger.warning(f"Error fetching price for {token_id}: {e}")
        
        logger.info(f"Fetched order books for {len(result)}/{len(token_ids)} tokens")
        return result


def snap_to_tick(
    price: Decimal,
    tick_size: Decimal,
    side: str,
) -> Decimal:
    """
    Snap price to tick size based on side.
    
    - BUY: round UP to tick (we pay more)
    - SELL: round DOWN to tick (we receive less)
    
    Args:
        price: Raw price as Decimal
        tick_size: Minimum price increment
        side: "BUY" or "SELL"
        
    Returns:
        Price snapped to tick size
    """
    if tick_size <= 0:
        return price
    
    # Calculate number of ticks
    ticks = price / tick_size
    
    if side.upper() == "BUY":
        # Round up for buys
        snapped_ticks = ticks.quantize(Decimal("1"), rounding=ROUND_UP)
    else:
        # Round down for sells
        snapped_ticks = ticks.quantize(Decimal("1"), rounding=ROUND_DOWN)
    
    return snapped_ticks * tick_size


def get_executable_price(
    live_price_info: Dict[str, Decimal],
    action: str,
) -> Optional[Decimal]:
    """
    Get the executable price for an action.
    
    - BUY: cross the spread, execute at ask
    - SELL: cross the spread, execute at bid
    
    Args:
        live_price_info: {"bid": Decimal, "ask": Decimal, ...}
        action: "BUY" or "SELL"
        
    Returns:
        Executable price as Decimal, or None if not available
    """
    if action.upper() == "BUY":
        return live_price_info.get("ask")
    else:
        return live_price_info.get("bid")


def get_provider(mode: str = "read_only") -> RealPolymarketProvider:
    """
    Factory function to create a configured provider.
    
    Args:
        mode: "read_only" or "trading"
        
    Returns:
        Configured RealPolymarketProvider
    """
    return RealPolymarketProvider(mode=mode)


def check_provider_config() -> Dict[str, bool]:
    """
    Check which environment variables are configured.
    
    Returns:
        Dict with config status (never includes actual values)
    """
    return {
        "user_address_set": bool(os.getenv("POLYMARKET_USER_ADDRESS")),
        "api_key_set": bool(os.getenv("POLYMARKET_API_KEY")),
        "api_secret_set": bool(os.getenv("POLYMARKET_API_SECRET")),
        "passphrase_set": bool(os.getenv("POLYMARKET_PASSPHRASE")),
        "private_key_set": bool(os.getenv("POLYMARKET_PRIVATE_KEY")),
    }
