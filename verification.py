import re, math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional, Tuple, Callable, Union, Any, Dict
from plotly.subplots import make_subplots

COLUMNS = [
    'arrival_timestamp','Source','StockId','SeqNo','EntryType',
    'OrderbookStateCode','OrderbookFlush','TradePrice',
    # Bids
    'BidPrice0','BidPrice1','BidPrice2','BidPrice3','BidPrice4',
    'BidPrice5','BidPrice6','BidPrice7','BidPrice8','BidPrice9',
    'BidPrice10','BidPrice11','BidPrice12','BidPrice13','BidPrice14',
    'BidPrice15','BidPrice16','BidPrice17','BidPrice18','BidPrice19',
    'BidSize0','BidSize1','BidSize2','BidSize3','BidSize4',
    'BidSize5','BidSize6','BidSize7','BidSize8','BidSize9',
    'BidSize10','BidSize11','BidSize12','BidSize13','BidSize14',
    'BidSize15','BidSize16','BidSize17','BidSize18','BidSize19',
    'BidOrders0','BidOrders1','BidOrders2','BidOrders3','BidOrders4',
    'BidOrders5','BidOrders6','BidOrders7','BidOrders8','BidOrders9',
    'BidOrders10','BidOrders11','BidOrders12','BidOrders13','BidOrders14',
    'BidOrders15','BidOrders16','BidOrders17','BidOrders18','BidOrders19',
    # Asks
    'AskPrice0','AskPrice1','AskPrice2','AskPrice3','AskPrice4',
    'AskPrice5','AskPrice6','AskPrice7','AskPrice8','AskPrice9',
    'AskPrice10','AskPrice11','AskPrice12','AskPrice13','AskPrice14',
    'AskPrice15','AskPrice16','AskPrice17','AskPrice18','AskPrice19',
    'AskSize0','AskSize1','AskSize2','AskSize3','AskSize4',
    'AskSize5','AskSize6','AskSize7','AskSize8','AskSize9',
    'AskSize10','AskSize11','AskSize12','AskSize13','AskSize14',
    'AskSize15','AskSize16','AskSize17','AskSize18','AskSize19',
    'AskOrders0','AskOrders1','AskOrders2','AskOrders3','AskOrders4',
    'AskOrders5','AskOrders6','AskOrders7','AskOrders8','AskOrders9',
    'AskOrders10','AskOrders11','AskOrders12','AskOrders13','AskOrders14',
    'AskOrders15','AskOrders16','AskOrders17','AskOrders18','AskOrders19'
]



# MATCHING ALGORITHM:

def get_tick_size(price: float) -> float:
    if price < 25.00:
        return 0.01
    elif price < 50.00:
        return 0.02
    elif price < 100.00:
        return 0.05
    elif price < 250.00:
        return 0.10
    elif price < 500.00:
        return 0.20
    else:
        return 0.50

# This is a helper function that prepares the order book snapshot in a clean format for the matching algorith to use.
def levels_from_row(row: pd.Series, side_prefix: str, debug: bool = False) -> list[tuple[float, float]]:
    out = []
    i = 0
    while True:
        price_key = f"{side_prefix}Price{i}"
        size_key  = f"{side_prefix}Size{i}"

        # Stop when neither column exists (we've passed the last level)
        if price_key not in row and size_key not in row:
            break

        p = pd.to_numeric(row.get(price_key), errors="coerce")
        s = pd.to_numeric(row.get(size_key),  errors="coerce")
        if pd.notna(p) and pd.notna(s):
            if debug:
                print(f"{side_prefix} level {i}: {price_key}={p}, {size_key}={s}")
            out.append((float(p), float(s)))
        i += 1

    if debug:
        print(f"Total {side_prefix} levels used: {len(out)}\n")

    return out



def auction_match_row(row: pd.Series, debug: bool = False) -> tuple[float, float, str]:
    # This reads the ladder.
    bids = levels_from_row(row, "Bid", debug=False)
    asks = levels_from_row(row, "Ask", debug=False)
    if not bids or not asks:
        return (np.nan, np.nan, "NONE")

    bp_all = np.array([p for p, _ in bids], dtype=float)
    bs_all = np.array([s for _, s in bids], dtype=float)
    ap_all = np.array([p for p, _ in asks], dtype=float)
    as_all = np.array([s for _, s in asks], dtype=float)

    # This separates the market orders (0.0 prices) from limit orders.
    bid_market_vol = bs_all[bp_all == 0.0].sum() if (bp_all == 0.0).any() else 0.0
    ask_market_vol = as_all[ap_all == 0.0].sum() if (ap_all == 0.0).any() else 0.0
    
    # This is the positive-price arrays (limit orders only).
    bid_prices = bp_all[bp_all > 0.0]
    bid_sizes  = bs_all[bp_all > 0.0]
    ask_prices = ap_all[ap_all > 0.0]
    ask_sizes  = as_all[ap_all > 0.0]
    if bid_prices.size == 0 or ask_prices.size == 0:
        return (np.nan, np.nan, "NONE")

    best_bid = float(bid_prices.max())
    best_ask = float(ask_prices.min())

    # The candidate prices: use all prices in crossed markets, crossing range in normal markets.
    cand_union = sorted(set(bid_prices.tolist() + ask_prices.tolist()))
    
    # This checks if market is crossed
    market_is_crossed = best_bid > best_ask
    
    if market_is_crossed:
        # In crossed markets, use all prices as candidates.
        cand_math = cand_union
    else:
        # In normal markets, use crossing range or fallback to all prices.
        cand_math = [p for p in cand_union if best_ask <= p <= best_bid] or cand_union

    has_zero_on_ladder = bool((bp_all == 0.0).any() or (ap_all == 0.0).any())
    cand_print = ([0.0] if has_zero_on_ladder else []) + cand_math

    # This builds the results for printing (includes p=0.0).
    results_print = []
    for p in cand_print:
        if p == 0.0:
            # At price 0, no limit orders execute, only market vs market.
            exec_vol = min(bid_market_vol, ask_market_vol)
            imbalance = bid_market_vol - ask_market_vol
        else:
            # For limit orders: market orders participate + limit orders at/better than price.
            bid_qty = bid_sizes[bid_prices >= p].sum() + bid_market_vol
            ask_qty = ask_sizes[ask_prices <= p].sum() + ask_market_vol
            exec_vol = min(bid_qty, ask_qty)
            imbalance = bid_qty - ask_qty
        results_print.append((p, exec_vol, imbalance))

    results = []
    for p in cand_math:
        # For limit orders: market orders participate + limit orders at/better than price.
        bid_qty = bid_sizes[bid_prices >= p].sum() + bid_market_vol
        ask_qty = ask_sizes[ask_prices <= p].sum() + ask_market_vol
        exec_vol = min(bid_qty, ask_qty)
        imbalance = bid_qty - ask_qty
        results.append((p, exec_vol, imbalance))

    if debug:
        print("=== Auction Debug Snapshot ===")
        print("Bid levels:", bids)
        print("Ask levels:", asks)
        print("Candidate prices:", cand_print) # This shows the 0.0 if present.
        for p, exec_vol, imb in results_print:
            print(f"p={p:.2f}, exec_vol={exec_vol}, imbalance={imb}")
        try:
            ev_2795 = next(ev for p, ev, imb in results_print if abs(p-27.95) < 1e-6)
            ev_2800 = next(ev for p, ev, imb in results_print if abs(p-28.00) < 1e-6)
            print(f"Extra buy needed at 28.00 = {ev_2795 - ev_2800}")
        except StopIteration:
            pass
        print("==============================")

    # This chooses the price with max executable volume.
    max_exec = max(r[1] for r in results)
    winners = [r for r in results if r[1] == max_exec]

    # This minimizes the residual (unmatched quantity).
    min_abs_imb = min(abs(r[2]) for r in winners)
    winners = [r for r in winners if abs(r[2]) == min_abs_imb]

    # A tie-break by side of residual.
    imbs = [r[2] for r in winners]
    pos_only = all(imb > 0 for imb in imbs)
    neg_only = all(imb < 0 for imb in imbs)

    if pos_only:
        # (a) An imbalance on buy side only -> highest price.
        p_star, exec_star, imb = max(winners, key=lambda r: r[0])
    elif neg_only:
        # (b) An imbalance on sell side only -> lowest price
        p_star, exec_star, imb = min(winners, key=lambda r: r[0])
    else:
        # (c) An imbalance on both sides -> average of (a) and (b), then round to closest tick.
        p_low  = min(r[0] for r in winners)
        p_high = max(r[0] for r in winners)
        p_avg  = (p_low + p_high) / 2.0
        tick   = get_tick_size(p_avg)
        p_star = float(np.round(p_avg / tick) * tick)

        # The exec volume and side at p_star should reflect that clearing (recompute imbalance sign).
        bid_qty = bid_sizes[bid_prices >= p_star].sum()
        ask_qty = ask_sizes[ask_prices <= p_star].sum()
        exec_star = float(min(bid_qty, ask_qty))
        imb = float(bid_qty - ask_qty)

    side = "BUY" if imb > 0 else ("SELL" if imb < 0 else "NONE")
    return (float(p_star), float(exec_star), side)


def _pick_trade_price_col(df: pd.DataFrame) -> Optional[str]:
    prefs = ["TradePrice", "LastPrice", "LastPx", "PriceLast", "TradedPrice"]
    for c in prefs:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    # heuristic fallback
    for c in df.columns:
        lc = c.lower()
        if "price" in lc and not (lc.startswith("bid") or lc.startswith("ask")):
            if pd.to_numeric(df[c], errors="coerce").notna().any():
                return c
    return None



# --- helper: compute auction for a given state code (4=open, 10=close) ---
def _compute_auction_price_in_state(
    df: pd.DataFrame,
    state_code: int,
    *,
    debug: bool=False,
    lookahead_rows: int = 200,
    time_col: str = "arrival_timestamp",
    lookahead_seconds: int = 180
) -> tuple[float, float]:
    if "OrderbookStateCode" not in df.columns:
        raise KeyError("Missing column 'OrderbookStateCode'")

    mask = (df["OrderbookStateCode"] == state_code)
    if not mask.any():
        return (np.nan, np.nan)

    idx_true = np.flatnonzero(mask.values)
    splits = np.where(np.diff(idx_true) > 1)[0] + 1
    last_run = np.split(idx_true, splits)[-1]
    start_pos, end_pos = int(last_run[0]), int(last_run[-1])

    block = df.iloc[start_pos:end_pos+1]

    # compute clearing price from quotes inside the block
    block_quotes = block.loc[block["EntryType"] == 1] if "EntryType" in block.columns else block
    computed = np.nan
    for _, row in block_quotes.iterrows():
        p, sz, side = auction_match_row(row, debug=False)
        if not np.isnan(p):
            computed = p

    # find reported price
    tp_col = _pick_trade_price_col(df)
    reported = np.nan
    if tp_col is not None:
        inblock = pd.to_numeric(block[tp_col], errors="coerce").dropna()
        if not inblock.empty:
            reported = float(inblock.iloc[-1])

        if np.isnan(reported):
            after_rows = pd.to_numeric(df.iloc[end_pos+1:end_pos+1+lookahead_rows][tp_col], errors="coerce").dropna()
            if not after_rows.empty:
                reported = float(after_rows.iloc[0])

        if np.isnan(reported) and time_col in df.columns:
            t0 = pd.to_datetime(df.iloc[end_pos][time_col], errors="coerce")
            if pd.notna(t0):
                sub = df.iloc[end_pos+1:].copy()
                sub[time_col] = pd.to_datetime(sub[time_col], errors="coerce")
                within = sub[(sub[time_col] >= t0) & (sub[time_col] <= t0 + pd.Timedelta(seconds=lookahead_seconds))]
                tp = pd.to_numeric(within[tp_col], errors="coerce").dropna()
                if not tp.empty:
                    reported = float(tp.iloc[0])

    if debug:
        print(f"[state={state_code}] tp_col={tp_col} computed={computed} reported={reported} "
              f"block_rows={len(block)} looked_ahead_rows={lookahead_rows} time_window={lookahead_seconds}s")

    return (computed, reported)


# --- path resolver: provide either a direct parquet_path or a pattern with {date} and {symbol} ---
def _resolve_parquet_path(date: Union[int, str], symbol: int, *, parquet_path: Optional[str], path_pattern: Optional[str]) -> str:
    if parquet_path is not None:
        return parquet_path
    if path_pattern is not None:
        try:
            return path_pattern.format(date=date, symbol=symbol)
        except KeyError as e:
            raise ValueError("path_pattern must contain '{date}' and '{symbol}'") from e
    raise ValueError("Provide either parquet_path=... or path_pattern='...{date}...{symbol}....'")

# --- main entrypoint you can call in your loop ---
def verify_auctions(
    date: Union[int, str],
    symbol: int,
    *,
    parquet_path: Optional[str] = None,
    path_pattern: Optional[str] = None,
    open_code: int = 4,
    close_code: int = 10,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Returns a dict containing:
      - open.match (bool), open.computed, open.reported
      - close.match (bool), close.computed, close.reported
      - verdict in {'both','open_only','close_only','none'}
    """
    path = _resolve_parquet_path(date, symbol, parquet_path=parquet_path, path_pattern=path_pattern)

    # Only load the columns your matcher needs.
    need_cols = set(COLUMNS) | {"EntryType", "OrderbookStateCode", "TradePrice"}
    df = pd.read_parquet(path)  # parquet 'columns=' can be used if your engine supports it
    missing = [c for c in ["OrderbookStateCode"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Compute open & close without debug first to check results
    open_comp,  open_rep  = _compute_auction_price_in_state(df, open_code,  debug=False)
    close_comp, close_rep = _compute_auction_price_in_state(df, close_code, debug=False)

    def ok(comp, rep) -> bool:
        return (pd.notna(comp) and pd.notna(rep) and np.isclose(float(comp), float(rep), rtol=0.0, atol=1e-6))

    open_ok  = ok(open_comp,  open_rep)
    close_ok = ok(close_comp, close_rep)
    
    # Only print debug info if requested AND one or both auctions failed
    if debug and not (open_ok and close_ok):
        print(f"Debug info for {date} {symbol} (open_ok={open_ok}, close_ok={close_ok}):")
        _compute_auction_price_in_state(df, open_code,  debug=True)
        _compute_auction_price_in_state(df, close_code, debug=True)

    if   open_ok and close_ok: verdict = "both"
    elif open_ok:              verdict = "open_only"
    elif close_ok:             verdict = "close_only"
    else:                      verdict = "none"

    return {
        "date": date,
        "symbol": symbol,
        "open":  {"computed": float(open_comp)  if pd.notna(open_comp)  else np.nan,
                  "reported": float(open_rep)   if pd.notna(open_rep)   else np.nan,
                  "match": bool(open_ok)},
        "close": {"computed": float(close_comp) if pd.notna(close_comp) else np.nan,
                  "reported": float(close_rep)  if pd.notna(close_rep)  else np.nan,
                  "match": bool(close_ok)},
        "verdict": verdict,
        "path": path,
    }

# Optional tiny boolean wrapper (True only if both auctions verify)
def verify_auctions_bool(date: Union[int, str], symbol: int, **kwargs) -> bool:
    r = verify_auctions(date, symbol, **kwargs)
    return r["open"]["match"] and r["close"]["match"]

# ======== CONFIG: edit this to your parquet layout ========
PATH_PATTERN = "20250811_4264.parquet"   # <-- change to your path pattern
# ==========================================================

dates = ['20250811']

symbols = [4264]

all_results = []
print("date      symbol   verdict        open(comp,rep,ok)        close(comp,rep,ok)")
print("-"*92)

for d in dates:
    for s in symbols:
        try:
            r = verify_auctions(
                d, s,
                path_pattern=PATH_PATTERN,
                open_code=4,   # open auction state
                close_code=10, # close auction state
                debug=True
            )
            oc = r["open"]["computed"];  orp = r["open"]["reported"];  ook = r["open"]["match"]
            cc = r["close"]["computed"]; crp = r["close"]["reported"]; cok = r["close"]["match"]
            print(f"{d}  {s:6d}  {r['verdict']:<12}  "
                  f"open({oc:.6g},{orp:.6g},{ook})  "
                  f"close({cc:.6g},{crp:.6g},{cok})")
            all_results.append(r)
        except Exception as e:
            print(f"{d}  {s:6d}  ERROR: {e}")

# Optional: quick summary counts
both  = sum(1 for r in all_results if r["verdict"] == "both")
oonly = sum(1 for r in all_results if r["verdict"] == "open_only")
conly = sum(1 for r in all_results if r["verdict"] == "close_only")
none  = sum(1 for r in all_results if r["verdict"] == "none")
print("\nSummary:", {"both": both, "open_only": oonly, "close_only": conly, "none": none})

# Optional: turn into a small DataFrame you can export
try:
    summary_rows = [{
        "date": r["date"],
        "symbol": r["symbol"],
        "open_computed": r["open"]["computed"],
        "open_reported": r["open"]["reported"],
        "open_match": r["open"]["match"],
        "close_computed": r["close"]["computed"],
        "close_reported": r["close"]["reported"],
        "close_match": r["close"]["match"],
        "verdict": r["verdict"],
        "path": r["path"],
    } for r in all_results]
    df_summary = pd.DataFrame(summary_rows)
    # df_summary.to_csv("auction_verification_summary.csv", index=False)
except Exception:
    pass

# This adds 3 new columns on auction rows only.
def add_auction_columns(df: pd.DataFrame, time_col: str = "arrival_timestamp") -> pd.DataFrame:
    # out = df.loc[:, COLUMNS].copy()
    out = df.copy()

    if time_col not in out.columns:
        raise KeyError(f"Missing time column '{time_col}'")
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
 
    # This only keeps quote updates of EntryType == 1.
    if "EntryType" in out.columns:
        out = out[out["EntryType"] == 1].copy()

    # This initializes new columns.
    out["auction_price"] = np.nan
    out["auction_size"]  = np.nan
    out["auction_side"]  = "NONE"

    # This runs the matcher only for rows where OrderbookSateCode = 4 (in the auction).
    idxs = out.index[out["OrderbookStateCode"] == 4]

    # This runs the matcher row-by-row on those indexed rows.
    for i in idxs:
        p, sz, side = auction_match_row(out.loc[i], debug=False)
        out.at[i, "auction_price"] = p
        out.at[i, "auction_size"] = sz
        out.at[i, "auction_side"] = side

    # This drops the first row.
    out = out.iloc[1:].copy()

    return out
