import re, math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# TICK SIZE:

# def get_tick_size(price):
#     if price < 10:
#         return 0.01
#     elif price < 25.00:
#         return 0.02
#     elif price < 50.00:
#         return 0.05
#     elif price < 100.00:
#         return 0.10
#     else:
#         return 0.20
        
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

# PBOOK:

def plot_order_book_table(df: pd.DataFrame, event_id, n: int = 1, max_cols_per_row: int = 5, fig_width: int = 2500):
    if not isinstance(df.index, pd.RangeIndex) or not df.index.is_unique:
        df = df.reset_index(drop=True)

    # --- config ---
    event_id_col = "event_id"
    bid_prefix, ask_prefix = "Bid", "Ask"
    price_field, size_field, orders_field = "Price", "Size", "Orders"
    bid_color, ask_color = "#2ecc71", "#e74c3c"
    bid_fill  = "rgba(46, 204, 113, 0.14)"
    ask_fill  = "rgba(231, 76, 60, 0.14)"
    ROW_HEIGHT = 360
    TOP_MARGIN_EXTRA = 120

    # --- locate starting row ---
    if event_id_col not in df.columns:
        raise ValueError(f"Expected '{event_id_col}' column in df.")
    if "EntryType" not in df.columns:
        raise ValueError("Expected 'EntryType' column in df.")

    row_df = df.loc[df[event_id_col] == event_id]
    if row_df.empty:
        raise ValueError(f"No rows found with {event_id_col} == {event_id!r}.")
    start_pos = int(df.index.get_indexer([row_df.index[0]])[0])
    # start_pos = int(row_df.index[0])

    # Build positions: only rows with EntryType == 1 starting at start_pos
    n = max(1, int(n))
    positions = []
    for i in range(start_pos, len(df)):
        et = df.iloc[i].get("EntryType")
        try:
            is_one = (pd.notna(et) and int(et) == 1)
        except Exception:
            is_one = False
        if is_one:
            positions.append(i)
            if len(positions) == n:
                break

    if not positions:
        raise ValueError("No snapshots with EntryType == 1 found at/after the given event_id.")

    # --- helpers ---
    def _extract_side(row: pd.Series, side_prefix: str):
        rx_price  = re.compile(rf"^{side_prefix}{price_field}(\d+)$",  re.IGNORECASE)
        rx_size   = re.compile(rf"^{side_prefix}{size_field}(\d+)$",   re.IGNORECASE)
        rx_orders = re.compile(rf"^{side_prefix}{orders_field}(\d+)$", re.IGNORECASE)
        prices, sizes, orders = {}, {}, {}
        for col, val in row.items():
            if pd.isna(val):
                continue
            m = rx_price.match(col);  m and prices.setdefault(int(m.group(1)), float(val))
            m = rx_size.match(col);   m and sizes.setdefault(int(m.group(1)),  float(val))
            m = rx_orders.match(col); m and orders.setdefault(int(m.group(1)), float(val))
        levels = sorted(prices.keys() & sizes.keys())
        if side_prefix.lower().startswith("bid"):
            levels.sort(key=lambda k: prices[k], reverse=True)
        else:
            levels.sort(key=lambda k: prices[k])
        return (
            [prices[k] for k in levels],
            [sizes[k]  for k in levels],
            [orders.get(k) for k in levels],
        )

    def _fmt_int(x):   return "" if x is None or pd.isna(x) else f"{int(round(float(x))):,}"
    def _fmt_size(x):  return "" if x is None or pd.isna(x) else (f"{int(x):,}" if abs(x-round(x))<1e-9 else f"{x:g}")
    def _fmt_price(x): return "" if x is None or pd.isna(x) else f"{float(x):g}"

    def _build_table_for_row(row: pd.Series):
        bid_p, bid_s, bid_o = _extract_side(row, bid_prefix)
        ask_p, ask_s, ask_o = _extract_side(row, ask_prefix)

        if not bid_p and not ask_p:
            values = [[""], [""], [""], [""], [""], [""]]
        else:
            r = max(len(bid_p), len(ask_p))
            col_bo, col_bs, col_bp, col_ap, col_as, col_ao = [], [], [], [], [], []
            for i in range(r):
                if i < len(bid_p):
                    col_bo.append(_fmt_int(bid_o[i])); col_bs.append(_fmt_size(bid_s[i])); col_bp.append(_fmt_price(bid_p[i]))
                else:
                    col_bo.append(""); col_bs.append(""); col_bp.append("")
                if i < len(ask_p):
                    col_ap.append(_fmt_price(ask_p[i])); col_as.append(_fmt_size(ask_s[i])); col_ao.append(_fmt_int(ask_o[i]))
                else:
                    col_ap.append(""); col_as.append(""); col_ao.append("")
            values = [col_bo, col_bs, col_bp, col_ap, col_as, col_ao]

        return go.Table(
            columnwidth=[0.9, 1.1, 1.1, 1.1, 1.1, 0.9],
            header=dict(
                values=["Bid Orders","Bid Size","Bid Price","Ask Price","Ask Size","Ask Orders"],
                align="center",
                fill_color=[bid_color, bid_color, bid_color, ask_color, ask_color, ask_color],
                font=dict(color="white", size=12),
                height=30,
            ),
            cells=dict(
                values=values,
                align=["right","right","right","left","right","right"],
                fill=dict(color=[
                    [bid_fill]*len(values[0]), [bid_fill]*len(values[1]), [bid_fill]*len(values[2]),
                    [ask_fill]*len(values[3]), [ask_fill]*len(values[4]), [ask_fill]*len(values[5]),
                ]),
                font=dict(
                    color=[
                        [bid_color]*len(values[0]), [bid_color]*len(values[1]), [bid_color]*len(values[2]),
                        [ask_color]*len(values[3]), [ask_color]*len(values[4]), [ask_color]*len(values[5]),
                    ],
                    size=12
                ),
                height=26,
            ),
        )

    # --- grid & titles ---
    cols = max_cols_per_row
    rows = math.ceil(len(positions) / cols)
    specs = [[{"type":"table"} for _ in range(cols)] for _ in range(rows)]

    titles = []
    for pos in positions:
        row = df.iloc[pos]
        base = (str(row["timestamp"]) if "timestamp" in df.columns and pd.notna(row.get("timestamp", None))
                else f"id={row[event_id_col]}")
        if "last_entrytype" in df.columns:
            le = row.get("last_entrytype")
            base = f"{base} — last_entrytype={'' if pd.isna(le) else le}"
        titles.append(base)
    titles += [""] * (rows*cols - len(titles))  # pad

    fig = make_subplots(
        rows=rows, cols=cols, specs=specs,
        horizontal_spacing=0.05, vertical_spacing=0.1,
        subplot_titles=tuple(titles),
    )

    for i, pos in enumerate(positions):
        r = i // cols + 1
        c = i % cols + 1
        fig.add_trace(_build_table_for_row(df.iloc[pos]), row=r, col=c)

    # --- fixed width; computed height ---
    fig_height = rows * ROW_HEIGHT + TOP_MARGIN_EXTRA
    fig.update_layout(
        title=f"Order Book Quotes snapshots starting at {event_id_col}={event_id}",
        autosize=False,
        width=fig_width,
        height=fig_height,
        margin=dict(l=10, r=10, t=90, b=10),
    )
    return fig


# This turns one wide snapshot row into ordered lists to render and compare reliably.
def _parse_side(row: pd.Series, side_prefix: str, price_field="Price", size_field="Size", orders_field="Orders"):
    """Parse one row into dicts per level + the display-ordered level list."""
    rx_price = re.compile(rf"^{side_prefix}{price_field}(\d+)$", re.IGNORECASE)
    rx_size = re.compile(rf"^{side_prefix}{size_field}(\d+)$", re.IGNORECASE)
    rx_orders = re.compile(rf"^{side_prefix}{orders_field}(\d+)$", re.IGNORECASE)
    prices, sizes, orders = {}, {}, {}
    for col, val in row.items():
        if pd.isna(val):
            continue
        m = rx_price.match(col);  m and prices.setdefault(int(m.group(1)), float(val))
        m = rx_size.match(col);   m and sizes.setdefault(int(m.group(1)),  float(val))
        m = rx_orders.match(col); m and orders.setdefault(int(m.group(1)), float(val))
    levels = sorted(prices.keys() & sizes.keys())
    if side_prefix.lower().startswith("bid"):
        levels.sort(key=lambda k: prices[k], reverse=True)
    else:
        levels.sort(key=lambda k: prices[k])
    return prices, sizes, orders, levels

# This returns true / false if flags say changed, and none if no relevant flags exist.
def _changed_from_flags(row: pd.Series, side: str, field: str, lvl: int):
    currentColumn = f"{side}{field}{lvl}"
    curr = row.get(currentColumn)

    # The previous column name.
    if field == "Price":
        previousColumn = f"{side}LevelPrevPrice{lvl}"
    elif field == "Size":
        previousColumn = f"{side}LevelPrevSize{lvl}"
    else: # Orders
        previousColumn = f"{side}LevelPrevOrders{lvl}"

    have_any_flag = False

    # This compares previous vs current if previous exists.
    if previousColumn in row.index and pd.notna(row[previousColumn]) and pd.notna(curr):
        have_any_flag = True
        try:
            return float(row[previousColumn]) != float(curr)
        except Exception:
            return row[previousColumn] != curr

    # Checking the Added / Removed (> 0 means a change).
    if field in ("Size","Orders"):
        addColumn = f"{side}{field}Added{lvl}"
        removeColumn = f"{side}{field}Removed{lvl}"
        for col in (addColumn, removeColumn):
            if col in row.index and pd.notna(row[col]):
                have_any_flag = True
                try:
                    if float(row[col]) > 0:
                        return True
                except Exception:
                    if bool(row[col]):
                        return True

    # If there is no relevant flags present.
    return None if not have_any_flag else False

# This finds the previous snapshot position before the given position.
def _prev_snapshot_pos(df, pos):
    for i in range(pos - 1, -1, -1):
        try:
            if int(df.at[i, "EntryType"]) == 1:
                return i
        except Exception:
            pass
    return None


def _cell_changed(row, previousRow, side, field, lvl):
    # This checks if the cell has flags indicating a change.
    byFlags = _changed_from_flags(row, side, field, lvl)
    if byFlags is not None:
        return byFlags
    # This compares to previous snapshot if flags absent.
    if previousRow is None:
        return False
    col = f"{side}{field}{lvl}"
    now = row.get(col)
    prev = previousRow.get(col)
    if pd.isna(now) and pd.isna(prev): 
        return False
    try:
        return float(now) != float(prev)
    except Exception:
        return now != prev

# This finds the next n snapshot positions starting at the given event_id.
def _find_snapshot_positions(df: pd.DataFrame, event_id, n: int):
    if not isinstance(df.index, pd.RangeIndex) or not df.index.is_unique:
        df = df.reset_index(drop=True)
    hits = df.index[df["event_id"] == event_id]
    if len(hits) == 0:
        raise ValueError(f"No rows found with event_id == {event_id!r}.")
    start_pos = int(hits[0])

    pos = []
    need = max(1, int(n))
    for i in range(start_pos, len(df)):
        try:
            if pd.notna(df.at[i, "EntryType"]) and int(df.at[i, "EntryType"]) == 1:
                pos.append(i)
                if len(pos) == need:
                    break
        except Exception:
            pass
    if not pos:
        raise ValueError("No snapshots with EntryType == 1 found at/after the given event_id.")
    return pos, df

# This plots the order book table with highlighted changes based on flags.
def plot_order_book_table_highlighted_flags(df: pd.DataFrame, event_id, n: int = 1, max_cols_per_row: int = 5, fig_width: int = 2500, changed_alpha: float = 0.65,):
    # This builds the base figure.
    figure = plot_order_book_table(df, event_id, n=n, max_cols_per_row=max_cols_per_row, fig_width=fig_width)

    positions, df = _find_snapshot_positions(df, event_id, n)

    # Similar to the ones above.
    bid_fill_base = "rgba(46, 204, 113, 0.14)"
    ask_fill_base = "rgba(231, 76,  60, 0.14)"
    bid_fill_changed = f"rgba(46, 204, 113, {changed_alpha})"
    ask_fill_changed = f"rgba(231, 76,  60, {changed_alpha})"

    # For each table trace, this computes per-cell fills from change flags on that row
    for traceIndex, pos in enumerate(positions):
        row = df.iloc[pos]

        # This finds the previous snapshot row (or None if none).
        prev_pos = _prev_snapshot_pos(df, pos)
        previousRow = df.iloc[prev_pos] if prev_pos is not None else None

        bid_p, bid_s, bid_o, bid_lvls = _parse_side(row, "Bid")
        ask_p, ask_s, ask_o, ask_lvls = _parse_side(row, "Ask")
        r = max(len(bid_lvls), len(ask_lvls))

        bo_fill, bs_fill, bp_fill = [], [], []
        ap_fill, as_fill, ao_fill = [], [], []

        for i in range(r):
            if i < len(bid_lvls):
                lvl = bid_lvls[i]
                bp_fill.append(bid_fill_changed if _cell_changed(row, previousRow, "Bid", "Price",  lvl) else bid_fill_base)
                bs_fill.append(bid_fill_changed if _cell_changed(row, previousRow, "Bid", "Size",   lvl) else bid_fill_base)
                bo_fill.append(bid_fill_changed if _cell_changed(row, previousRow, "Bid", "Orders", lvl) else bid_fill_base)
            else:
                bp_fill.append(bid_fill_base); bs_fill.append(bid_fill_base); bo_fill.append(bid_fill_base)

            if i < len(ask_lvls):
                lvl = ask_lvls[i]
                ap_fill.append(ask_fill_changed if _cell_changed(row, previousRow, "Ask", "Price",  lvl) else ask_fill_base)
                as_fill.append(ask_fill_changed if _cell_changed(row, previousRow, "Ask", "Size",   lvl) else ask_fill_base)
                ao_fill.append(ask_fill_changed if _cell_changed(row, previousRow, "Ask", "Orders", lvl) else ask_fill_base)
            else:
                ap_fill.append(ask_fill_base); as_fill.append(ask_fill_base); ao_fill.append(ask_fill_base)

        t = figure.data[traceIndex]
        t.cells.update(fill=dict(color=[bo_fill, bs_fill, bp_fill, ap_fill, as_fill, ao_fill]))

    return figure


def plot_recent_trades_table(df: pd.DataFrame, n: int = 50, newest_first: bool = True, up_to_event_id: int | None = None, up_to_index: int | None = None, event_id_col: str = "event_id", timestamp_candidates: tuple[str, ...] = ("timestamp", "arrival_timestamp"),):
    # This keeps only trades (EntryType == 4).
    etTrades = pd.to_numeric(df["EntryType"], errors="coerce") # Numeric filtering.
    trades = df.loc[etTrades.eq(4)].copy()

    cutDone = False

    # 1. Here, we take everything up to the snapshot row index.
    if up_to_index is not None:
        trades = trades.iloc[: up_to_index + 1]
        cutDone = True
        # head_df = df.iloc[: up_to_index + 1]
        # et = pd.to_numeric(head_df["EntryType"], errors="coerce")
        # trades = head_df.loc[et.eq(4)].copy()
        # cutDone = True

    # 2. If not, then by event_id, only if trades actually have event_id values.
    if not cutDone and up_to_event_id is not None and event_id_col in trades.columns:
        ev = pd.to_numeric(trades[event_id_col], errors="coerce")
        if ev.notna().any():
            trades = trades.loc[ev <= up_to_event_id]
            cutDone = True

    # 3. If not, then by time, we find the snapshot timestamp then keep trades up to that time.
    if not cutDone and up_to_event_id is not None:
        ts_col = next((c for c in timestamp_candidates if c in df.columns), None)
        if ts_col:
            # find the snapshot row’s timestamp
            snap = df.loc[pd.to_numeric(df[event_id_col], errors="coerce") == up_to_event_id]
            if not snap.empty:
                cutoff = pd.to_datetime(snap.iloc[0][ts_col])
                trades_ts = pd.to_datetime(trades[ts_col], errors="coerce")
                trades = trades.loc[trades_ts <= cutoff]
                cutDone = True


    if trades.empty:
        raise ValueError(
            f"No trades found up to this point "
            f"({'index' if up_to_index is not None else event_id_col} "
            f"= {up_to_index if up_to_index is not None else up_to_event_id})."
        )

    # This ensures the newest trades are at the top.
    if "timestamp" in trades.columns:
        trades = trades.sort_values("timestamp")
    elif "arrival_timestamp" in trades.columns:
        trades = trades.sort_values("arrival_timestamp")
    elif event_id_col in trades.columns:
        trades = trades.sort_values(event_id_col)
    else:
        trades = trades.sort_index()

    last_n = trades.tail(int(max(1, n)))
    if newest_first:
        last_n = last_n.iloc[::-1] # Flips the row order.

    # This reads the real trade fields in the dataset (price columns with data).
    if "TradePrice" in last_n.columns and last_n["TradePrice"].notna().any():
        price_col = "TradePrice"
    elif "LastTradePrice" in last_n.columns and last_n["LastTradePrice"].notna().any():
        price_col = "LastTradePrice"
    else:
        raise KeyError(
            "No price column with values found. Expected one of: TradePrice, LastTradePrice."
        )

    # TradeSide, 1 = sell -> red, -1 = buy -> green.
    if "shift_tradeside" not in last_n.columns:
        raise KeyError("TradeSide column is missing; cannot color trades.")
    ts = pd.to_numeric(last_n["shift_tradeside"], errors="coerce")
    tradeSide = ts.fillna(0).astype(int)

    def row_font(s):
        if s == 1: 
            return "#e74c3c"
        elif s == -1:
            return "#2ecc71"
        else:
            return "#2c3e50"
    
    def row_fill(s):
        if s == 1: 
            return "rgba(231,76,60,0.45)"
        elif s == -1:
            return "rgba(46,204,113,0.45)"
        else:
            return "rgba(127,140,141,0.12)"

    font_colors = [row_font(s) for s in tradeSide]
    fill_colors = [row_fill(s) for s in tradeSide]

    # Formatting the values.
    def fmt_qty(x):
        if pd.isna(x): return ""
        try:
            f = float(x)
            return f"{int(f):,}" if abs(f - round(f)) < 1e-9 else f"{f:g}"
        except Exception:
            return str(x)

    def fmt_price(x):
        if pd.isna(x): return ""
        try:
            return f"{float(x):g}"
        except Exception:
            return str(x)

    qty_vals   = [fmt_qty(v) for v in last_n.get("TradeQuantityTotal", [])]
    price_vals = [fmt_price(v) for v in (last_n[price_col] if price_col else pd.Series([""]*len(last_n), index=last_n.index))]

    table = go.Table(
        columnwidth=[1.2, 1.1],
        header=dict(
            values=["TradeQuantityTotal", price_col or "TradePrice"],
            align="center",
            fill_color="#2c3e50",
            font=dict(color="white", size=12),
            height=28,
        ),
        cells=dict(
            values=[qty_vals, price_vals],
            align=["right", "right"],
            height=26,
            fill_color=[fill_colors, fill_colors],
            font=dict(color=[font_colors, font_colors], size=12),
        ),
    )
    fig = go.Figure([table])
    fig.update_layout(
        title=f"Last {len(last_n)} Trades",
        title_x=0.5,
        margin=dict(l=10, r=10, t=58, b=10),
        width=700,
        height=max(220, 26*len(last_n) + 90),
    )
    return fig


# BBO AND VOLATILITY FIGURES:

# This calculates the BBO and volatility.
def plot_pbook_interactive(pbook: pd.DataFrame, window: int = 50) -> go.Figure:
    df = pbook.copy()
    df['arrival_timestamp'] = pd.to_datetime(df['arrival_timestamp'])

    # This filters to only keep data between 10:00 and 15:00.
    df = df[
        (df['arrival_timestamp'].dt.time >= pd.to_datetime("10:01").time()) &
        (df['arrival_timestamp'].dt.time <= pd.to_datetime("15:00").time())
    ].copy()

    df = df.replace(0, np.nan) # This turn 0 into NaN
    df = df.dropna(subset=["AskPrice0", "BidPrice0"]) # This drops the rows where either column is NaN.


    df = df.sort_values('arrival_timestamp')

    # This computess the mid-price.
    df['mid_price'] = (df["AskPrice0"] + df["BidPrice0"]) / 2

    # Rolling std as volatility.
    df['volatility'] = df['mid_price'].rolling(window=window).std()

    # Marks the Bid/Ask Changes.
    bid_chg = df['BidPrice0'].ne(df['BidPrice0'].shift())
    ask_chg = df['AskPrice0'].ne(df['AskPrice0'].shift())

    # A Two-row subplot with a shared x-axis.
    figure = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.5], # Top taller than bottom.
        subplot_titles=("BBO (Best Bid/Ask)", f"Volatility (rolling std, window={window})")
    )

    # This is Row 1: BBO.
    figure.add_trace(go.Scatter(
        x=df['arrival_timestamp'], y=df['BidPrice0'],
        mode='lines', name='BidPrice0', line=dict(color='green')
    ), row=1, col=1)
    figure.add_trace(go.Scatter(
        x=df['arrival_timestamp'], y=df['AskPrice0'],
        mode='lines', name='AskPrice0', line=dict(color='red')
    ), row=1, col=1)

    # This marks the changes.
    figure.add_trace(go.Scatter(
        x=df.loc[bid_chg, 'arrival_timestamp'], y=df.loc[bid_chg, 'BidPrice0'],
        mode='markers', name='Bid change', marker=dict(symbol='x', size=6, color='green')
    ), row=1, col=1)
    figure.add_trace(go.Scatter(
        x=df.loc[ask_chg, 'arrival_timestamp'], y=df.loc[ask_chg, 'AskPrice0'],
        mode='markers', name='Ask change', marker=dict(symbol='x', size=6, color='red')
    ), row=1, col=1)

    # This is Row 2: Volatility.
    figure.add_trace(go.Scatter(
        x=df['arrival_timestamp'], y=df['volatility'],
        mode='lines', name=f'Volatility ({window})', line=dict(color='blue', dash='dot')
    ), row=2, col=1)

    # Layout.
    figure.update_layout(
        title=f"BBO & Volatility for symbol {df.StockId.iloc[0]} on {df.arrival_timestamp.iloc[0].date()}",
        height=800,
        hovermode='x unified',
        showlegend=True,
        margin=dict(l=40, r=20, t=40, b=40)
    )
    figure.update_yaxes(title_text="Price", row=1, col=1)
    figure.update_yaxes(title_text="Volatility", row=2, col=1)
    figure.update_xaxes(title_text="Timestamp", tickformat="%H:%M")

    return figure

# MATCHING ALGORITHM:

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
