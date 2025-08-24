import re, math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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