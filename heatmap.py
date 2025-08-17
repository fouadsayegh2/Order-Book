import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# We build a two-tone heat map (asks red, bids green) where Y is price and darkness reflects size.
def orderbook_heatmap(df: pd.DataFrame, levels: int = 10, time_col: str = "arrival_timestamp", ask_prefix: tuple[str, str] = ("AskPrice", "AskSize"), bid_prefix: tuple[str, str] = ("BidPrice", "BidSize"), max_time_points: int | None = 800, zmax_percentile: float = 95.0, tick_size: float | None = None,):
    if time_col not in df.columns:
        if "timestamp" in df.columns:
            time_col = "timestamp"
        else:
            raise KeyError(f"Neither '{time_col}' nor 'timestamp' found in df.")
    data = df.copy()
    data = data.sort_values(time_col).reset_index(drop=True) # Orders the x-axis by time.

    # This collects the price / size columns for the levels.
    def existing_cols(prefix):
        cols = []
        for i in range(levels):
            c = f"{prefix}{i}"
            if c in data.columns:
                cols.append(c)
        return cols

    ask_price_cols = existing_cols(ask_prefix[0])
    ask_size_cols  = existing_cols(ask_prefix[1])
    bid_price_cols = existing_cols(bid_prefix[0])
    bid_size_cols  = existing_cols(bid_prefix[1])

    if not ask_price_cols and not bid_price_cols:
        raise ValueError("No Bid/Ask price columns found.")

    # This ensures that we have the same number of ask and bid levels.
    ask_n = min(len(ask_price_cols), len(ask_size_cols))
    bid_n = min(len(bid_price_cols), len(bid_size_cols))
    ask_price_cols, ask_size_cols = ask_price_cols[:ask_n], ask_size_cols[:ask_n]
    bid_price_cols, bid_size_cols = bid_price_cols[:bid_n], bid_size_cols[:bid_n]

    # A time downsample which helps for speed.
    if max_time_points and len(data) > max_time_points:
        step = max(1, len(data) // max_time_points)
        data = data.iloc[::step].reset_index(drop=True)

    # Ensures the x-axis is a datetime type.
    times = pd.to_datetime(data[time_col], errors="coerce").astype("datetime64[ns]")

    # This gathers all visible prices for the grind (y-axis).
    all_prices = []
    for cols in (ask_price_cols, bid_price_cols):
        for c in cols:
            # Ensure everything is converted to numbers.
            all_prices.append(pd.to_numeric(data[c], errors="coerce").values)

    if not all_prices:
        raise ValueError("No prices found to build a price grid.")
    all_prices = np.concatenate(all_prices) # Concatenates all price arrays.
    all_prices = all_prices[np.isfinite(all_prices)] # Remove NaNs and invalids.
    if all_prices.size == 0:
        raise ValueError("All prices are NaN/invalid.")

    def get_tick_size(price):
        if price <= 24.99: return 0.01
        elif price <= 49.98: return 0.02
        elif price <= 99.95: return 0.05
        elif price <= 249.90: return 0.10
        elif price <= 499.80: return 0.20
        else: return 0.50

    if tick_size is None:
        mid_price = np.median(all_prices)
        tick_size = get_tick_size(mid_price)

    pmin = float(np.min(all_prices))
    pmax = float(np.max(all_prices))
    
    # Ensures the grid is centered around the min / max prices with a margin.
    grid = np.arange(pmin - tick_size, pmax + 2*tick_size, tick_size)

    # Here, we prepare the Z matrices (order size (color intensity values)) for the asks and bids.
    ZAsk = np.full((grid.size, len(data)), np.nan, dtype=float)
    ZBid = np.full((grid.size, len(data)), np.nan, dtype=float)

    # This converts a price to a bin index based on the grid and tick size.
    def to_bin_idx(p):
        return int(round((p - grid[0]) / tick_size))

    # We fill Z, and iterate over the time (columns).
    for timeIndex in range(len(data)):
        # For asks.
        for pc, sc in zip(ask_price_cols, ask_size_cols):
            p = data.at[timeIndex, pc] # Price at that level and time.
            s = data.at[timeIndex, sc] # Size at that level and time.
            p = pd.to_numeric(p, errors="coerce")
            s = pd.to_numeric(s, errors="coerce")
            if pd.notna(p) and pd.notna(s):
                bi = to_bin_idx(float(p))
                if 0 <= bi < grid.size:
                    # This accumulates sizes if multiple levels share the same bin (Makes the heatmap color darker).
                    ZAsk[bi, timeIndex] = (0 if np.isnan(ZAsk[bi, timeIndex]) else ZAsk[bi, timeIndex]) + float(s)

        # For bids.
        for pc, sc in zip(bid_price_cols, bid_size_cols):
            p = data.at[timeIndex, pc]
            s = data.at[timeIndex, sc]
            p = pd.to_numeric(p, errors="coerce")
            s = pd.to_numeric(s, errors="coerce")
            if pd.notna(p) and pd.notna(s):
                bi = to_bin_idx(float(p))
                if 0 <= bi < grid.size:
                    ZBid[bi, timeIndex] = (0 if np.isnan(ZBid[bi, timeIndex]) else ZBid[bi, timeIndex]) + float(s)

    # This chooses zmax to avoid outliers.
    def robust_zmax(Z):
        # Keeps only valid numbers.
        vals = Z[np.isfinite(Z)]
        if vals.size == 0:
            return 1.0
        return float(np.percentile(vals, zmax_percentile))

    zmax_ask = robust_zmax(ZAsk)
    zmax_bid = robust_zmax(ZBid)

    # This builds the figure (two heatmaps overlaid).
    fig = go.Figure()

    # For the ask heatmap (Reds).
    fig.add_trace(go.Heatmap(
        x=times,
        y=grid,
        z=ZAsk,
        colorscale="Reds",
        zmin=0, zmax=zmax_ask,
        showscale=False,
        hovertemplate="Time: %{x}<br>Price: %{y}<br>Ask Size: %{z}<extra></extra>",
        name="Ask size"
    ))

    # For the bid heatmap (Greens).
    fig.add_trace(go.Heatmap(
        x=times,
        y=grid,
        z=ZBid,
        colorscale="Greens",
        zmin=0, zmax=zmax_bid,
        showscale=False,
        hovertemplate="Time: %{x}<br>Price: %{y}<br>Bid Size: %{z}<extra></extra>",
        name="Bid size"
    ))

    fig.update_layout(
        title="Order Book Heatmap (Ask=Red, Bid=Green)",
        title_x=0.5,
        xaxis_title="Time",
        yaxis_title="Price",
        margin=dict(l=10, r=10, t=60, b=10),
    )

    # Ensures the x-axis is a date / time type.
    fig.update_xaxes(type="date")
    return fig
