import pandas as pd
import yfinance as yf


def fetch_price_data(symbol: str, start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    """
    Fetch OHLCV data using yfinance.
    Example symbol for Nifty 50: ^NSEI
    """
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for symbol: {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index().rename(columns={"Adj Close": "Adj_Close"})
    if "Close" not in df.columns:
        raise ValueError("yfinance data must include a 'Close' column.")
    return df


def ema_crossover_signals(df: pd.DataFrame, fast: int = 20, slow: int = 100) -> pd.DataFrame:
    """
    Adds EMA columns and signal/position columns.
    signal: 1 for long, 0 for flat. position: 1 when in trade, else 0.
    """
    df = df.copy()
    df["EMA_fast"] = df["Close"].ewm(span=fast, adjust=False).mean()
    df["EMA_slow"] = df["Close"].ewm(span=slow, adjust=False).mean()

    df["signal"] = 0
    df.loc[df["EMA_fast"] > df["EMA_slow"], "signal"] = 1
    df["position"] = df["signal"].shift(1).fillna(0)
    return df


def apply_trend_filter(df: pd.DataFrame, filter_span: int = 200) -> pd.DataFrame:
    """
    Only allow long positions when Close is above long-term EMA.
    """
    df = df.copy()
    df["EMA_filter"] = df["Close"].ewm(span=filter_span, adjust=False).mean()
    df["trend_ok"] = df["Close"].to_numpy() > df["EMA_filter"].to_numpy()
    return df


def apply_trailing_stop(df: pd.DataFrame, stop_pct: float = 0.01) -> pd.Series:
    """
    Apply trailing stop to a long-only position series.
    Returns adjusted position series (1 or 0).
    """
    position = df["position"].copy()
    close = df["Close"].values

    in_trade = False
    peak = 0.0

    for i in range(len(df)):
        if position.iloc[i] == 1 and not in_trade:
            in_trade = True
            peak = close[i]
        if in_trade:
            if close[i] > peak:
                peak = close[i]
            if close[i] <= peak * (1 - stop_pct):
                position.iloc[i] = 0
                in_trade = False

        if position.iloc[i] == 0:
            in_trade = False

    return position


def backtest_ema_20_100_trail_1pct(
    df: pd.DataFrame,
    initial_capital: float = 100000.0,
    trend_filter_span: int = 200,
    trailing_stop_pct: float = 0.01,
) -> pd.DataFrame:
    """
    EMA 20/100 crossover + 200 EMA filter + 1% trailing stop.
    """
    df = ema_crossover_signals(df, fast=20, slow=100)

    df = apply_trend_filter(df, filter_span=trend_filter_span)
    df.loc[~df["trend_ok"], "position"] = 0

    df["position"] = apply_trailing_stop(df, stop_pct=trailing_stop_pct)

    df["returns"] = df["Close"].pct_change().fillna(0)
    df["strategy_returns"] = df["position"] * df["returns"]

    df["equity_curve"] = (1 + df["strategy_returns"]).cumprod()
    df["equity_value"] = df["equity_curve"] * initial_capital
    return df


if __name__ == "__main__":
    symbol = "^NSEI"
    start_date = "2022-01-01"
    initial_capital = 100000.0

    data = fetch_price_data(symbol, start=start_date)
    result = backtest_ema_20_100_trail_1pct(data, initial_capital=initial_capital)

    final_value = float(result["equity_value"].iloc[-1])
    total_return = float(result["equity_curve"].iloc[-1] - 1)

    print(f"Symbol: {symbol}")
    print(f"Backtest from {result['Date'].iloc[0].date()} to {result['Date'].iloc[-1].date()}")
    print(f"Initial capital: {initial_capital:.2f}")
    print("Strategy: EMA 20/100 + 200 EMA filter + 1% trailing stop")
    print(f"Final value: {final_value:.2f}")
    print(f"Total return: {total_return:.4f}")
