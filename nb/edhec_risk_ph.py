import pandas as pd
import numpy as np
import scipy.stats as sps
import os
import yfinance as yf


def get_from_yahoo(tickers, period='10y'):
    """
    Download historical price data from Yahoo Finance.

    Parameters:
    -----------
    tickers : list
        List of ticker symbols to download.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns:
    --------
    DataFrame with historical price data.
    """
    
    data = yf.download(tickers, period=period, auto_adjust=True)[['Close','Volume']]
    return data

def calculate_daily_returns(prices_df, column='Close'):
    """
    Calculate daily returns from price data.
    
    Parameters:
    -----------
    prices_df : DataFrame
        DataFrame with price data (can be multi-level columns from yfinance)
    column : str
        The price column to use ('Close', 'Adj Close', etc.)
    
    Returns:
    --------
    DataFrame of daily returns
    """
    if isinstance(prices_df.columns, pd.MultiIndex):
        # For multi-level columns (like yfinance with multiple tickers)
        prices = prices_df[column]
    else:
        # For single-level columns
        prices = prices_df
    
    return prices.pct_change().dropna()


def drawdown(rets: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the drawdowns of a return series.

    A drawdown is the peak-to-trough decline during a specific period,
    expressed as a percentage from the highest point to the current value.

    Parameters
    ----------
    rets : pd.DataFrame
        DataFrame containing return series (not cumulative returns).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for Wealth, Peaks, and Drawdown for each asset.
    """
    wealth_index = (1 + rets).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks

    result = pd.concat([wealth_index, previous_peaks, drawdowns], axis=1, keys=['Wealth', 'Peaks', 'Drawdown'])
    return result

def get_ffme_returns() -> pd.DataFrame:
    """
    Load and preprocess the Fama-French ME (Market Equity) returns dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame containing monthly returns for SmallCap and LargeCap portfolios.

    Raises
    ------
    FileNotFoundError
        If the data file does not exist.
    """
    file_path = '../data/Portfolios_Formed_on_ME_monthly_EW.csv'

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    me_m = pd.read_csv(file_path, header=0, parse_dates=True, index_col=0, na_values=-99.99)
    rets = me_m[['Lo 20', 'Hi 20']] / 100
    rets.columns = ['SmallCap', 'LargeCap']
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period("M")
    return rets

def get_hfi_returns() -> pd.DataFrame:
    """
    Load and preprocess the Hedge Fund Index (HFI) returns dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame containing monthly returns for the HFI index.

    Raises
    ------
    FileNotFoundError
        If the data file does not exist.
    """
    file_path = '../data/edhec-hedgefundindices.csv'

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    hfi = pd.read_csv(file_path, header=0, parse_dates=True, index_col=0)
    hfi.index = pd.to_datetime(hfi.index, format="%Y-%m").to_period("M")
    hfi = hfi / 100
    return hfi

def get_ind_returns() -> pd.DataFrame:
    """
    Load and preprocess the Industry Returns dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame containing monthly returns for various industry portfolios.

    Raises
    ------
    FileNotFoundError
        If the data file does not exist.
    """
    file_path = '../data/ind30_m_vw_rets.csv'

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    ind = pd.read_csv(file_path, index_col=0) / 100
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()  # Remove any leading/trailing spaces in column names
    return ind

def skewness(rets: pd.Series) -> float:
    """
    Calculate the skewness of a return series.

    Parameters
    ----------
    rets : pd.Series
        Series containing return data.

    Returns
    -------
    float
        Skewness of the return series.
    """
    demeaned_rets = rets - rets.mean()
    sigma = rets.std(ddof=0)
    exp = (demeaned_rets ** 3).mean()
    return exp / sigma**3

def kurtosis(rets: pd.Series) -> float:
    """
    Calculate the kurtosis of a return series.

    Parameters
    ----------
    rets : pd.Series
        Series containing return data.

    Returns
    -------
    float
        Kurtosis of the return series.
    """
    demeaned_rets = rets - rets.mean()
    sigma = rets.std(ddof=0)
    exp = (demeaned_rets ** 4).mean()
    return exp / sigma**4

def var_historic(r, level=5):
    """Returns the historic VaR at a specified level"""
    if isinstance(r, pd.DataFrame):
        return r.aggregate(lambda x: var_historic(x, level=level))
        #return r.aggregate(var_historic(r, level=level))
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")

def semi_deviation(r: pd.Series) -> float:
    """
    Calculate the semi-deviation (downside deviation) of a return series.

    Parameters
    ----------
    r : pd.Series
        Series containing return data.

    Returns
    -------
    float
        Semi-deviation of the return series.
    """
    negative_returns = r[r < 0]
    return negative_returns.std(ddof=0)

def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not.
    Test is applied at the 1% level by default.
    Returns True if the hypothesis of normality is accepted, False otherwise.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = sps.jarque_bera(r)
        return p_value > level
    
def var_gaussian(r: pd.Series, level=5, modified = False) -> float:
    """
    Calculate the Parametric Gaussian VaR of a return series.

    Parameters
    ----------
    r : pd.Series
        Series containing return data.
    level : float, optional
        The confidence level for VaR calculation (default is 5).

    Returns
    -------
    float
        The Parametric Gaussian VaR of the return series.
    """
    z = sps.norm.ppf(level / 100)
    if modified:
        S = skewness(r)
        K = kurtosis(r)
        z = (z +
             (z**2 - 1) * S / 6 +
             (z**3 - 3 * z) * (K - 3) / 24 -
             (2 * z**3 - 5 * z) * (S ** 2) / 36)



    return -(r.mean() + z * r.std(ddof=0))

def cvar_historic(r: pd.Series, level=5) -> float:
    """
    Calculate the Conditional VaR (Expected Shortfall) of a return series.

    Parameters
    ----------
    r : pd.Series
        Series containing return data.
    level : float, optional
        The confidence level for CVaR calculation (default is 5).

    Returns
    -------
    float
        The Conditional VaR of the return series.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(lambda x: cvar_historic(x, level=level))
    elif isinstance(r, pd.Series):
        var_level = var_historic(r, level=level)
        is_beyond_var = r <= -var_level
        return -r[is_beyond_var].mean()
    else:
        raise TypeError("Expected r to be Series or DataFrame")

def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns.

    Parameters
    ----------
    r : pd.Series or pd.DataFrame
        Series or DataFrame containing return data.
    periods_per_year : int
        Number of periods per year (e.g., 12 for monthly, 252 for daily).

    Returns
    -------
    float or pd.Series
        Annualized returns.
    """
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compounded_growth ** (periods_per_year / n_periods) - 1

def annualize_vol(r, periods_per_year):
    """
    Annualizes the volatility of a set of returns.

    Parameters
    ----------
    r : pd.Series or pd.DataFrame
        Series or DataFrame containing return data.
    periods_per_year : int
        Number of periods per year (e.g., 12 for monthly, 252 for daily).

    Returns
    -------
    float or pd.Series
        Annualized volatility.
    """
    return r.std() * np.sqrt(periods_per_year)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Calculate the annualized Sharpe ratio of a return series.

    Parameters
    ----------
    r : pd.Series or pd.DataFrame
        Series or DataFrame containing return data.
    riskfree_rate : float
        The risk-free rate as a decimal (e.g., 0.03 for 3%).
    periods_per_year : int
        Number of periods per year (e.g., 12 for monthly, 252 for daily).
    Returns
    -------
    float or pd.Series
        Annualized Sharpe ratio.
    """
    # Convert risk-free rate to per period
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_excess_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_excess_ret / ann_vol

def portfolio_return(weights, returns):
    return weights.T @ returns

def portfolio_vol(weights, cov):
    return (weights.T @ cov @ weights) ** 0.5

def plot_ef2(n_points, er, cov, style='.-'):
    """
    Plots the 2-asset efficient frontier.

    Parameters
    ----------
    n_points : int
        Number of points to plot on the efficient frontier.
    er : pd.Series
        Expected returns for the two assets.
    cov : pd.DataFrame
        Covariance matrix for the two assets.
    style : str, optional
        Matplotlib style string for the plot (default is '.-').

    Returns
    -------
    None
    """
    weights = [np.array([w, 1 - w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]

    import matplotlib.pyplot as plt
    plt.plot(vols, rets, style)
    plt.xlabel("Volatility")
    plt.ylabel("Return")
    plt.title("2-Asset Efficient Frontier")
    plt.show()