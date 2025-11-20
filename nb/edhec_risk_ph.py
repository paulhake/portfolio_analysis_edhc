import pandas as pd
import numpy as np
import scipy.stats as sps
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import yfinance as yf
import ipywidgets as widgets


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

def get_ind_sizes() -> pd.DataFrame:
    """
    Load and preprocess the Industry Sizes dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame containing monthly market capitalizations for various industry portfolios.

    Raises
    ------
    FileNotFoundError
        If the data file does not exist.
    """
    file_path = '../data/ind30_m_size.csv'

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    ind = pd.read_csv(file_path, index_col=0)
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()  # Remove any leading/trailing spaces in column names
    return ind

def get_ind_nfirms() -> pd.DataFrame:
    """
    Load and preprocess the Industry Number of Firms dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame containing monthly number of firms for various industry portfolios.

    Raises
    ------
    FileNotFoundError
        If the data file does not exist.
    """
    file_path = '../data/ind30_m_nfirms.csv'

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    ind = pd.read_csv(file_path, index_col=0)
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

def plot_ef(n_points, er, cov, style=".-",show_cml=False, risk_free_rate=0,show_ew=False, show_gmv=False):
    """
    Plots the efficient frontier given expected returns and covariance matrix.

    Parameters:
    -----------
    n_points : int
        Number of points to plot on the efficient frontier.
    er : pd.Series
        Series of expected returns for each asset.
    cov : pd.DataFrame
        Covariance matrix of asset returns.
    style : str
        Matplotlib style string for the plot.

    Returns:
    --------
    matplotlib axes object
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    vols = []
    rets = []
    for target_r in target_rs:
        weights = minimize_vol(target_r, er, cov)
        vol = portfolio_vol(weights, cov)
        vols.append(vol)
        rets.append(target_r)
    plt.plot(vols, rets, style)
    plt.xlabel("Volatility")
    plt.ylabel("Return")
    plt.xlim(left=0)
    if show_cml:
        # Calculate the tangency portfolio
        t_weights = msr(risk_free_rate, er, cov)
        t_ret = portfolio_return(t_weights, er)
        t_vol = portfolio_vol(t_weights, cov)
        # CML line
        cml_x = [0, t_vol]
        cml_y = [risk_free_rate, t_ret]
        plt.plot(cml_x, cml_y, color='green', linestyle='--', label='CML')
        plt.plot(t_vol, t_ret, 'r*', markersize=15.0)
        plt.annotate("MSR", xy=(t_vol, t_ret),
            xytext=(20, -10), textcoords='offset points',
            fontsize=12, color='red')
        plt.legend()
    if show_ew:
        n = er.shape[0]
        ew_weights = np.repeat(1/n, n)
        ew_ret = portfolio_return(ew_weights, er)
        ew_vol = portfolio_vol(ew_weights, cov)
        plt.plot(ew_vol, ew_ret, 'b*', markersize=12.0)
        plt.annotate("EW", xy=(ew_vol, ew_ret),
            xytext=(-10, 10), textcoords='offset points',
            fontsize=12, color='blue')
    if show_gmv:
        n = er.shape[0]
        init_guess = np.repeat(1/n, n)
        bounds = ((0.0, 1.0),) * n
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1}
        results = minimize(portfolio_vol, init_guess,
                           args=(cov,), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1),
                           bounds=bounds)
        gmv_weights = results.x
        gmv_ret = portfolio_return(gmv_weights, er)
        gmv_vol = portfolio_vol(gmv_weights, cov)
        plt.plot(gmv_vol, gmv_ret, 'm*', markersize=12.0)
        plt.annotate("GMV", xy=(gmv_vol, gmv_ret),
            xytext=(10, 5), textcoords='offset points',
            fontsize=12, color='magenta')


    return plt.gca()


def minimize_vol(target_return, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights, er)}
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1}
    results = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(return_is_target,
                                    weights_sum_to_1),
                       bounds=bounds)
    return results.x

def msr(risk_free_rate, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1}

    def neg_sharpe_ratio(weights, cov,er, risk_free_rate=risk_free_rate):
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - risk_free_rate) / vol




    results = minimize(neg_sharpe_ratio, init_guess,
                       args=(cov,er, risk_free_rate), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1),
                       bounds=bounds)
    return results.x

def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

def get_total_market_index_returns():
    """
    Calculate total market index returns based on cap-weighted industry returns.
    
    Returns
    -------
    DataFrame
        DataFrame containing total market returns with date index
    """
    # Load all required data
    ind_return = get_ind_returns()
    ind_nfirm = get_ind_nfirms()
    ind_size = get_ind_sizes()
    
    # Calculate market capitalization for each industry
    ind_mktcap = ind_nfirm * ind_size
    
    # Calculate total market cap across all industries
    total_mktcap = ind_mktcap.sum(axis=1)
    
    # Calculate cap weights for each industry
    ind_capweights = ind_mktcap.div(total_mktcap, axis='rows')
    
    # Calculate total market return as weighted sum
    total_market_return = (ind_capweights * ind_return).sum(axis='columns')
    
    # Return as DataFrame with proper column name
    return pd.DataFrame({'TotalMarketReturn': total_market_return})

def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8,riskfree_rate=0.03, drawdown=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, and Risky Weight History
    """
    # Set up the safe asset
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r[:] = riskfree_rate / 12
    
    # Set up some parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = start

    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["R"])
    
    # Set up DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown)
        cushion = (account_value - floor_value) / account_value
        risky_w = m * cushion
        risky_w = np.minimum(risky_w, 1)  # cap risky weight at 100%
        risky_w = np.maximum(risky_w, 0)  # floor risky weight at 0%
        safe_w = 1 - risky_w
        risky_alloc = account_value * risky_w
        safe_alloc = account_value * safe_w
        
        # recompute the account value based on the returns
        account_value = risky_alloc * (1 + risky_r.iloc[step]) + safe_alloc * (1 + safe_r.iloc[step])
        
        # store history
        account_history.iloc[step] = account_value
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w

    risky_wealth = start * (1 + risky_r).cumprod()

    return {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Weight": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r": risky_r,
        "safe_r": safe_r
    }



def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })

def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val

def show_cppi(n_scenarios=50, mu=0.07, sigma=0.15, m=3, floor=0., riskfree_rate=0.03, y_max=100):
    """
    Plot the results of a Monte Carlo Simulation of CPPI
    """
    start = 100
    sim_rets = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, prices=False, steps_per_year=12)
    risky_r = pd.DataFrame(sim_rets)
    # run the "back"-test
    btr = run_cppi(risky_r=pd.DataFrame(risky_r),riskfree_rate=riskfree_rate,m=m, start=start, floor=floor)
    wealth = btr["Wealth"]
    y_max=wealth.values.max()*y_max/100
    ax = wealth.plot(legend=False, alpha=0.3, color="indianred", figsize=(12, 6))
    ax.axhline(y=start, ls=":", color="black")
    ax.axhline(y=start*floor, ls="--", color="red")
    ax.set_ylim(top=y_max)

    cppi_controls = widgets.interactive(show_cppi, 
                                   n_scenarios=widgets.IntSlider(min=1, max=1000, step=5, value=50), 
                                   mu=(0., +.2, .01),
                                   sigma=(0, .30, .05),
                                   floor=(0, 2, .1),
                                   m=(1, 5, .5),
                                   riskfree_rate=(0, .05, .01),
                                   y_max=widgets.IntSlider(min=0, max=100, step=1, value=100,
                                                          description="Zoom Y Axis")
                                                          
                                        )