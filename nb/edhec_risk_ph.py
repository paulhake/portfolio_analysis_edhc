import pandas as pd
import numpy as np
import scipy.stats as sps
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import math
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

    def neg_sharpe_ratio(weights, cov, er, rfr):
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - rfr) / vol




    results = minimize(neg_sharpe_ratio, init_guess,
                       args=(cov, er, risk_free_rate), method='SLSQP',
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
    ind_capweights = ind_mktcap.div(total_mktcap, axis=0)

    # Calculate total market return as weighted sum
    total_market_return = (ind_capweights * ind_return).sum(axis=1)
    
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
    
def discount(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time period t
    and r is the per-period interest rate
    returns a |t| x |r| Series or DataFrame
    r can be a float, Series or DataFrame
    returns a DataFrame indexed by t
    """
    discounts = pd.DataFrame([(r+1)**-i for i in t])
    discounts.index = t
    return discounts

def pv(flows, r):
    """
    Compute the present value of a sequence of cash flows given by the time (as an index) and amounts
    r can be a scalar, or a Series or DataFrame with the number of rows matching the num of rows in flows
    """
    dates = flows.index
    discounts = discount(dates, r)
    return discounts.multiply(flows, axis=0).sum()

def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of a series of liabilities, based on an interest rate and current value of assets
    """
    return pv(assets, r)/pv(liabilities, r)

def inst_to_ann(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Convert an annual interest rate to an instantaneous interest rate
    """
    return np.log1p(r)

def cir(n_years = 10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None: r_0 = b 
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    ## For Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ####

    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    ####
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices


def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Returns the series of cash flows generated by a bond,
    indexed by the payment/coupon number
    """
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupons = np.repeat(coupon_amt, n_coupons)
    coupon_times = np.arange(1, n_coupons+1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows
    
def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    This is not designed to be efficient, rather,
    it is to illustrate the underlying principle behind bond pricing!
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date
    and the bond value is computed over time.
    i.e. The index of the discount_rate DataFrame is assumed to be the coupon number
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year,
                                      discount_rate.loc[t])
        return prices
    else: # base case ... single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)
    
def macaulay_duration(flows, discount_rate):
    """
    Computes the Macaulay Duration of a sequence of cash flows, given a per-period discount rate
    """
    discounted_flows = discount(flows.index, discount_rate)*pd.DataFrame(flows)
    weights = discounted_flows/discounted_flows.sum()
    return np.average(flows.index, weights=weights.iloc[:,0])

def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """
    Returns the weight W in cf_s that, along with (1-W) in cf_l will have an effective
    duration that matches cf_t
    """
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t)/(d_l - d_s)

def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Computes the total return of a Bond based on monthly bond prices and coupon payments
    Assumes that dividends (coupons) are paid out at the end of the period (e.g. end of 3 months for quarterly div)
    and that dividends are reinvested in the bond
    """
    coupons = pd.DataFrame(data = 0.0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
    coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
    total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_returns.dropna()

def bt_mix(r1, r2, allocator, **kwargs):
    """
    Runs a back test (simulation) of allocating between a two sets of returns
    r1 and r2 are T x N DataFrames or returns where T is the time step index and N is the number of scenarios.
    allocator is a function that takes two sets of returns and allocator specific parameters, and produces
    an allocation to the first portfolio (the rest of the money is invested in the GHP) as a T x 1 DataFrame
    Returns a T x N DataFrame of the resulting N portfolio scenarios
    """
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 should have the same shape")
    weights = allocator(r1, r2, **kwargs)
    if not weights.shape == r1.shape:
        raise ValueError("Allocator returned weights with a different shape than the returns")
    r_mix = weights*r1 + (1-weights)*r2
    return r_mix


def fixedmix_allocator(r1, r2, w1, **kwargs):
    """
    Produces a time series over T steps of allocations between the PSP and GHP across N scenarios
    PSP and GHP are T x N DataFrames that represent the returns of the PSP and GHP such that:
     each column is a scenario
     each row is the price for a timestep
    Returns an T x N DataFrame of PSP Weights
    """
    return pd.DataFrame(data = w1, index=r1.index, columns=r1.columns)

def terminal_values(rets):
    """
    Computes the terminal values from a set of returns supplied as a T x N DataFrame
    Return a Series of length N indexed by the columns of rets
    """
    return (rets+1).prod()

def terminal_stats(rets, floor = 0.8, cap=np.inf, name="Stats"):
    """
    Produce Summary Statistics on the terminal values per invested dollar
    across a range of N scenarios
    rets is a T x N DataFrame of returns, where T is the time-step (we assume rets is sorted by time)
    Returns a 1 column DataFrame of Summary Stats indexed by the stat name 
    """
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = reach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (-cap+terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        "mean": terminal_wealth.mean(),
        "std" : terminal_wealth.std(),
        "p_breach": p_breach,
        "e_short":e_short,
        "p_reach": p_reach,
        "e_surplus": e_surplus
    }, orient="index", columns=[name])
    return sum_stats

def glidepath_allocator(r1, r2, start_glide=1, end_glide=0.0):
    """
    Allocates weights to r1 starting at start_glide and ends at end_glide
    by gradually moving from start_glide to end_glide over time
    """
    n_points = r1.shape[0]
    n_col = r1.shape[1]
    path = pd.Series(data=np.linspace(start_glide, end_glide, num=n_points))
    paths = pd.concat([path]*n_col, axis=1)
    paths.index = r1.index
    paths.columns = r1.columns
    return paths

def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC Prices must have the same shape")
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] ## PV of Floor assuming today's rates and flat YC
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history


def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = (1-maxdd)*peak_value ### Floor is based on Prev Peak
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value and prev peak at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        peak_value = np.maximum(peak_value, account_value)
        w_history.iloc[step] = psp_w
    return w_history

