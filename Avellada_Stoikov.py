import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd


def price_fluctuation(S_0, N, T, sigma, mean_reversion=0, long_term_mean=None):
    """
    Simulates price fluctuations with optional mean reversion (Ornstein-Uhlenbeck process)
    """
    dt = T / N
    if mean_reversion > 0 and long_term_mean is not None:
        # Ornstein-Uhlenbeck process for mean reversion
        S = np.zeros(N + 1)
        S[0] = S_0
        for i in range(1, N + 1):
            dS = mean_reversion * (long_term_mean - S[i-1]) * dt + sigma * np.sqrt(dt) * ss.norm.rvs()
            S[i] = S[i-1] + dS
        return S
    else:
        # Standard Brownian motion
        b_increments = ss.norm.rvs(size=N, scale=np.sqrt(dt))
        b_t = np.cumsum(b_increments)
        b_t = np.insert(b_t, 0, 0)
        S = S_0 + sigma * b_t
        return S


def optimal_spread(gamma, sigma, k, T, t, q=0, q_max=None, market_vol_adjustment=0):
    """
    Calculate optimal spread based on Avellaneda-Stoikov model
    With adjustments for inventory position and market volatility
    """
    base_spread = 2 / gamma * np.log(1 + gamma / k) + gamma * sigma**2 * (T - t)
    
    # Inventory risk adjustment - increase spread as inventory approaches limits
    inventory_risk_factor = 0
    if q_max is not None and q_max > 0:
        inventory_risk_factor = 0.5 * (q / q_max)**2
    
    # Market volatility adjustment
    vol_adjustment = market_vol_adjustment * sigma
    
    total_spread = base_spread * (1 + inventory_risk_factor + vol_adjustment)
    return total_spread / 2


def reservation_price(mid_price, q, gamma, sigma, T, t, q_max=None):
    """
    Calculate the reservation price with enhanced inventory risk management
    """
    # Basic Avellaneda-Stoikov reservation price
    r = mid_price - q * gamma * sigma**2 * (T - t)
    
    # Additional inventory risk penalty as position approaches limits
    if q_max is not None and q_max > 0:
        inventory_risk = gamma * (q / q_max)**2 * sigma**2 * (T - t)
        r -= np.sign(q) * inventory_risk
    
    return r


def trading_intensity(
    mid_price, sigma, gamma, k, T, t, q_current, lambda_ask, lambda_bid, delta, dt,
    q_max=None, trans_cost=0, market_impact=0
):
    """
    Calculate trading intensity with transaction costs and market impact
    """
    # Enhanced reservation price calculation with inventory limits
    r = reservation_price(mid_price, q_current, gamma, sigma, T, t, q_max)

    delta_ask = delta
    delta_bid = delta
    
    # Adjust quotes based on inventory position
    if q_max is not None and q_max > 0:
        inventory_skew = q_current / q_max
        # Skew quotes to encourage inventory rebalancing
        delta_ask -= inventory_skew * delta * 0.2  # More aggressive ask when long
        delta_bid += inventory_skew * delta * 0.2  # More aggressive bid when short

    # Calculate intensities with adjusted spreads
    intensity_ask = lambda_ask * np.exp(-k * delta_ask) * dt
    intensity_bid = lambda_bid * np.exp(-k * delta_bid) * dt

    # Generate Poisson distributed trade events
    N_b = ss.poisson.rvs(intensity_bid)  # Buyer hits our ask
    N_a = ss.poisson.rvs(intensity_ask)  # Seller hits our bid

    # Apply transaction costs to cash flow
    cash_jump = N_b * (r + delta_ask - trans_cost) - N_a * (r - delta_bid + trans_cost)
    q_change = N_a - N_b
    
    # Apply market impact - price moves against large trades
    price_impact = 0
    if market_impact > 0 and (N_a > 0 or N_b > 0):
        price_impact = market_impact * sigma * (N_a - N_b) / np.sqrt(dt)
    
    return cash_jump, q_change, price_impact


def bid_ask(mid_price, inventory, gamma, sigma, delta, T, t, q_max=None):
    """
    Calculate bid-ask prices with inventory risk management
    """
    r = reservation_price(mid_price, inventory, gamma, sigma, T, t, q_max)
    
    # Adjust spreads based on inventory position
    skew_factor = 0
    if q_max is not None and q_max > 0:
        skew_factor = inventory / q_max
    
    ask = r + delta * (1 - 0.2 * skew_factor)  # Reduce ask when long inventory
    bid = r - delta * (1 + 0.2 * skew_factor)  # Reduce bid when long inventory
    
    return ask, bid


def simulate_trading(S_0, N, T, sigma, gamma, k, lambda_ask, lambda_bid, 
                     q_max=None, trans_cost=0, market_impact=0, 
                     mean_reversion=0, long_term_mean=None,
                     market_vol_adjustment=0):
    """
    Enhanced simulation with additional risk parameters
    """
    dt = T / N
    S = price_fluctuation(S_0, N, T, sigma, mean_reversion, long_term_mean)
    q = np.zeros(N + 1)
    cash = np.zeros(N + 1)
    pnl = np.zeros(N + 1)
    bid = np.zeros(N)
    ask = np.zeros(N)
    spread = np.zeros(N)
    trades = []
    
    for t in range(N):
        current_time = t * dt
        mid_price = S[t]
        q_current = q[t]

        # Skip quoting if inventory exceeds limits
        if q_max is not None and abs(q_current) >= q_max:
            # Only quote on the side that reduces inventory
            if q_current >= q_max:  # Too long, only quote asks
                delta = optimal_spread(gamma, sigma, k, T, current_time, q_current, q_max, market_vol_adjustment)
                r = reservation_price(mid_price, q_current, gamma, sigma, T, current_time, q_max)
                ask[t] = r + delta
                bid[t] = 0  # Don't quote bid
            elif q_current <= -q_max:  # Too short, only quote asks
                delta = optimal_spread(gamma, sigma, k, T, current_time, q_current, q_max, market_vol_adjustment)
                r = reservation_price(mid_price, q_current, gamma, sigma, T, current_time, q_max)
                ask[t] = 0  # Don't quote ask
                bid[t] = r - delta
            else:
                delta = optimal_spread(gamma, sigma, k, T, current_time, q_current, q_max, market_vol_adjustment)
                ask[t], bid[t] = bid_ask(mid_price, q_current, gamma, sigma, delta, T, current_time, q_max)
        else:
            delta = optimal_spread(gamma, sigma, k, T, current_time, q_current, q_max, market_vol_adjustment)
            ask[t], bid[t] = bid_ask(mid_price, q_current, gamma, sigma, delta, T, current_time, q_max)
        
        spread[t] = ask[t] - bid[t] if (ask[t] > 0 and bid[t] > 0) else None
        
        cash_jump, q_change, price_impact = trading_intensity(
            mid_price,
            sigma,
            gamma,
            k,
            T,
            current_time,
            q_current,
            lambda_ask,
            lambda_bid,
            delta,
            dt,
            q_max,
            trans_cost,
            market_impact
        )
        
        # Record trades if they happened
        if q_change != 0:
            trades.append({
                'time': current_time, 
                'price': mid_price, 
                'q_change': q_change, 
                'inventory': q_current, 
                'new_inventory': q_current + q_change
            })
        
        # Update inventory and cash
        q[t + 1] = q_current + q_change
        cash[t + 1] = cash[t] + cash_jump
        
        # Apply market impact to price process
        if price_impact != 0:
            if t < N-1:  # Avoid index errors
                S[t+1] += price_impact
                
        # Calculate mark-to-market P&L
        pnl[t+1] = cash[t+1] + q[t+1] * S[t+1] - (cash[0] + q[0] * S[0])

    # Calculate performance metrics
    performance = calculate_performance_metrics(S, q, cash, pnl, bid, ask, trades, dt)
    
    return S, q, cash, pnl, bid, ask, spread, trades, performance


def calculate_performance_metrics(S, q, cash, pnl, bid, ask, trades, dt):
    """
    Calculate various performance metrics for the trading strategy
    """
    # Convert trades to DataFrame for easier analysis
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    # Calculate metrics
    metrics = {}
    
    # Final P&L and return
    metrics['final_pnl'] = pnl[-1]
    initial_capital = cash[0] + q[0] * S[0]
    if initial_capital > 0:
        metrics['return'] = pnl[-1] / initial_capital
    else:
        metrics['return'] = 0
    
    # Sharpe ratio (annualized)
    if len(pnl) > 1:
        pnl_diff = np.diff(pnl)
        if np.std(pnl_diff) > 0:
            metrics['sharpe_ratio'] = np.mean(pnl_diff) / np.std(pnl_diff) * np.sqrt(1/dt)
        else:
            metrics['sharpe_ratio'] = 0
    else:
        metrics['sharpe_ratio'] = 0
    
    # Maximum drawdown
    running_max = np.maximum.accumulate(pnl)
    drawdown = running_max - pnl
    metrics['max_drawdown'] = np.max(drawdown)
    
    # Inventory metrics
    metrics['max_inventory'] = np.max(np.abs(q))
    metrics['avg_inventory'] = np.mean(np.abs(q))
    
    # Trade metrics
    metrics['num_trades'] = len(trades)
    if len(trades) > 0:
        metrics['avg_trade_size'] = np.mean([abs(t['q_change']) for t in trades])
    else:
        metrics['avg_trade_size'] = 0
    
    # Average spread and quote presence
    valid_spreads = [s for s in (ask - bid) if s > 0]
    metrics['avg_spread'] = np.mean(valid_spreads) if valid_spreads else 0
    metrics['quote_presence'] = np.sum([(a > 0 and b > 0) for a, b in zip(ask, bid)]) / len(ask)
    
    return metrics


def plot_results(S, q, cash, pnl, bid, ask, spread, title="Avellaneda-Stoikov Market Making Simulation"):
    """
    Plot comprehensive results of the simulation
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig)
    
    # Price and quotes chart
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(S, label="Mid Price", color='black')
    ax1.plot(ask, label="Ask Price", linestyle="--", color='red', alpha=0.7)
    ax1.plot(bid, label="Bid Price", linestyle="--", color='green', alpha=0.7)
    ax1.set_title(f"{title} - Prices and Quotes")
    ax1.legend()
    ax1.grid(True)
    
    # Inventory chart
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(q, label="Inventory", color='blue')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_title("Inventory Position")
    ax2.legend()
    ax2.grid(True)
    
    # PnL chart
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(pnl, label="PnL", color='green')
    ax3.set_title("Profit and Loss")
    ax3.legend()
    ax3.grid(True)
    
    # Spread chart
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(spread, label="Spread", color='purple')
    ax4.set_title("Bid-Ask Spread")
    ax4.legend()
    ax4.grid(True)
    
    # Cash balance chart
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(cash, label="Cash", color='orange')
    ax5.set_title("Cash Balance")
    ax5.legend()
    ax5.grid(True)
    
    plt.tight_layout()
    return fig


def print_performance(metrics):
    """
    Print formatted performance metrics
    """
    print("\n===== PERFORMANCE METRICS =====")
    print(f"Final P&L: {metrics['final_pnl']:.2f}")
    print(f"Return: {metrics['return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}")
    print(f"Maximum Absolute Inventory: {metrics['max_inventory']:.2f}")
    print(f"Average Absolute Inventory: {metrics['avg_inventory']:.2f}")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Average Trade Size: {metrics['avg_trade_size']:.2f}")
    print(f"Average Spread: {metrics['avg_spread']:.4f}")
    print(f"Quote Presence: {metrics['quote_presence']:.2%}")
    print("===============================")


if __name__ == "__main__":
    # Parameters
    S_0 = 100               # Initial price
    N = 1000                # Number of time steps
    T = 1                   # Time horizon in days
    sigma = 0.2             # Volatility
    gamma = 0.1             # Risk aversion parameter
    k = 1.5                 # Shape parameter for order flow
    lambda_ask = 1.0        # Base intensity for market sell orders (hitting our bid)
    lambda_bid = 1.0        # Base intensity for market buy orders (hitting our ask)
    
    # Enhanced parameters
    q_max = 10              # Maximum inventory position
    trans_cost = 0.01       # Transaction cost per share
    market_impact = 0.1     # Market impact factor
    mean_reversion = 0.05   # Mean reversion factor (0 = no mean reversion)
    long_term_mean = S_0    # Long-term mean price for mean reversion
    market_vol_adjustment = 0.2  # Adjust spread based on market volatility
    
    # Run simulation
    S, q, cash, pnl, bid, ask, spread, trades, performance = simulate_trading(
        S_0, N, T, sigma, gamma, k, lambda_ask, lambda_bid,
        q_max, trans_cost, market_impact, mean_reversion, long_term_mean,
        market_vol_adjustment
    )
    
    # Plot results
    fig = plot_results(S, q, cash, pnl, bid, ask, spread)
    
    # Print performance metrics
    print_performance(performance)
    
    plt.show()
    
    # Run parameter sensitivity analysis
    print("\nRunning parameter sensitivity analysis...")
    
    # Test different risk aversion parameters
    gamma_values = [0.05, 0.1, 0.2, 0.5]
    gamma_results = []
    
    for g in gamma_values:
        _, _, _, pnl, _, _, _, _, perf = simulate_trading(
            S_0, N, T, sigma, g, k, lambda_ask, lambda_bid,
            q_max, trans_cost, market_impact, mean_reversion, long_term_mean
        )
        gamma_results.append((g, perf['final_pnl'], perf['sharpe_ratio']))
    
    print("\nRisk Aversion (gamma) Sensitivity:")
    for g, pnl, sharpe in gamma_results:
        print(f"Gamma: {g}, P&L: {pnl:.2f}, Sharpe: {sharpe:.2f}")