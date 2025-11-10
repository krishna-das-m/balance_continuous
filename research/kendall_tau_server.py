import numpy as np
import pandas as pd
import sys
import pickle

def kernel(k: str):
    """Kernel function

    Args:
        k (str): kernel name: ['gaussian', 'epanechnikov']
    """
    if k not in ['gaussian', 'epanechnikov']:
        raise ValueError('Unknown kernel')
    
    def bounded(f): # decorator
        def _f(x):  # wrapper function
            return f(x) if np.abs(x) <= 1 else 0
        return _f
    
    if k == 'gaussian':
        return lambda u: 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * u * u)
    elif k == 'epanechnikov':
        return bounded(lambda u: (3 / 4 * (1 - u * u)))
    
    
def kernel_weights(t, s, h, S, k=None):
    if k is None:
        k = kernel('epanechnikov')
    wh = k((t - s)/(S*h))
    return wh/(S*h)

# calculate tau 
def kendall_tau_estimator(Y_A, Y_B, h=None):
    assert len(Y_A) == len(Y_B), "Length Y_A and Y_B must be same"
    if h is None:
        h = 0.1
        print(h)
    S = len(Y_A) # both Y_A and Y_B needs to be of equal length
    tau_estimates = []
    # Create kernel function
    k = kernel('epanechnikov')
    for t in range(1,S+1):
        # weights = np.array([k((t - s) / (S * h)) / (S * h) for s in range(1, S + 1)])
        weights = np.array([kernel_weights(t, s, h, S) for s in range(1, S+1)])
        normalize = 1 - np.sum(weights**2)
        if normalize <=0:
            raise ValueError("Normalization is less than 0, Adjust 'h'")
        summed_weights = 0

        # Create comparison matrices
        Y_A_comparison = Y_A[:, np.newaxis] < Y_A[np.newaxis, :]  # Shape: (S, S)
        Y_B_comparison = Y_B[:, np.newaxis] < Y_B[np.newaxis, :]  # Shape: (S, S)
        
        # Concordant pairs indicator
        indicator_matrix = (Y_A_comparison & Y_B_comparison).astype(int)
        
        # Weight matrix (outer product)
        weight_matrix = weights[:, np.newaxis] * weights[np.newaxis, :]  # Shape: (S, S)
        
        # Sum all weighted indicators
        summed_weights = np.sum(weight_matrix * indicator_matrix)
        tau_t = (4/normalize )*summed_weights-1
        tau_estimates.append(tau_t)
    return tau_estimates

# tickers_df = pd.read_parquet('artifacts/data_ingestion/tickers_df.parquet')

# start_date = pd.to_datetime('2000-01-01')
# end_date = pd.to_datetime('2001-12-31')

# # Filter for a date range
# price_data_range = tickers_df[
#     (tickers_df['Date'] >= start_date) & 
#     (tickers_df['Date'] <= end_date)
# ]
# price_data = price_data_range.pivot(index='Date', columns='Ticker', values='Adj Close')
# len(price_data)

# ticker_name = list(price_data.columns) 

# tau_stocks = []
# for i in range(len(ticker_name)):
#     tickerA = ticker_name[i]
#     YA = price_data[tickerA].to_numpy()
#     for j in range(i, len(ticker_name)):
#         tickerB = ticker_name[j]
#         YB = price_data[tickerB].to_numpy()
#         tau_AB = kendall_tau_estimator(YA, YB, h=0.1)
#         # print(i,j, tickerA, tickerB)
#         tau_stocks.append(tau_AB)

# save the estimated kendall tau as a pickle file
# with open('tau_stocks_alltime','wb') as f:
#     pickle.dump(tau_stocks, f)

def run_tau_estimator():
    # Load your data
    tickers_df = pd.read_parquet('artifacts/data_ingestion/tickers_df.parquet')

    # Filter for a date range
    start_date = pd.to_datetime('2000-01-01')
    end_date = pd.to_datetime('2001-05-31')
    price_data_range = tickers_df[
        (tickers_df['Date'] >= start_date) & 
        (tickers_df['Date'] <= end_date)
    ]

    price_data = price_data_range.pivot(index='Date', columns='Ticker', values='Adj Close')

    # Compute Kendall tau
    YA = price_data[tickerA].dropna().to_numpy()
    YB = price_data[tickerB].dropna().to_numpy()

    tau_AB = kendall_tau_estimator(YA, YB, h=0.1)

    # Save result
    output_file = f"output_tau/tau_{tickerA}_{tickerB}.csv"
    pd.DataFrame([[price_data.index, tau_AB]], columns=['Date', 'Tau']).to_csv(output_file, index=False)

    print(f"Saved τ({tickerA}, {tickerB}) = {tau_AB} to {output_file}")

if __name__== "__main__":
    if len(sys.argv)!=3:
        print("Usage")
        sys.exit(1)
    tickerA = sys.argv[1]
    tickerB = sys.argv[2]
    run_tau_estimator()