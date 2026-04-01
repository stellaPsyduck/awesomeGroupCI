import yfinance as yf

tickers = [
    "AMD",
    "IBM",
    "CSCO",
    "AAPL",
    "MSFT",
    "ORCL",
    "INTC",
    "CRM",
    ]

for ticker in tickers:
    try:
        stock_data = yf.download(ticker, start="2007-01-01", end="2017-12-31", auto_adjust=False, progress=False)
        
        if not stock_data.empty:
            unadjusted_prices = stock_data[['Close']]
            
            filename = f"{ticker}_unadjusted_prices_2007_2018.csv"
            
            unadjusted_prices.to_csv(filename)
            
        else:
            print(f"Warning: No data found for {ticker}.\n")
            
    except Exception as e:
        print(f"An error occurred while processing {ticker}: {e}\n")
