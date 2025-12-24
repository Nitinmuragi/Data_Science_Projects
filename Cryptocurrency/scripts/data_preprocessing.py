import pandas as pd

def preprocess_data(input_file, output_file):
    df = pd.read_csv(input_file)
    df['snapped_at'] = pd.to_datetime(df['snapped_at'])
    df.rename(columns={'price': 'Close'}, inplace=True)
    df.set_index('snapped_at', inplace=True)
    df.dropna(inplace=True)
    df['7_day_MA'] = df['Close'].rolling(7).mean()
    df.to_csv(output_file)
    print("Saved cleaned file:", output_file)


preprocess_data(
    "data/raw_data/BTC-USD.csv", 
    "data/processed_data/cleaned_btc_data.csv"
)
