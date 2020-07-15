
def smooth_data(df, window_size=3):
    min_window_size = 1
    date_col = 3  # Beginning of date column
    df.iloc[:, date_col:] = df.iloc[:, date_col:].rolling(
        window_size, axis=1, min_periods=min_window_size).mean()
    return df
