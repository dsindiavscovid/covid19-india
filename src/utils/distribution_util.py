def weights_to_pdf(weights):
    return weights/weights.sum()


def pdf_to_cdf(pdf):
    return pdf.cumsum()


def get_best_index(df, percentile, tolerance=1):
    """Gets the index of the row with lowest loss corresponding to a percentile (within a tolerance limit)

    Args:
        df (pd.Dataframe): dataframe with cdf and loss index (lower index indicates lower loss)
        percentile (int): percentile for which row index is to be found
        tolerance (int, optional): tolerance limit for percentile (default: 0.01)

    Returns:
        str: index of the row with lowest loss corresponding to the percentile (within a tolerance limit)
    """

    df_window = df[df['cdf'].between(max(0.0, (percentile - tolerance)/100), min(1.0, (percentile + tolerance)/100))]
    if not df_window.dropna().empty:
        idx = df_window['cdf'].idxmin()
    else:
        # TODO: Some indication to the user that the percentile is off
        idx = (df['cdf'] - percentile / 100).apply(abs).idxmin()
    return df.iloc[idx, :]['index']

