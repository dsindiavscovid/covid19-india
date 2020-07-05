def weights_to_pdf(weights):
    return weights/weights.sum()


def pdf_to_cdf(pdf):
    return pdf.cumsum()


# def get_best_index(df, percentile, window):
#     one_side = window//2
#     idx = (df['cdf'] - percentile / 100).apply(abs).idxmin()
#     best_idx = df.iloc[idx - one_side:idx + one_side, :]['index'].min()
#     return best_idx

def get_best_index(df, percentile, window):
    one_side = window//2
    idx = (df['cdf'] - percentile / 100).apply(abs).idxmin()
#     best_idx = df.iloc[idx - one_side:idx + one_side, :]['index'].min()
    return str(idx)
