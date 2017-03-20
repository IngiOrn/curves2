import pandas as pd
import pickle
import numpy as np
import datetime as dt

swaprate = np.array([0.0036,
                    0.0052,
                    0.0093,
                    0.0121,
                    0.0146,
                    0.0166,
                    0.0184,
                    0.0199,
                    0.0213,
                    0.0221,
                    0.0263,
                    0.0273,
                    0.0271])

swaps = np.array([1,
                  2,
                  3,
                  4,
                  5,
                  6,
                  7,
                  8,
                  9,
                 10,
                 15,
                 20,
                 30])


spotday = dt.date(2017,3, 15)

def get_maturities(spotday, swaps):
    return np.array([dt.date(spotday.year +d, spotday.month, spotday.day) for d in swaps])



maturity = get_maturities(spotday,swaps)

market = pd.DataFrame({'SwapRate': swaprate,
                       'Maturity': maturity})
market['Source'] = "Swap2"


market.to_pickle('swap_market.p')

