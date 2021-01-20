import pandas as pd
import pyfolio as pf
import matplotlib

def report_gen():
    results = pd.read_pickle('test_zipline.pickle')
    returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(results)
    pf.create_full_tear_sheet(returns, positions=positions, transactions=transactions,
                              round_trips=True)

    fig.savefig('returns_tear_sheet.pdf')

if __name__ == '__main__':
    path = "C:/Users/16477/Desktop/zipline/dat"
    os.chdir(path)
    report_gen()
