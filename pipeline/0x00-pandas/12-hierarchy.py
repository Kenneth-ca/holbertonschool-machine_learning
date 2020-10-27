#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

df1.sort_values(by='Timestamp', ascending=True, inplace=True)
df2.sort_values(by='Timestamp', ascending=True, inplace=True)

df1.set_index('Timestamp', inplace=True)
df2.set_index('Timestamp', inplace=True)

df1 = df1.loc["1417411980":"1417417980"]
df2 = df2.loc["1417411980":"1417417980"]

df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
df = df.swaplevel(i=0, j=1, axis=0).sort_index()

print(df)
