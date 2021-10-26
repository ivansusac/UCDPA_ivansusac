# Importing libraries
from sqlalchemy import create_engine
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce



""" IMPORTING DATA """
# Data is available in the csv format and in the database.
# I will import two tables from database and two tables from csv files.

# defining connection to the database
db_connection_str = 'mysql+pymysql://root:@localhost/crypto_prices'
db_connection = create_engine(db_connection_str)

# reading tables from database
btc_df = pd.read_sql('SELECT * FROM bitcoin', con=db_connection)
doge_df = pd.read_sql('SELECT * FROM dogecoin', con=db_connection)

# reading data from csv files
eth_df = pd.read_csv('Ethereum/ETH-USD.csv')
xrp_df = pd.read_csv('XRP/XRP-USD.csv')

# printing first 5 rows of each dataframe
print(btc_df.head())
print(doge_df.head())
print(eth_df.head())
print(xrp_df.head())



""" ANALYSING DATA """
# Checking correct dates with Regex, Iterators and Functions
# Here I am using regex to check if all the dates are in correct format or not
r = re.compile(r'\d{4}-\d{2}-\d{2}')

def check_correct_date(df):
    # getting no. of rows in the dataframe
    rows = df.shape[0]
    # getting no. of rows where date is in correct format
    matched_rows = df.Date.apply(lambda x: bool(r.match(x))).sum()
    if rows == matched_rows:
        print('All dates seems to be in correct format')
        return
    bad_rows = rows - matched_rows
    print(bad_rows, 'seems to be incorrect')
# running our function to check format
check_correct_date(btc_df)
check_correct_date(doge_df)
check_correct_date(eth_df)
check_correct_date(xrp_df)



""" CHECKING DATATYPES """
print(btc_df.info())
print(doge_df.info())
print(eth_df.info())
print(xrp_df.info())

# replacing strings 'null'
btc_df = btc_df.replace('null', np.nan)
doge_df = doge_df.replace('null', np.nan)
eth_df = eth_df.replace('null', np.nan)
xrp_df = xrp_df.replace('null', np.nan)

# looks like BTC and DOGE was imported from SQL as strings
# convert all the columns except the first one to numeric in BTC and DOGE
for col in btc_df.columns[1:]:
    btc_df[col] = pd.to_numeric(btc_df[col])

for col in doge_df.columns[1:]:
    doge_df[col] = pd.to_numeric(doge_df[col])

# convert first column to datetime object
btc_df['Date'] = pd.to_datetime(btc_df.Date)
doge_df['Date'] = pd.to_datetime(doge_df.Date)
eth_df['Date'] = pd.to_datetime(eth_df.Date)
xrp_df['Date'] = pd.to_datetime(xrp_df.Date)



""" HANDLING NULL VALUES """
# check for null values
print(btc_df[btc_df.isna().any(axis=1)])
print(btc_df[btc_df.isna().any(axis=1)])
print(btc_df[btc_df.isna().any(axis=1)])
print(btc_df[btc_df.isna().any(axis=1)])

# fill null values with the next value
btc_df.fillna(method='backfill', inplace=True)
doge_df.fillna(method='backfill', inplace=True)
eth_df.fillna(method='backfill', inplace=True)
xrp_df.fillna(method='backfill', inplace=True)


""" HANDLING DUPLICATE DATA """

# check for duplicate data
print(btc_df[btc_df.duplicated()])
print(doge_df[doge_df.duplicated()])
print(eth_df[eth_df.duplicated()])
print(xrp_df[xrp_df.duplicated()])

# None of the dataframes contain any duplicated values.
# If let suppose there are duplicates values we can drop by using df.drop_duplicates().


""" VISUALISING THE DATA"""
# Plotting based on "Close" price
fig = plt.figure(figsize = (15,10))

plt.subplot(2, 2, 1)
plt.plot(btc_df['Date'], btc_df['Close'], color="red")
plt.title('Bitcoin Close Price')

plt.subplot(2, 2, 2)
plt.plot(doge_df['Date'], doge_df['Close'], color="orange")
plt.title('Dogecoin Close Price')

plt.subplot(2, 2, 3)
plt.plot(eth_df['Date'], eth_df['Close'], color="green")
plt.title('Ethereum Close Price')

plt.subplot(2, 2, 4)
plt.plot(xrp_df['Date'], xrp_df['Close'], color="black")
plt.title('XRP Close Price')

plt.show();



# Plotting based on "Adjusted Close"
# Will not be used in report as it is only relevant to stocks
fig = plt.figure(figsize = (15,10))

plt.subplot(2, 2, 1)
plt.plot(btc_df['Date'], btc_df['Adj Close'], color="red")
plt.title('Bitcoin Adj Close Price')

plt.subplot(2, 2, 2)
plt.plot(doge_df['Date'], doge_df['Adj Close'], color="orange")
plt.title('Dogecoin Adj Close Price')

plt.subplot(2, 2, 3)
plt.plot(eth_df['Date'], eth_df['Adj Close'], color="green")
plt.title('Ethereum Adj Close Price')

plt.subplot(2, 2, 4)
plt.plot(xrp_df['Date'], xrp_df['Adj Close'], color="black")
plt.title('XRP Adj Close Price')
plt.show();



# Plotting based on "Volume"
fig = plt.figure(figsize = (15,10))

plt.subplot(2, 2, 1)
plt.plot(btc_df['Date'], btc_df['Volume'], color="red")
plt.title('Bitcoin Volume')

plt.subplot(2, 2, 2)
plt.plot(doge_df['Date'], doge_df['Volume'], color="orange")
plt.title('Dogecoin Volume')

plt.subplot(2, 2, 3)
plt.plot(eth_df['Date'], eth_df['Volume'], color="green")
plt.title('Ethereum Volume')

plt.subplot(2, 2, 4)
plt.plot(xrp_df['Date'], xrp_df['Volume'], color="black")
plt.title('XRP Volume')
plt.show();



# Plotting based on "Daily Return"
fig = plt.figure(figsize = (15,10))

plt.subplot(2, 2, 1)
sns.distplot(btc_df['Adj Close'].pct_change(),kde=True,color='red')
plt.xlabel('Daily Return')
plt.ylabel('Daily Return')
plt.title('Bitcoin')

plt.subplot(2, 2, 2)
sns.distplot(doge_df['Adj Close'].pct_change(),color='orange')
plt.xlabel('Daily Return')
plt.ylabel('Daily Return')
plt.title('Dogecoin')

plt.subplot(2, 2, 3)
sns.distplot(eth_df['Adj Close'].pct_change(),color='green')
plt.xlabel('Daily Return')
plt.ylabel('Daily Return')
plt.title('Ethereum')

plt.subplot(2, 2, 4)
sns.distplot(xrp_df['Adj Close'].pct_change(),color='black')
plt.title('XRP')
plt.xlabel('Daily Return')
plt.ylabel('Daily Return')

fig.tight_layout()
plt.show();



""" MERGING DATAFRAMES"""
# As we want only close column, we select those columns and merge them in a single dataframe
btc_df['BTC'] = btc_df['Close']
doge_df['DOGE'] = doge_df['Close']
eth_df['ETH'] = eth_df['Close']
xrp_df['XRP'] = xrp_df['Close']

# keeping only dates and closed prices
btc_df = btc_df[['Date', 'BTC']]
doge_df = doge_df[['Date', 'DOGE']]
eth_df = eth_df[['Date', 'ETH']]
xrp_df = xrp_df[['Date', 'XRP']]
# merge all the dataframes
df = reduce(lambda x,y: pd.merge(x,y, on='Date', how='inner'), [btc_df, doge_df, eth_df, xrp_df])
df

# checking correlation between crypto prices
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation Matrix')
plt.show();


"""MACHINE LEARNING"""
# To keep our machine learning better, we will keep only recent data i.e., the prices from the previous year
df = df[df['Date'] > '2020-10-20']




