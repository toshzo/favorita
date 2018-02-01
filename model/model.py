from datetime import date, timedelta

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

df_train = pd.read_csv(
    '../input/train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 40672217) #Skip dates before 2016-08-01

)

df_test = pd.read_csv(
    "../input/test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]  # , date_parser=parser
).set_index(
    ['store_nbr', 'item_nbr', 'date']
)

items = pd.read_csv(
    "../input/items.csv",
).set_index("item_nbr")


df_train['store_item'] = (df_train['store_nbr'].astype(str)+df_train['item_nbr'].astype(str)).values
df_2017 = df_train.loc[df_train.date>=pd.datetime(2017,7,1)]
train_store_item_id = df_2017['store_item'].copy().values 

df_2017 = df_train.loc[df_train.date>=pd.datetime(2015,1,1)]
del df_train

print (df_2017.shape)
df_2017 = df_2017.query('store_item in @train_store_item_id')
df_2017.drop(['store_item'], axis=1 , inplace=True)
print (df_2017.shape)

promo_2017_train = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)
promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
df_2017_index = promo_2017_test.index
promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)

del promo_2017_test, promo_2017_train

df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
df_2017.columns = df_2017.columns.get_level_values(1)

items = items.reindex(df_2017.index.get_level_values(1))

df_2017['class'] = items['class'].values
df_2017['family'] = items['family'].values

df_2017[pd.datetime(2016,12,25)] = np.float64(0)
df_2017[pd.datetime(2015,12,25)] = np.float64(0)


def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def get_weekend_timespan(df, dt, minus, periods):
    week_end_span = pd.date_range(dt - timedelta(days=minus), periods=periods, freq='D')[
    pd.date_range(dt - timedelta(days=minus), periods=periods, freq='D').weekday > 4]
    return df[week_end_span]

import numba
def seq_zero(df_bf):
    df_bf['custum'] = ""
    for c in df_bf:
        df_bf['custum'] += df_bf[c].astype(str)
    return df_bf['custum']


@numba.jit(nopython=True)
def single_autocorr(series, lag):
    s1 = series[lag:]
    s2 = series[:-lag]
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    ds1 = s1 - ms1
    ds2 = s2 - ms2
    divide = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
    return np.sum(ds1 * ds2) / divide if divide != 0 else 0

def count_zero(self):
    count = 0
    for k in list(self):
        if k == '0':
            count += 1
        else:
            break
    return count


def inv_count_zero(self):
    count = 0
    for k in list(self)[::-1]:
        if k == '0':
            count += 1
        else:
            break
    return count


def prepare_dataset(t2017, is_train=True):
    year = 365
    X = pd.DataFrame({
#         "autocorrelation_90_2017": get_timespan(df_2017, t2017, 180, 180).apply(lambda x:single_autocorr(x.values, 90),axis=1).values,
        "autocorrelation_60_2017": get_timespan(df_2017, t2017, 90, 90).apply(lambda x:single_autocorr(x.values, 60),axis=1).values,
        "autocorrelation_30_2017": get_timespan(df_2017, t2017, 60, 60).apply(lambda x:single_autocorr(x.values, 30),axis=1).values,
        "autocorrelation_14_2017": get_timespan(df_2017, t2017, 28, 28).apply(lambda x:single_autocorr(x.values, 14),axis=1).values,        
        "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
        "mean_7_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
        "mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
        "mean_21_2017": get_timespan(df_2017, t2017, 21, 21).mean(axis=1).values,
        "mean_28_2017": get_timespan(df_2017, t2017, 28, 28).mean(axis=1).values,
        "mean_60_2017": get_timespan(df_2017, t2017, 60, 60).mean(axis=1).values,
        "mean_90_2017": get_timespan(df_2017, t2017, 90, 90).mean(axis=1).values,
        "mean_112_2017": get_timespan(df_2017, t2017, 112, 112).mean(axis=1).values,
        "mean_140_2017": get_timespan(df_2017, t2017, 140, 140).mean(axis=1).values,
        "promo_7_2017": get_timespan(promo_2017, t2017, 7, 7).sum(axis=1).values,
        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
        "promo_21_2017": get_timespan(promo_2017, t2017, 21, 21).sum(axis=1).values,
        "promo_28_2017": get_timespan(promo_2017, t2017, 28, 28).sum(axis=1).values,
        "promo_60_2017": get_timespan(promo_2017, t2017, 60, 60).sum(axis=1).values,
#         "promo_140_2017": get_timespan(promo_2017, t2017, 140, 140).sum(axis=1).values
        "median_3_2017": get_timespan(df_2017, t2017, 3, 3).median(axis=1).values,
        "median_7_2017": get_timespan(df_2017, t2017, 7, 7).median(axis=1).values,
        "median_14_2017": get_timespan(df_2017, t2017, 14, 14).median(axis=1).values,
        "median_21_2017": get_timespan(df_2017, t2017, 21, 21).median(axis=1).values,
        "median_28_2017": get_timespan(df_2017, t2017, 28, 28).median(axis=1).values,
        "median_56_2017": get_timespan(df_2017, t2017, 28, 28).median(axis=1).values,
        "median_84_2017": get_timespan(df_2017, t2017, 28, 28).median(axis=1).values,
        "median_140_2017":get_timespan(df_2017, t2017, 140, 140).replace(0, np.nan).median(axis=1, skipna=True).values,
        "max_3_2017": get_timespan(df_2017, t2017, 3, 3).max(axis=1).values,
        "max_7_2017": get_timespan(df_2017, t2017, 7, 7).max(axis=1).values,
        "max_14_2017": get_timespan(df_2017, t2017, 14, 14).max(axis=1).values,
        "max_28_2017": get_timespan(df_2017, t2017, 28, 28).max(axis=1).values,
        "rolling_median_3_2017":get_timespan(df_2017, t2017, 3+(3-1), 3+(3-1)).rolling(window=3,axis=1).mean().mean(axis=1).values,
        "rolling_median_7_2017":get_timespan(df_2017, t2017, 7+(7-1), 7+(7-1)).rolling(window=7,axis=1).mean().mean(axis=1).values,
        "rolling_median_14_2017":get_timespan(df_2017,  t2017, 14+(14-1), 14+(14-1)).rolling(window=14,axis=1).mean().mean(axis=1).values,
        "rolling_median_21_2017":get_timespan(df_2017,  t2017, 21+(21-1), 21+(21-1)).rolling(window=21,axis=1).mean().mean(axis=1).values,
        "rolling_median_28_2017":get_timespan(df_2017,  t2017, 28+(28-1), 28+(28-1)).rolling(window=28,axis=1).mean().mean(axis=1).values,
        "rolling_median_60_2017":get_timespan(df_2017,  t2017, 60+(60-1), 60+(60-1)).rolling(window=60,axis=1).mean().mean(axis=1).values,
        "rolling_median_90_2017":get_timespan(df_2017,  t2017, 90+(90-1), 90+(90-1)).rolling(window=90,axis=1).mean().mean(axis=1).values,
        "rolling_mean_3_2017":get_timespan(df_2017, t2017, 3+(3-1), 3+(3-1)).rolling(window=3,axis=1).median().median(axis=1).values,
        "rolling_mean_7_2017":get_timespan(df_2017, t2017, 7+(7-1), 7+(7-1)).rolling(window=7,axis=1).median().median(axis=1).values,
        "rolling_mean_14_2017":get_timespan(df_2017,  t2017, 14+(14-1), 14+(14-1)).rolling(window=14,axis=1).median().median(axis=1).values,
        "rolling_mean_21_2017":get_timespan(df_2017,  t2017, 21+(21-1), 21+(21-1)).rolling(window=21,axis=1).median().median(axis=1).values,
        "rolling_mean_28_2017":get_timespan(df_2017,  t2017, 28+(28-1), 28+(28-1)).rolling(window=28,axis=1).median().median(axis=1).values,
        "mean_diff7_3_2017": get_timespan(df_2017, t2017, 3+7, 3).mean(axis=1).values,
        "mean_diff7_7_2017": get_timespan(df_2017, t2017, 7+7, 7).mean(axis=1).values,
        "mean_diff7_14_2017": get_timespan(df_2017, t2017, 14+7, 14).mean(axis=1).values,
        "mean_diff7_21_2017": get_timespan(df_2017, t2017, 21+7, 21).mean(axis=1).values,
        "mean_diff7_28_2017": get_timespan(df_2017, t2017, 28+7, 28).mean(axis=1).values,
        "mean_diff28_3_2017": get_timespan(df_2017, t2017, 3+28, 3).mean(axis=1).values,
        "mean_diff28_7_2017": get_timespan(df_2017, t2017, 7+28, 7).mean(axis=1).values,
        "mean_diff28_14_2017": get_timespan(df_2017, t2017, 14+28, 14).mean(axis=1).values,
        "mean_diff28_21_2017": get_timespan(df_2017, t2017, 21+28, 21).mean(axis=1).values,
        "mean_diff28_28_2017": get_timespan(df_2017, t2017, 28+28, 28).mean(axis=1).values,
        "mean_diff28_60_2017": get_timespan(df_2017, t2017, 60+28, 60).mean(axis=1).values,
        "mean_diff60_30_2017": get_timespan(df_2017, t2017, 30+60, 30).mean(axis=1).values,
        "mean_diff90_30_2017": get_timespan(df_2017, t2017, 30+90, 30).mean(axis=1).values,
        "mean_diff120_30_2017": get_timespan(df_2017, t2017, 30+120, 30).mean(axis=1).values,
        "mean_diff120_60_2017": get_timespan(df_2017, t2017, 60+120, 60).mean(axis=1).values,
        "mean_diff180_60_2017": get_timespan(df_2017, t2017, 90+180, 90).mean(axis=1).values,
        "max_diff7_28_2017": get_timespan(df_2017, t2017, 28+7, 28).max(axis=1).values,
        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
        "mean_3_2016": get_timespan(df_2017, t2017, 3+year, 3).mean(axis=1).values,
        "mean_7_2016": get_timespan(df_2017, t2017, 7+year, 7).mean(axis=1).values,
        "mean_14_2016": get_timespan(df_2017, t2017, 14+year, 14).mean(axis=1).values,
        "mean_21_2016": get_timespan(df_2017, t2017, 21+year, 21).mean(axis=1).values,
        "mean_28_2016": get_timespan(df_2017, t2017, 28+year, 28).mean(axis=1).values,
        "mean_56_2016": get_timespan(df_2017, t2017, 56+year, 56).mean(axis=1).values,
        "rolling_median_14_2016":get_timespan(df_2017,  t2017, year+14+(14-1), 14+(14-1)).rolling(window=14,axis=1).median().median(axis=1).values,
        "rolling_median_21_2016":get_timespan(df_2017,  t2017, year+21+(21-1), 21+(21-1)).rolling(window=21,axis=1).median().median(axis=1).values,
        "rolling_median_28_2016":get_timespan(df_2017,  t2017, year+28+(28-1), 28+(28-1)).rolling(window=28,axis=1).median().median(axis=1).values,
        "rolling_median_60_2016":get_timespan(df_2017,  t2017, year+60+(60-1), 60+(60-1)).rolling(window=60,axis=1).median().median(axis=1).values,
        "sum7_zero_value": np.sum((get_timespan(df_2017, t2017, 7, 7)==0).values, axis=1),        
        "sum14_zero_value": np.sum((get_timespan(df_2017, t2017, 14, 14)==0).values, axis=1),        
        "sum21_zero_value": np.sum((get_timespan(df_2017, t2017, 21, 21)==0).values, axis=1),        
        "sum28_zero_value": np.sum((get_timespan(df_2017, t2017, 28, 28)==0).values, axis=1),        
        "seq7_zero_value":seq_zero(get_timespan(df_2017, t2017, 7, 7).astype(int).mask(get_timespan(df_2017, t2017, 7, 7).astype(int) > 0, 1)).astype(int).values,
        "seq14_zero_value":seq_zero(get_timespan(df_2017, t2017, 14, 14).astype(int).mask(get_timespan(df_2017, t2017, 14, 14).astype(int) > 0, 1)).astype(np.float32).values,
        "seq21_zero_value":seq_zero(get_timespan(df_2017, t2017, 21, 21).astype(int).mask(get_timespan(df_2017, t2017, 21, 21).astype(int) > 0, 1)).astype(np.float32).values,
        "seq28_zero_value":seq_zero(get_timespan(df_2017, t2017, 28, 28).astype(int).mask(get_timespan(df_2017, t2017, 28, 28).astype(int) > 0, 1)).astype(np.float32).values,
        "item_mean_7_2017":get_timespan(df_2017, t2017, 7, 7).groupby('item_nbr').transform('mean').mean(axis=1).values,
        "item_mean_14_2017":get_timespan(df_2017, t2017, 14, 14).groupby('item_nbr').transform('mean').mean(axis=1).values,
        "item_mean_28_2017":get_timespan(df_2017, t2017, 28, 28).groupby('item_nbr').transform('mean').mean(axis=1).values,
        "item_mean_56_2017":get_timespan(df_2017, t2017, 56, 56).groupby('item_nbr').transform('mean').mean(axis=1).values,
        "item_mean_84_2017":get_timespan(df_2017, t2017, 84, 84).groupby('item_nbr').transform('mean').mean(axis=1).values,
        "mean_weekend90_90_2017": get_weekend_timespan(df_2017, t2017, 90,90).mean(axis=1).values,
        "mean_weekend56_56_2017": get_weekend_timespan(df_2017, t2017, 56,56).mean(axis=1).values,
        "mean_weekend28_28_2017": get_weekend_timespan(df_2017, t2017, 28,28).mean(axis=1).values,
        "mean_weekend14_14_2017": get_weekend_timespan(df_2017, t2017, 14,14).mean(axis=1).values,
        'mean_7_dw_2017' : get_timespan(df_2017, t2017, 7, 5, freq='B').mean(axis=1).values,
        'mean_14_dw_2017' : get_timespan(df_2017, t2017, 14, 10, freq='B').mean(axis=1).values,
        'mean_28_dw_2017' : get_timespan(df_2017, t2017, 28, 20, freq='B').mean(axis=1).values,
        'mean_56_dw_2017' : get_timespan(df_2017, t2017, 56, 40, freq='B').mean(axis=1).values,
        'mean_84_dw_2017' : get_timespan(df_2017, t2017, 84, 60, freq='B').mean(axis=1).values,
        'sale_na_duration' : seq_zero(df_2017.drop(['class', 'family'], axis=1).astype(int)).apply(inv_count_zero).values,
        'sale_duration' : seq_zero(df_2017.drop(['class', 'family'], axis=1).astype(int)).apply(count_zero).values,
    })
    
    for i in range(7):
        X['mean_2_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 14-i, 2, freq='7D').mean(axis=1).values
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['mean_6_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 42-i, 6, freq='7D').mean(axis=1).values
        X['mean_8_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 56-i,8, freq='7D').mean(axis=1).values
        X['mean_10_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 70-i,10, freq='7D').mean(axis=1).values
        X['mean_12_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 84-i,12, freq='7D').mean(axis=1).values
        X['mean_14_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 98-i,14, freq='7D').mean(axis=1).values
        X['mean_16_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 112-i,16, freq='7D').mean(axis=1).values
        X['mean_18_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 126-i,18, freq='7D').mean(axis=1).values
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').mean(axis=1).values
        X['mean_24_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 168-i, 24, freq='7D').mean(axis=1).values


    for i in [7, 28]:
        X['diff_mean_diff{}_3_2017'.format(i)] =  X["mean_3_2017"] - X["mean_diff{}_3_2017".format(i)]
        X['diff_mean_diff{}_7_2017'.format(i)] =  X["mean_7_2017"] - X["mean_diff{}_7_2017".format(i)]
        X['diff_mean_diff{}_14_2017'.format(i)] =  X["mean_14_2017"] - X["mean_diff{}_14_2017".format(i)]
        X['diff_mean_diff{}_21_2017'.format(i)] =  X["mean_21_2017"] - X["mean_diff{}_21_2017".format(i)]
        X['diff_mean_diff{}_28_2017'.format(i)] =  X["mean_28_2017"] - X["mean_diff{}_28_2017".format(i)]


    #concat unit_sales by day
    X2 = pd.DataFrame(get_timespan(df_2017, t2017, 28, 28).values)
    for col in X2.columns:
        X2.rename(columns={col:'unit28_'+str(col)}, inplace=True)
    X = pd.concat((X, X2), axis=1)
    

    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[
            t2017 + timedelta(days=i)].values.astype(np.uint8)
        
    global cols
    cols = list(X.columns)
    cols.extend((range(32)))
        
    X = np.hstack((X.values, df_2017[
            pd.date_range(t2017-timedelta(days=365), periods=32)
        ].values))
    
    X = pd.DataFrame(X, columns=cols).astype(np.float32)
        
    if is_train:
        y = df_2017[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    return X



print("Preparing dataset...")
#１週間ずつデータの期間をずらして、移動平均を作成
t2017 = date(2017, 6, 21)
X_l, y_l = [], []
for i in range(4):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(
        t2017 + delta
    )
    X_l.append(X_tmp)
    y_l.append(y_tmp)
    
t2017 = date(2016, 7, 11)
for i in range(4):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(
        t2017 + delta
    )
    X_l.append(X_tmp)
    y_l.append(y_tmp)

X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l ,y_l

X_val, y_val = prepare_dataset(date(2017, 7, 26))
X_val2, y_val2 = prepare_dataset(date(2016, 8, 16))
X_val = pd.concat((X_val, X_val2))
y_val = np.vstack((y_val, y_val2))
X_test = prepare_dataset(date(2017, 8, 16), is_train=False)
del X_val2, y_val2

print("Training and predicting models...")
params = {
    'num_leaves': 31,
    'objective': 'regression',
    'min_data_in_leaf': 300,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'rmse',
    'num_threads': 8
}


params = {
    'num_leaves': 33,
    'objective': 'regression',
    'min_data_in_leaf': 250,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'rmse',
    'num_threads': 22,
    "min_child_samples": 10,
    "min_child_weight": 150,
    "subsample": 0.9
}


stores = pd.read_csv('../input/stores.csv')
stores = df_2017.reset_index().merge(stores, on='store_nbr')[['type', 'cluster']]


X_train["type"] = pd.concat([stores["type"]] * 8).values
X_test["type"] = stores["type"].values
X_val["type"] = pd.concat([stores["type"]] * 2).values

from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
X_train["type"] = lbl.fit_transform(X_train["type"])
X_test["type"] = lbl.transform(X_test["type"])
X_val["type"] = lbl.transform(X_val["type"])
X_train = pd.concat((X_train, pd.get_dummies(X_train["type"])), axis=1)
X_test = pd.concat((X_test, pd.get_dummies(X_test["type"])), axis=1)
X_val = pd.concat((X_val, pd.get_dummies(X_val["type"])), axis=1)


X_train["cluster"] = pd.concat([stores["cluster"]] * 8).values
X_test["cluster"] = stores["cluster"].values
X_val["cluster"] = pd.concat([stores["cluster"]] * 2).values

X_train["cluster"] = lbl.fit_transform(X_train["cluster"])
X_test["cluster"] = lbl.transform(X_test["cluster"])
X_val["cluster"] = lbl.transform(X_val["cluster"])
X_train = pd.concat((X_train, pd.get_dummies(X_train["cluster"])), axis=1)
X_test = pd.concat((X_test, pd.get_dummies(X_test["cluster"])), axis=1)
X_val = pd.concat((X_val, pd.get_dummies(X_val["cluster"])), axis=1)


X_train["perishable"] = pd.concat([items["perishable"]] * 8).values
X_test["perishable"] = items["perishable"].values
X_val["perishable"] = pd.concat([items["perishable"]] * 2).values

X_train["class"] = pd.concat([items["class"]] * 8).values
X_test["class"] = items["class"].values
X_val["class"] = pd.concat([items["class"]] * 2).values

X_train["class"] = lbl.fit_transform(X_train["class"])
X_test["class"] = lbl.transform(X_test["class"])
X_val["class"] = lbl.transform(X_val["class"])

X_train["family"] = pd.concat([items["family"]] * 8).values
X_test["family"] = items["family"].values
X_val["family"] = pd.concat([items["family"]] * 2).values


X_train["family"] = lbl.fit_transform(X_train["family"])
X_test["family"] = lbl.transform(X_test["family"])
X_val["family"] = lbl.transform(X_val["family"])
X_train = pd.concat((X_train, pd.get_dummies(X_train["family"])), axis=1)
X_test = pd.concat((X_test, pd.get_dummies(X_test["family"])), axis=1)
X_val = pd.concat((X_val, pd.get_dummies(X_val["family"])), axis=1)

cols = ['mean_7_2017', 'mean_14_2017', 'mean_28_2017', 'mean_28_2016' ]
X_train = pd.concat((X_train, X_train.groupby('family').transform('mean')[cols]),axis=1)
X_test = pd.concat((X_test, X_test.groupby('family').transform('mean')[cols]),axis=1)
X_val = pd.concat((X_val, X_val.groupby('family').transform('mean')[cols]),axis=1)


def NWRMSLE(y, pred, w):
    return mean_squared_error(y, pred, sample_weight=w)**0.5

del df_2017


MAX_ROUNDS = 500
MAX_ROUNDS = 3000
f_name = 'single'
val_pred = []
test_pred = []
train_pred = []
cate_vars = []
for i in range(16):
    seq_num = np.zeros((X_train.shape[0], 16))
    seq_num_test = np.zeros((X_test.shape[0], 16))
    seq_num_val = np.zeros((X_val.shape[0], 16))
    seq_num[:,i] = 1
    seq_num_test[:,i] = 1
    seq_num_val[:,i] = 1
    col_idx = range(X_train.shape[1])
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    dtrain = lgb.Dataset(
        np.hstack((X_train.iloc[:, col_idx].values, seq_num)) , label=y_train[:, i],
        categorical_feature=cate_vars,
        weight=pd.concat([items["perishable"]] * 8) * 0.25 + 1
    )
    dval = lgb.Dataset(
        np.hstack((X_val.iloc[:, col_idx].values, seq_num_val)) , label=y_val[:, i], reference=dtrain,
        weight=(pd.concat([items["perishable"]] * 2) * 0.25 + 1),
        categorical_feature=cate_vars)
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=50
    )
    """
    print("\n".join(("%s: %.2f" % x) for x in sorted(
        zip(X_train.iloc[:, col_idx].columns, bst.feature_importance("gain")),
        key=lambda x: x[1], reverse=True
    )))
    """
    train_pred.append(bst.predict(
        np.hstack((X_train.iloc[:, col_idx].values, seq_num)), num_iteration=bst.best_iteration or MAX_ROUNDS))
    val_pred.append(bst.predict(
        np.hstack((X_val.iloc[:, col_idx].values, seq_num_val)), num_iteration=bst.best_iteration or MAX_ROUNDS))
    test_pred.append(bst.predict(
        np.hstack((X_test.iloc[:, col_idx].values, seq_num_test)), num_iteration=bst.best_iteration or MAX_ROUNDS))


print("5days Validation mse:", mean_squared_error(
    y_val[: ,:5], np.array(val_pred).transpose()[:, :5]))

print("all Validation mse:", mean_squared_error(
    y_val, np.array(val_pred).transpose()))

print ('alltrain', 
NWRMSLE(y_train[:int(y_train.shape[0]/2)], np.array(train_pred).transpose()[:int(y_train.shape[0]/2)], w=pd.concat([items["perishable"]] * 4) * 0.25 + 1)
)

print ('all days',
NWRMSLE(y_val[:int(y_val.shape[0]/2)], np.array(val_pred).transpose()[:int(y_val.shape[0]/2)], w=(items["perishable"] * 0.25 + 1))
)

print ('5days train',
    NWRMSLE(y_train[:int(y_train.shape[0]/2), :5], np.array(train_pred).transpose()[:int(y_train.shape[0]/2), :5], w=pd.concat([items["perishable"]] * 4) * 0.25 + 1)
)

print ('11days train', 
NWRMSLE(y_train[:int(y_train.shape[0]/2), 5:], np.array(train_pred).transpose()[:int(y_train.shape[0]/2), 5:], w=pd.concat([items["perishable"]] * 4) * 0.25 + 1)
)

print ('5days ',
NWRMSLE(y_val[:int(y_val.shape[0]/2), :5], np.array(val_pred).transpose()[:int(y_val.shape[0]/2), :5], w=(items["perishable"] * 0.25 + 1))
      )
print ('11days',
NWRMSLE(y_val[:int(y_val.shape[0]/2), 5:], np.array(val_pred).transpose()[:int(y_val.shape[0]/2), 5:], w=(items["perishable"] * 0.25 + 1))
)

print ('5days train',
    NWRMSLE(y_train[int(y_train.shape[0]/2):, :5], np.array(train_pred).transpose()[int(y_train.shape[0]/2):, :5], w=pd.concat([items["perishable"]] * 4) * 0.25 + 1)
)

print ('11days train', 
NWRMSLE(y_train[int(y_train.shape[0]/2):, 5:], np.array(train_pred).transpose()[int(y_train.shape[0]/2):, 5:], w=pd.concat([items["perishable"]] * 4) * 0.25 + 1)
)

print ('5days ',
NWRMSLE(y_val[int(y_val.shape[0]/2):, :5], np.array(val_pred).transpose()[int(y_val.shape[0]/2):, :5], w=(items["perishable"] * 0.25 + 1))
      )
print ('11days',
NWRMSLE(y_val[int(y_val.shape[0]/2):, 5:], np.array(val_pred).transpose()[int(y_val.shape[0]/2):, 5:], w=(items["perishable"] * 0.25 + 1))
)

y_test = np.array(test_pred).transpose()

print("Making submission...")
df_preds = pd.DataFrame(
    y_test, index=df_2017_index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
cut = 0.+1e-12 # 0.+1e-15
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.to_csv(str(f_name)+'_lgb_20170701_cluster_type_v32.csv', float_format='%.4f', index=None)
