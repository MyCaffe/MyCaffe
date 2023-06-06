import os
import glob
import pickle
from datetime import datetime
from pathlib import Path
from weakref import ProxyTypes
from tqdm import tqdm
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer, LabelEncoder, StandardScaler, MinMaxScaler

data_path = os.getcwd() + "\\data\\favorita\\raw"
# set parent directory as the output path
output_path = Path(data_path).parent.absolute()

# No records will be considered outside these bounds
start_date = datetime(2013, 7, 1)
end_date = datetime(2017, 4, 1)

# Where training period ends and the validation period begins
validation_bound = datetime(2016, 7, 1)

history_len = 90 # historical scope in time-steps
future_len = 30  # futureistic scope in time-steps (prediction period)

samp_interval = 5 # time-steps

# These are the variables that are known in advance, and will compose the futuristic time-series
known_attrs = ['onpromotion',
               'day_of_week',
               'day_of_month',
               'month',
               'national_holiday',
               'regional_holiday',
               'local_holiday',
               'open']

# The following set of variables will be considered as static, i.e. containing non-temporal information
# every attribute which is not listed here will be considered as temporal.
static_attrs = ['item_nbr',
                'store_nbr',
                'city',
                'state',
                'store_type',
                'store_cluster',
                'item_family',
                'item_class',
                'perishable']

# The following set of variables will be considred as categorical.
# The rest of the variables (which are not listed below) will be considered as numeric.
categorical_attrs = ['item_nbr',
                     'store_nbr',
                     'city',
                     'state',
                     'store_type',
                     'store_cluster',
                     'item_family',
                     'item_class',
                     'perishable',
                     'onpromotion',
                     'open',
                     'day_of_week',
                     'month',
                     'national_holiday',
                     'regional_holiday',
                     'local_holiday']

# Define the prediction signal
target_signal = 'log_sales'

# these will not be included as part of the input data which will end up feeding the model
meta_attrs = ['date', 'combination_id', 'temporal_id', 'unit_sales']

# Listing the relevant files.
file_names = [os.path.basename(f) for f in glob.glob(os.path.join(data_path, '*.{}'.format('csv')))]
print(file_names)

# Load the CSV files
transactions_df = pd.read_csv(os.path.join(data_path, 'transactions.csv'), parse_dates=['date'], infer_datetime_format=True)
items_df = pd.read_csv(os.path.join(data_path, 'items.csv'), index_col='item_nbr')
oil_df = pd.read_csv(os.path.join(data_path, 'oil.csv'), parse_dates=['date'], infer_datetime_format=True, index_col='date')
holiday_df = pd.read_csv(os.path.join(data_path, 'holidays_events.csv'), parse_dates=['date'], infer_datetime_format=True, dtype={'transferred':bool})
stores_df = pd.read_csv(os.path.join(data_path, 'stores.csv'), index_col='store_nbr')
data_df = pd.read_csv(os.path.join(data_path, 'train.csv'), dtype={'onpromotion':object}, index_col='id', parse_dates=['date'], infer_datetime_format=True)
# We will not use the test data in this demonstration - 
# the entire dataset will be created using the 'train.csv' file
test_df = pd.read_csv(os.path.join(data_path, 'test.csv'), index_col='id', parse_dates=['date'], infer_datetime_format=True)

if ptypes.is_object_dtype(data_df['onpromotion']):
    data_df['onpromotion'] = data_df['onpromotion'] = 'True'

stores_df.rename(columns={'type': 'store_type', 'cluster': 'store_cluster'}, inplace=True)
items_df.rename(columns={'class': 'item_class', 'family': 'item_family'}, inplace=True)
oil_df.rename(columns={'dcoilwtico': 'oil_price'}, inplace=True)
holiday_df.rename(columns={'type': 'holiday_type'}, inplace=True)

# Lose the null records on the raw dataframe representing the oil prices
oil_df = oil_df.loc[~oil_df.oil_price.isna()]
oil_df = oil_df.resample('1d').ffill().reset_index()

data_df = data_df.loc[(data_df['date'] >= start_date) & (data_df['date'] <= end_date)]
data_df = data_df.assign(combination_id=data_df['store_nbr'].apply(str) + '_' + data_df['item_nbr'].apply(str))
# another index can be used to identify the unique combination of (store, product, date)
data_df = data_df.assign(temporal_id=data_df['combination_id'] + '_' + data_df['date'].dt.strftime('%Y-%m-%d'))

# for each combination, we calculate the minimal unit_sales value
min_sales = data_df.groupby('combination_id', as_index=False)['unit_sales'].min()
# deep only combination with non-negative sales.
data_df = data_df.loc[data_df['combination_id'].isin(min_sales.loc[min_sales.unit_sales >= 0, 'combination_id'])]

# mark all the existing records as days in which the relevant stores were open
data_df = data_df.assign(open=1)

sequence_per_combination = [] # a list to contain all resamples sequences

# for each combination
for comb_id, comb_df in tqdm(data_df.groupby('combination_id')):
    resamp_seq = comb_df.copy()
    resamp_seq = resamp_seq.set_index('date').resample('1d').last().reset_index()
    resamp_seq['log_sales'] = np.log10(1 + resamp_seq['unit_sales'])
    #newly generated records are assumed to be days in which the store was not open
    resamp_seq['open'] = resamp_seq['open'].fillna(0)
    #pad with the corresponding information according to the previously available record
    for col in ['store_nbr', 'item_nbr', 'onpromotion']:
        resamp_seq[col] = resamp_seq[col].fillna(method='ffill')

    sequence_per_combination.append(resamp_seq)

# combine all the resample sequences
data_df = pd.concat(sequence_per_combination, axis=0)

data_df['day_of_week'] = pd.to_datetime(data_df['date'].values).dayofweek
data_df['day_of_month'] = pd.to_datetime(data_df['date'].values).day
data_df['month'] = pd.to_datetime(data_df['date'].values).month

# Add the metadata associated with each store and item

data_df = data_df.merge(stores_df, how='left', on='store_nbr')
data_df = data_df.merge(items_df, how='left', on='item_nbr')

# Add the holiday-related information associated with each date
# we'll ignore holiday-related information associated with each date
holiday_df = holiday_df.loc[~holiday_df.transferred]

# National holidays will mark every relevant record (by date)
data_df = data_df.assign(national_holiday=data_df.merge(holiday_df.loc[holiday_df.locale == 'National'], on='date', how='left')['description'].fillna('None'))

# Regional holidays will mark every relevant record (by date and state)
data_df = data_df.assign(regional_holiday=data_df.merge(holiday_df.loc[holiday_df.locale == 'Regional'], left_on=['date','state'], right_on=['date','locale_name'], how='left')['description'].fillna('None'))

# Local holidates will mark every relevant record (by date and city)
data_df = data_df.assign(local_holiday=data_df.merge(holiday_df.loc[holiday_df.locale == 'Local'], left_on=['date','city'], right_on=['date','locale_name'], how='left')['description'].fillna('None'))

data_df = data_df.merge(transactions_df, how='left', on=['date','store_nbr'])
data_df['transactions'] = data_df['transactions'].fillna(-1)
data_df = data_df.merge(oil_df, on='date', how='left')

all_cols = list(data_df.columns)
feature_cols = [col for col in all_cols if col not in meta_attrs]

# Create the list of attributes for each channel of input.  We'll need such list for each combination of (static/historical/futuristic) and (numeric/categorical)
feature_map = {
    'static_feats_numeric': [col for col in feature_cols if col in static_attrs and col not in categorical_attrs],
    'static_feats_categorical': [col for col in feature_cols if col in static_attrs and col in categorical_attrs],
    'historical_ts_numeric': [col for col in feature_cols if col not in static_attrs and col not in categorical_attrs],
    'historical_ts_categorical': [col for col in feature_cols if col not in static_attrs and col in categorical_attrs],
    'future_ts_numeric': [col for col in feature_cols if col in known_attrs and col not in categorical_attrs],
    'future_ts_categorical': [col for col in feature_cols if col in known_attrs and col in categorical_attrs]
}

# Data Scaling - we would like all of the input variables fed to the model to have similar scales. Hence each variable will be scaled (if it is numeric) or encoded
# if it is categorical.

# allocate a dictionary to contain the scaler and encoder objects after fitting them
scalers = {'numeric': dict(), 'categorical':dict() }
# for the categorical variables we would like to keep the cardinalities (how many categories for each variable)
categorical_cardinalities = dict()
# take only the the train time range
only_train = data_df.loc[data_df['date'] < validation_bound]

# Fitting the scalers/encoders
for col in tqdm(feature_cols):
    if col in categorical_attrs:
        scalers['categorical'][col] = LabelEncoder().fit(only_train[col].values)
        categorical_cardinalities[col] = only_train[col].nunique()
    else:
        if col in ['log_sales']:
            scalers['numeric'][col] = StandardScaler().fit(only_train[col].values.astype(float).reshape(-1, 1))
        elif col in ['day_of_month']:
            scalers['numeric'][col] = MinMaxScaler().fit(only_train[col].values.astype(float).reshape(-1, 1))
        else:
            scalers['numeric'][col] = QuantileTransformer(n_quantiles=256).fit(
                only_train[col].values.astype(float).reshape(-1, 1))


# After fitting the scalers and the encoders we apply them in order to transform the entire dataset.
for col in tqdm(feature_cols):

    if col in categorical_attrs:
        le = scalers['categorical'][col]
        # handle cases with unseen keys
        le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        data_df[col] = data_df[col].apply(lambda x: le_dict.get(x, max(le.transform(le.classes_)) + 1))
        data_df[col] = data_df[col].astype(np.int32)
    else:
        data_df[col] = scalers['numeric'][col].transform(data_df[col].values.reshape(-1, 1)).squeeze()
        data_df[col] = data_df[col].astype(np.float32)

# After performing the transformations above, and in order to avoid null records on the target variable, we impute target signal.
data_df['log_sales'].fillna(0.0, inplace=True)

# Splitting the data.
data_sets = {'train': dict(), 'validation': dict(), 'test': dict()}

for combination_id, combination_seq in tqdm(data_df.groupby('combination_id')):

    # take the complete sequence associated with this combination and break it into the relevant periods
    train_subset = combination_seq.loc[combination_seq['date'] < validation_bound]
    num_train_records = len(train_subset)
    validation_subset_len = num_train_records + future_len
    validation_subset = combination_seq.iloc[num_train_records - history_len: validation_subset_len]
    test_subset = combination_seq.iloc[validation_subset_len - history_len:]

    subsets_dict = {'train': train_subset,
                    'validation': validation_subset,
                    'test': test_subset}

    # for the specific combination we're processing in the current iteration,
    # we'd like to go over each subset separately
    for subset_key, subset_data in subsets_dict.items():
        # sliding window, according to samp_interval skips between adjacent windows
        for i in range(0, len(subset_data), samp_interval):
            # slice includes history period and horizons period
            slc = subset_data.iloc[i: i + history_len + future_len]

            if len(slc) < (history_len + future_len):
                # skip edge cases, where not enough steps are included
                continue

            # meta
            data_sets[subset_key].setdefault('time_index', []).append(slc.iloc[history_len - 1]['date'])
            data_sets[subset_key].setdefault('combination_id', []).append(combination_id)

            # static attributes
            data_sets[subset_key].setdefault('static_feats_numeric', []).append(
                slc.iloc[0][feature_map['static_feats_numeric']].values.astype(np.float32))
            data_sets[subset_key].setdefault('static_feats_categorical', []).append(
                slc.iloc[0][feature_map['static_feats_categorical']].values.astype(np.int32))

            # historical
            data_sets[subset_key].setdefault('historical_ts_numeric', []).append(
                slc.iloc[:history_len][feature_map['historical_ts_numeric']].values.astype(np.float32).reshape(
                    history_len, -1))
            data_sets[subset_key].setdefault('historical_ts_categorical', []).append(
                slc.iloc[:history_len][feature_map['historical_ts_categorical']].values.astype(np.int32).reshape(
                    history_len, -1))

            # futuristic (known)
            data_sets[subset_key].setdefault('future_ts_numeric', []).append(
                slc.iloc[history_len:][feature_map['future_ts_numeric']].values.astype(np.float32).reshape(future_len,
                                                                                                           -1))
            data_sets[subset_key].setdefault('future_ts_categorical', []).append(
                slc.iloc[history_len:][feature_map['future_ts_categorical']].values.astype(np.int32).reshape(future_len,
                                                                                                             -1))

            # target
            data_sets[subset_key].setdefault('target', []).append(
                slc.iloc[history_len:]['log_sales'].values.astype(np.float32))

# for each set
for set_key in list(data_sets.keys()):
    # for each component in the set
    for arr_key in list(data_sets[set_key].keys()):
        # list of arrays will be concatenated
        if isinstance(data_sets[set_key][arr_key], np.ndarray):
            data_sets[set_key][arr_key] = np.stack(data_sets[set_key][arr_key], axis=0)
        # lists will be transformed into arrays
        else:
            data_sets[set_key][arr_key] = np.array(data_sets[set_key][arr_key])

with open(os.path.join(output_path, 'data.pickle'), 'wb') as f:
    pickle.dump({
        'data_sets': data_sets,
        'feature_map': feature_map,
        'scalers': scalers,
        'categorical_cardinalities': categorical_cardinalities
    }, f, pickle.HIGHEST_PROTOCOL)

print("done!");
