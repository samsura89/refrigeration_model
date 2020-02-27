#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import config
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

for file_name in config.uuid_files:
    print(file_name)
    df = pd.read_pickle(file_name)

    # flag the instances the door opened
    df['door_toggle'] = df['door'] - df['door'].shift(periods=1, fill_value =0)
    # make a smoothing filter to remove the times when the door is closed, and then suddenly
    # opened again (over each samples)
    convolution_filter = np.ones_like(np.arange(4))/4
    df['door_opened_error'] = np.convolve(abs(df['door_toggle']), convolution_filter, mode='same')
    df.loc[df['door_opened_error'] == 0.5, 'door_toggle'] = 0
    # now, door_toggle has only +1 and -1 toggle, specifically one after the other. +1 denotes door open,
    # -1 denotes door closed

    # flag the instances the compressor turned on
    df['compressor_toggle'] =  df['compressor'] -  df['compressor'].shift(periods=1, fill_value =0)

    # calculate the the duration the door was open (using same logic as above)
    df.loc[df['door_toggle'] != 0,'door_open_duration'] = df.loc[df['door_toggle'] != 0,'timestamp']\
        - df.loc[df['door_toggle']!=0,'timestamp'].shift(periods=1)

    # calculate the the duration the door was open
    df.loc[df['compressor_toggle'] != 0,'compressor_on_duration'] = df.loc[df['compressor_toggle']!=0,'timestamp'] \
        - df.loc[df['compressor_toggle'] != 0,'timestamp'].shift(periods=1)
    df[df['compressor_toggle'] != 0].head()

    # let's look into the toggling activity, by keeping only the rows where something was toggled
    toggle_df = df.loc[np.logical_or(df['compressor_toggle'] != 0, df['door_toggle'] != 0), :].reset_index()

    # verify if the temperature inside the fridge and setpoint is the same when the door is opened.
    # this could affect accuracy
    toggle_df['result'] = toggle_df.loc[toggle_df['door_toggle'] == 1, 'setpoint']\
        - toggle_df.loc[toggle_df['door_toggle']==1,'temp']

    # toggle_df.loc[np.logical_and(toggle_df['door_toggle']==1,toggle_df['result']!=0),:]
    # shows that door was shut and opened while compressor was still running. Let's remove those
    toggle_df = toggle_df.drop(toggle_df.loc[np.logical_and(toggle_df['door_toggle'] == 1,
                                                            toggle_df['result'] != 0), :].index)

    #clean up the data by removing the unnecessary columns
    toggle_df = toggle_df.drop(columns=['result','index'])

    # get time when door was opened, and the temperature at that time
    toggle_df.loc[toggle_df['door_toggle'] == 1,'door_activated_time'] = toggle_df.loc[toggle_df['door_toggle'] == 1,
                                                                                       'timestamp']
    toggle_df.loc[toggle_df['door_toggle'] == 1,'door_activated_roomTemp'] = toggle_df.loc[toggle_df['door_toggle'] == 1,
                                                                                         'roomTemp']

    # forward fill those columns, so that the values are propogated further, thereby resulting in a simple
    # subtraction to find difference in times and temperatures
    toggle_df['door_activated_time'] = toggle_df['door_activated_time'].fillna(method='ffill')
    toggle_df['door_activated_roomTemp'] = toggle_df['door_activated_roomTemp'].fillna(method='ffill')
    toggle_df['door_open_duration'] = toggle_df['door_open_duration'].fillna(method='ffill')
    toggle_df['delta_temp'] = toggle_df['door_activated_roomTemp'] - toggle_df['setpoint']

    # prepare the final dataframe to model the compressor being switched on
    final_df = toggle_df.loc[toggle_df['compressor_toggle']==-1,['timestamp','delta_temp',
                                                                 'door_open_duration',
                                                                 'compressor_on_duration']].reset_index(drop=True)

    final_df['door_open_duration'] = final_df['door_open_duration'].dt.total_seconds()
    final_df['compressor_on_duration'] = final_df['compressor_on_duration'].dt.total_seconds()

    print('Linear Regression')
    reg = LinearRegression().fit(final_df[['door_open_duration','delta_temp']], final_df['compressor_on_duration'])
    print(reg.score(final_df[['door_open_duration','delta_temp']], final_df['compressor_on_duration']))

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(final_df[['door_open_duration','delta_temp']],
                                                                                final_df['compressor_on_duration'],
                                                                                test_size=0.2,
                                                                                random_state=4)

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    train_features = StandardScaler().fit_transform(train_features)
    test_features = StandardScaler().fit_transform(test_features)

    # Instantiate model with 1000 decision trees
    rf_reg = RandomForestRegressor(n_estimators = 1000)
    # Train the model on training data
    rf_reg.fit(train_features, train_labels)
    # Use the forest's predict method on the test data
    rf_reg_predictions = rf_reg.predict(test_features)
    # Performance metrics
    rf_reg_errors = abs(rf_reg_predictions - test_labels)
    # print performance evaluations
    print('Random forest search')
    print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, rf_reg_predictions))
    print('Mean Squared Error:', metrics.mean_squared_error(test_labels, rf_reg_predictions))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, rf_reg_predictions)))
    print('R2 value:', rf_reg.score(test_features, test_labels))

    # Perform Grid-Search
    search = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': [5, 10, 20],
            'n_estimators': [100, 500, 1000, 2000],},
        cv=5, n_jobs=-1)
    search_result = search.fit(train_features, train_features)
    best_params = search_result.best_params_

    # use the optimal parameters
    rf_search = RandomForestRegressor(max_depth=best_params["max_depth"],
                                      n_estimators=best_params["n_estimators"])

    # Train the model on training data
    rf_search.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    rf_search_predictions = rf_search.predict(test_features)

    # Performance metrics
    print('Grid search')
    print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, rf_search_predictions))
    print('Mean Squared Error:', metrics.mean_squared_error(test_labels, rf_search_predictions))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, rf_search_predictions)))
    print('R2 value:', rf_search.score(test_features, test_labels))

    pickle.dump(reg, open(str(file_name[:-7]+'_reg_params.pickle'), 'wb'))
    print('------')
