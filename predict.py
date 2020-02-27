import getopt
import sys
import pandas as pd
import pickle
import os


# function to call models according to uuid and setpoint and return the energy consumption
def calculate_energy(uuid, setpoint):
    #uuid = '09ac4a10-7e8e-40f3-a327-1f93a5cf2383'
    #setpoint = 42

    test_df = pd.read_csv(os.getcwd() + '\\external\\LGA_temp.csv')
    test_df['day'] = test_df['DATE'].str[3:5].astype(int)
    test_df['hour'] = test_df['DATE'].str[6:8].astype(int)
    test_df = test_df.drop(columns=['DATE'])
    # drop DATE column
    test_df['setpoint'] = setpoint

    prob_wt_dict = pickle.load(open(uuid + '_prob_weights.pickle', 'rb'))
    test_df['prob_weight'] = test_df['hour'].map(prob_wt_dict['hour_weight'])
    test_df['prob_weight'] = test_df['prob_weight'].fillna(0)

    kde_dict = pickle.load(open(uuid + '_kde_params.pickle', 'rb'))
    reg = pickle.load(open(uuid + '_reg_params.pickle', 'rb'))

    energy_trials_sum = 0.0

    print('models loaded, starting simulation')

    # try 1000 trials, and initialize list
    for trial in range(0, 1000):
        for hour in list(prob_wt_dict['hour_weight'].keys()):
            if hour >= 18 and hour <= 21:
                test_df.loc[test_df['hour'] == hour, 'unweighted_door_open_duration'] = \
                kde_dict['dinner'].sample(31).reshape(1, -1)[0]
            elif hour >= 13 and hour <= 14:
                test_df.loc[test_df['hour'] == hour, 'unweighted_door_open_duration'] = \
                kde_dict['lunch'].sample(31).reshape(1, -1)[0]
            else:
                test_df.loc[test_df['hour'] == hour, 'unweighted_door_open_duration'] = \
                kde_dict['other'].sample(31).reshape(1, -1)[0]

        test_df['door_open_duration'] = test_df['unweighted_door_open_duration'] * test_df['prob_weight']
        test_df['door_open_duration'] = test_df['door_open_duration'].fillna(0)
        test_df['delta_temp'] = test_df['roomTemp'] - test_df['setpoint']

        test_df['compressor_on_duration'] = reg.predict(test_df[['door_open_duration', 'delta_temp']])

        energy_trials_sum = energy_trials_sum + test_df['compressor_on_duration'].sum()
        if trial%250 == 0:
            print()
    energy = energy_trials_sum /1000 / 3600 * 0.2
    return(energy)


# main function
def main(argv):
    # parse input
    try:
        options, remainder = getopt.gnu_getopt(
            sys.argv[1:],
            's:u',
            ['setpoint=',
             'uuid=',
             ])
    except getopt.GetoptError as err:
        print('ERROR:', err)
        sys.exit(1)

    for opt, arg in options:
        if opt in '--setpoint':
            setpoint_arg = arg
        elif opt == '--uuid':
            uuid_arg = arg

    result = calculate_energy(uuid=uuid_arg, setpoint=setpoint_arg)
    print(result)

if __name__ == "__main__":
    main(sys.argv[1:])




