import time

import numpy as np
import os.path
import random
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from numpy.linalg import norm
import statistics

from autokoopman import auto_koopman
import autokoopman.core.trajectory as traj

"""set the variable PATH to the directory with measurements folder"""
PATH = os.getcwd()


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_data(benchmark):
    """load the measured data"""

    path = os.path.join(PATH, benchmark)
    cnt = 0
    data = []

    while True:
        dirpath = os.path.join(path, 'measurement_' + str(cnt))
        if os.path.isdir(dirpath):
            states = np.asarray(pd.read_csv(os.path.join(dirpath, 'trajectory.csv')))
            inputs = np.asarray(pd.read_csv(os.path.join(dirpath, 'input.csv')))
            time = np.asarray(pd.read_csv(os.path.join(dirpath, 'time.csv'))) * 0.02
            time = np.resize(time, (time.shape[0],))
            data.append(traj.Trajectory(time, states, inputs))
            cnt += 1
        else:
            break

    if len(data) > 100:
        data = data[0:100]


    return data


def split_data(data, num_test=10):
    """randomly split data into training and test set"""

    random.seed(0)
    ind = random.sample(range(0, len(data)), num_test)

    test_data = [data[i] for i in ind]
    training_data = [data[i] for i in range(0, len(data)) if i not in ind]

    ids = np.arange(0, len(training_data)).tolist()
    training_data = traj.TrajectoriesData(dict(zip(ids, training_data)))

    ids = np.arange(0, len(test_data)).tolist()
    test_data = traj.TrajectoriesData(dict(zip(ids, test_data)))

    return training_data, test_data


def train_model(data):
    """train the Koopman model using the AutoKoopman library"""

    dt = data._trajs[0].times[1] - data._trajs[0].times[0]

    # learn model from data
    experiment_results = auto_koopman(
        data,  # list of trajectories
        sampling_period=dt,
        obs_type='rff',
        opt='grid',
        n_obs=200,
        rank=(1,200,20),
        grid_param_slices=5,
        n_splits=5,
        max_opt_iter=100
    )

    # get the model from the experiment results
    model = experiment_results['tuned_model']

    print(experiment_results['hyperparameters'])
    print(experiment_results['hyperparameter_values'])

    return model


def compute_trajectory(model, times, states, inputs, resample):
    test_traj = []
    test_traj.append(list(states[0]))

    current_state = states[0]
    
    for time_step in range(len(times) - 1):

        # Reset to actual ground truth
        #if time_step % resample == 0:
        #    current_state = states[time_step]
        
        time = times[time_step]

        if inputs is not None:
            action = inputs[time_step]
        else:
            action = None

        current_state = model.step(time, current_state, action)

        test_traj.append(list(current_state))
    
    return np.array(test_traj)
    

def compute_error(model, test_data):
    """compute error between model prediction and real data"""
    euc_norms = []

    # loop over all test trajectories
    tmp = list(test_data._trajs.values())


    for t in tmp:
        try:
            # simulate using the learned model
            iv = t.states[0, :]
            start_time = t.times[0]
            end_time = t.times[len(t.times) - 1]
            teval = np.linspace(start_time, end_time, len(t.times))

            trajectory = model.solve_ivp(
                initial_state=iv,
                tspan=(start_time, end_time),
                sampling_period=t.times[1] - t.times[0],
                inputs=t.inputs,
                teval=teval
            )

            # compute error
            y_true = np.matrix.flatten(t.states)
            y_pred = np.matrix.flatten(trajectory.states)
            euc_norm = norm(y_true - y_pred) / norm(y_true)
            euc_norms.append(euc_norm)
        except:
            print("ERROR--solve_ivp failed (likely unstable model)")
            # NOTE: Robot has constant 0 states, resulting in high error numbers (MSE is good)
            euc_norms.append(np.infty)


    '''
    for t in tmp:
        y_true = np.matrix.flatten(t.states)
        y_pred = np.matrix.flatten(compute_trajectory(model, t.times, t.states, t.inputs, 1))
        euc_norm = norm(y_true - y_pred) / norm(y_true)
        euc_norms.append(euc_norm)
    '''
    
    return statistics.mean(euc_norms)


def plot(trajectory, true_trajectory, var_1, var_2):
    plt.figure(figsize=(10, 6))
    # plot the results
    if var_2 == -1:  # plot against time
        plt.plot(trajectory.states[:, var_1], label='Trajectory Prediction')
        plt.plot(true_trajectory.states[:, var_1], label='Ground truth')
    else:
        plt.plot(trajectory.states.T[var_1], trajectory.states.T[var_2], label='Trajectory Prediction')
        plt.plot(true_trajectory.states.T[var_1], true_trajectory.states.T[var_2], label='Ground Truth')

    plt.grid()
    plt.show()

if __name__ == '__main__':
    benchmark = 'DubinsRejoin'

    # load data
    set_seed()
    data = load_data(benchmark)
    
    # split into training and validation set

    training_data, test_data = split_data(data, 80)

    start = time.time()
    model = train_model(training_data)
    end = time.time()


    # compute error
    euc_norm = compute_error(model, test_data)
    comp_time = round(end - start, 3)

    print(benchmark)
    print(f"The average euc norm perc error is {round(euc_norm * 100, 2)}%")
    print("time taken: ", comp_time)

    # loop over all test trajectories and plot
    tmp = list(test_data._trajs.values())

    t = tmp[5]

    test_traj = compute_trajectory(model, t.times, t.states, t.inputs, 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(test_traj.T[0], test_traj.T[1], label='Trajectory Prediction')
    plt.plot(t.states.T[0], t.states.T[1], label='Ground Truth')

    plt.legend()
    plt.grid()
    
    plt.show()

    '''
    t = tmp[0]
    # simulate using the learned model
    iv = t.states[0, :]
    start_time = t.times[0]
    sampling_period = t.times[1] - t.times[0]
    end_time = t.times[len(t.times) - 1]
    teval = np.linspace(start_time, end_time, len(t.times))

    trajectory = model.solve_ivp(
        initial_state=iv,
        tspan=(start_time, end_time),
        sampling_period=t.times[1] - t.times[0],
        inputs=t.inputs,
        teval=teval
    )
    plot(trajectory, t, 0, 1)

    #print(trajectory.states)
    '''

