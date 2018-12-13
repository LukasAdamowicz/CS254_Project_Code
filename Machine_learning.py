from IMU_JAE.DataStructures import LowerBody3 as LB3
from IMU_JAE.imu_import import ReadAPDM
from IMU_JAE import vicon_methods
from sklearn import linear_model, preprocessing, svm, tree, neural_network, metrics
import os
import pandas as pd
import pickle
import matplotlib.pyplot as pl
import numpy as np
from matplotlib.patches import Patch
import sys

from timeit import default_timer as timer

# TODO remove if TF/keras ever get updated to Python 3.7
if sys.version_info[1] == 6:
    from keras import Sequential
    from keras.layers import Dense, Conv1D as conv1d, Conv2D as conv2d, Flatten
    from keras import optimizers


def run_first(ml_save, bodies_save):
    if os.sep == '\\':
        study_path = 'W:\\Study Data\\Healthy Subjects\\'
    elif os.sep == '/':
        study_path = '/media/lukas/Work/Study Data/Healthy Subjects/'
    subs = [i for i in os.listdir(study_path) if os.path.isdir(study_path + i)]
    apdm_path = os.sep + 'Lab' + os.sep + 'APDM' + os.sep + 'monitorData' + os.sep
    vicon_path = os.sep + 'Lab' + os.sep + 'VICON' + os.sep

    # TODO make this save the subjects instead of keeping them all in memory
    bodies = {i: LB3() for i in subs}
    ml = pd.DataFrame()
    for sub in subs:
        bodies[sub].create_sensors()
        bodies[sub].create_joints()
        files = [i for i in os.listdir(study_path + sub + apdm_path) if 'treadmill' in i.lower()]
        vfiles = [i for i in os.listdir(study_path + sub + vicon_path) if 'treadmill' in i.lower()]
        for file in files:
            name = ' '.join(file.split('_', 1)[1].split('.')[0].split('_'))
            print(f'Subject: {sub}   Trial: {name}')

            vpath_mkr = study_path + sub + vicon_path + name + '.mkr'
            vpath_mdl = study_path + sub + vicon_path + name + '.mdl'

            if os.path.isfile(vpath_mkr):
                apath = study_path + sub + apdm_path + file
                ReadAPDM.ImportH5(bodies[sub], apath, name)

                vicon_methods.process_validation_data(bodies[sub], vpath_mkr, vpath_mdl, name)
        vicon_methods.all_event_rotation_process(bodies[sub])
        ml = vicon_methods.machine_learning_accumulation(bodies[sub], ml, sub=sub)

    pickle.dump(ml, open(ml_save, 'wb'))
    pickle.dump(bodies, open(bodies_save, 'wb'))


def c2s(x):
    # convert cartesian to spherical
    xs = np.zeros_like(x)
    xs[:, 0] = np.linalg.norm(x, axis=1)
    xs[:, 1] = np.arccos(x[:, 2]/xs[:, 0])
    xs[:, 2] = np.arctan2(x[:, 1], x[:, 0])

    return xs


def s2c(x):
    # convert spherical to cartesian
    xc = np.zeros_like(x)
    xc[:, 0] = x[:, 0] * np.cos(x[:, 2]) * np.sin(x[:, 1])
    xc[:, 1] = x[:, 0] * np.sin(x[:, 2]) * np.sin(x[:, 1])
    xc[:, 2] = x[:, 0] * np.cos(x[:, 1])

    return xc


def plot_jc(data, sub, subs, locs):
    f, ax = pl.subplots(figsize=(12, 7))

    ulocs = np.unique(locs)

    handles = []
    labels = []
    if isinstance(data, pd.DataFrame):
        for i, l in enumerate(ulocs):
            ax.set_prop_cycle(None)
            ax.plot(data.loc[(subs == sub) & (locs == l), ('cx1', 'cy1', 'cz1')])
            x = data.loc[(subs == sub) & (locs == l)].index
            ax.axvspan(x[0], x[-1], alpha=0.4, color='C' + str(i+3))
            handles.append(Patch(facecolor='C'+str(i+3), alpha=0.4))
            labels.append(f'{l}')
    elif isinstance(data, np.ndarray):
        index = np.arange(len(subs))
        for i, l in enumerate(ulocs):
            ax.set_prop_cycle(None)
            ax.plot(data[(subs == sub) & (locs == l), :3])
            x = index[(subs == sub)]
            ax.axvspan(x[0], x[-1], alpha=0.4, color='C' + str(i + 3))
            handles.append(Patch(facecolor='C' + str(i + 3), alpha=0.4))
            labels.append(f'{l}')

    ax.legend(handles, labels)
    f.tight_layout()


def ML(model, data, target, subjs):
    sc = []
    r2 = []
    rmse = []
    vrmse = []
    pl.close('all')
    for sub in np.unique(subjs):

        ind = subjs != sub
        nind = subjs == sub

        model.fit(data[ind], target[ind])
        sc.append(model.score(data[nind], target[nind]))

        pred = model.predict(data[nind])

        r2.append(metrics.r2_score(target[nind], pred))
        print(f'Subject: {sub}  Score: {sc[-1]:.4f}  Variance Score: {r2[-1]:.4f}')

        f, ax = pl.subplots(nrows=2, ncols=3, figsize=(12, 8))
        f.suptitle(f'Subject: {sub}  Score: {sc[-1]:.4f}')
        for i in range(3):
            ax[0, i].plot(target[nind, i], label='Target')
            ax[0, i].plot(pred[:, i], '.', label='Prediction', alpha=0.25)
            ax[0, i].set_title('Sensor 1')
            ax[0, i].set_ylabel('Position [m]')
            ax[0, i].set_xlabel('Sample')
            ax[0, i].legend()
        for i in range(3):
            ax[1, i].plot(target[nind, i + 3], label='Target')
            ax[1, i].plot(pred[:, i + 3], '.', label='Prediction', alpha=0.25)
            ax[1, i].set_title('Sensor 2')
            ax[1, i].set_ylabel('Position [m]')
            ax[1, i].set_xlabel('Sample')
            ax[1, i].legend()

    return sc


def ML_tf(model, data, target, subjs, epochs=200, batch=128, verbose=0, shuffle=True, validation_split=0):
    sc = []
    pl.close('all')
    for sub in ['S0008']:  # np.unique(subjs):
        ind = subjs != sub
        nind = subjs == sub

        model.fit(data[ind], target[ind], epochs=epochs, batch_size=batch, verbose=verbose, shuffle=shuffle,
                  validation_split=validation_split)
        sc.append(model.evaluate(data[nind], target[nind])[1])
        print(f'Score: {sc[-1]}')

        pred = model.predict(data[nind])

        f, ax = pl.subplots(nrows=2, ncols=1, figsize=(12, 8))
        f.suptitle(f'Subject: {sub}  Score: {sc[-1]:.4f}')
        sn = ['Radius', 'Angle 1', 'Angle 2']
        su = ['Radius [mm]', 'Angle [rad]', 'Angle [rad]']
        for i in range(1):
            ax[0, i].plot(target[nind, i], label='Target')
            ax[0, i].plot(pred[:, i], label='Prediction', alpha=0.25)
            ax[0, i].set_title(f'Sensor 1: {sn[i]}')
            ax[0, i].set_ylabel(su[i])
            ax[0, i].set_xlabel('Sample')
            ax[0, i].legend()
        for i in range(1):
            ax[1, i].plot(target[nind, i+3], label='Target')
            ax[1, i].plot(pred[:, i+3], '+', label='Prediction', alpha=0.25)
            ax[1, i].set_title(f'Sensor 2: {sn[i]}')
            ax[1, i].set_ylabel(su[i])
            ax[1, i].set_xlabel('Sample')
            ax[1, i].legend()
        f.tight_layout()
    return sc


def ML_cnn(model, data, target, epochs=200, verbose=0):
    sc = []
    pl.close('all')
    inds = np.arange(0, len(data))
    for i in range(len(data)):
        dind = inds != i  # data mask
        tind = inds == i  # training mask

        model.fit(data[dind], target[dind], epochs=epochs, verbose=verbose)
        sc.append(model.evaluate(data[tind], target[tind])[1])
        print(f'Score: {sc[-1]}')

    return sc


def cnn_data(df, sensor1_2='split', t_sys='sphere'):
    a1 = df.loc[:, ('ax1', 'ay1', 'az1')]
    w1 = df.loc[:, ('wx1', 'wy1', 'wz1')]
    wd1 = df.loc[:, ('wdx1', 'wdy1', 'wdz1')]
    a2 = df.loc[:, ('ax2', 'ay2', 'az2')]
    w2 = df.loc[:, ('wx2', 'wy2', 'wz2')]
    wd2 = df.loc[:, ('wdx2', 'wdy2', 'wdz2')]

    _c1 = df.loc[:, ('cx1', 'cy1', 'cz1')] / 1000
    _c2 = df.loc[:, ('cx2', 'cy2', 'cz2')] / 1000

    if t_sys == 'sphere':
        c1 = c2s(_c1.values)
        c2 = c2s(_c2.values)
    else:
        c1 = _c1
        c2 = _c2

    cnn = []
    cnn_t = []
    for s in df.Subject.unique():
        ind = df.Subject == s
        if sensor1_2 == 'split':
            sd = np.concatenate((a1[ind].values.reshape((-1, 3, 1)), a2[ind].values.reshape((-1, 3, 1)),
                                 w1[ind].values.reshape((-1, 3, 1)), w2[ind].values.reshape((-1, 3, 1)),
                                 wd1[ind].values.reshape((-1, 3, 1)), wd2[ind].values.reshape((-1, 3, 1))), axis=2)
        elif sensor1_2 == 'comb':
            a = np.concatenate((a1[ind].values.reshape((-1, 3, 1)), a2[ind].values.reshape((-1, 3, 1))), axis=1)
            w = np.concatenate((w1[ind].values.reshape((-1, 3, 1)), w2[ind].values.reshape((-1, 3, 1))), axis=1)
            wd = np.concatenate((wd1[ind].values.reshape((-1, 3, 1)), wd2[ind].values.reshape((-1, 3, 1))), axis=1)
            sd = np.concatenate((a, w, wd), axis=2)

        c = np.concatenate((c1[ind].reshape((-1, 3, 1)), c2[ind].reshape((-1, 3, 1))), axis=2)
        cnn_t.append(c)

        cnn.append(sd)

    ns = [d.shape[0] for d in cnn]
    nmin = np.min(ns)
    for i in range(len(cnn)):
        cnn[i] = cnn[i][:nmin, :, :]
        cnn_t[i] = cnn_t[i][:nmin, :, :]

    return np.array(cnn), np.array(cnn_t)


if os.sep == '\\':
    path = 'W:\\Study Data\\Healthy Subjects\\'
elif os.sep == '/':
    path = '/media/lukas/Work/Study Data/Healthy Subjects/'
ml_save = path + 'ml_data.pickle'
bodies_save = path + 'bodies.pickle'

if not os.path.isfile(ml_save) and not os.path.isfile(bodies_save):
    run_first(ml_save, bodies_save)

# run_first(ml_save, bodies_save)


def plot_sensor_jc(sub, bodies):
    pl.close('all')
    for se in bodies[sub].imus:
        mkr = bodies[sub].__dict__[se].mkr
        for ev in mkr.jc_s.keys():
            pl.figure(figsize=(9, 5))
            pl.plot(mkr.jc_s[ev].reshape((-1, 3)))
            pl.title(f'{se}   {ev}')


def plot_world_jc(sub, bodies):
    for j in bodies[sub].joints:
        mdl = bodies[sub].__dict__[j].mdl
        for ev in mdl.jc_w.keys():
            pl.figure()
            pl.plot(mdl.jc_w[ev])
            pl.title(f'{j}  {ev}')


bodies = pickle.load(open(bodies_save, 'rb'))
ml_data = pickle.load(open(ml_save, 'rb'))

ml_thresh = ml_data[(np.linalg.norm(ml_data.loc[:, ('wx1', 'wy1', 'wz1')], axis=1) > 0.5)]

a1 = ml_thresh.loc[:, ('ax1', 'ay1', 'az1')]
w1 = ml_thresh.loc[:, ('wx1', 'wy1', 'wz1')]
wd1 = ml_thresh.loc[:, ('wdx1', 'wdy1', 'wdz1')]
a2 = ml_thresh.loc[:, ('ax2', 'ay2', 'az2')]
w2 = ml_thresh.loc[:, ('wx2', 'wy2', 'wz2')]
wd2 = ml_thresh.loc[:, ('wdx2', 'wdy2', 'wdz2')]

c1 = ml_thresh.loc[:, ('cx1', 'cy1', 'cz1')] / 1000
c2 = ml_thresh.loc[:, ('cx2', 'cy2', 'cz2')] / 1000

cs1 = c2s(c1.values)
cs2 = c2s(c2.values)

# pickle.dump([a1, w1, wd1, a2, w2, wd2, c1, c2, cs1, cs2, ml_thresh.Subject, ml_thresh.location, ml_thresh.event],
# open('ml_raw.pickle', 'wb'))

data = np.concatenate((a1.values, w1.values, wd1.values, a2.values, w2.values, wd2.values), axis=1)

target = np.concatenate((cs1[:, 0].reshape((-1, 1)), cs2[:, 0].reshape((-1, 1))), axis=1)

data, target, subjs, locs, events = vicon_methods.window_ml_data(data, target, 10, ml_thresh.Subject,
                                                                 ml_thresh.location, ml_thresh.event)

# CNN Data
# d_cnn_s, t_cnn_s = cnn_data(ml_thresh, sensor1_2='split')
# d_cnn_c, t_cnn_c = cnn_data(ml_thresh, sensor1_2='comb')

# Model Creation
model = linear_model.LinearRegression()
preprocessing.PolynomialFeatures(2).fit_transform(data)
# model = neural_network.MLPRegressor((18, 18), activation='tanh', solver='adam', verbose=True, max_iter=300)

start = timer()
sc = ML(model, data, target, subjs)
print(f'Took {timer()-start:.4f}s to train and test')

# model_tf = Sequential()
# model_tf.add(Dense(50, input_dim=data[0].size))
# model_tf.add(Dense(50))
# model_tf.add(Dense(6))
#
# model_tf.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# CNN model
# cnn = Sequential()
# cnn.add(conv2d(3, (64, 1), strides=1, padding='valid', data_format='channels_last', use_bias=False,
#                input_shape=(d_cnn_s[0].shape[0], 3, 6)))  # use input_shape=(None, 6) for combined data
# cnn.add(conv2d(6, (3, 3), strides=1, padding='valid', data_format='channels_last', use_bias=False))
# cnn.add(Flatten())
# cnn.add(Dense(32))
# cnn.add(Dense(6))
#
# cnn.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# sc_tf = ML_tf(model_tf, data, target, subjs, epochs=50, batch=128, verbose=0, shuffle=False, validation_split=0.15)

# sc_cnn = ML_cnn(cnn, d_cnn_s, t_cnn_s, epochs=50, verbose=1)

