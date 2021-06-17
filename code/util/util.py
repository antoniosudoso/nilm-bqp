import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from time import mktime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilmtk import DataSet

MAX_CLUSTERS = 4
LIMIT = 30
NOISY_APPLIANCES = {'noise'}
NOISY_PERCENTAGE = 'real'
PERCENTILE = 95
CLIPPED_PARAM = 0.25
AMPDS_NOISY_APP = {'real': ['security_and_network', 'basement', 'bedroom1',
                            'bedroom2', 'clothes_washer', 'dining_room',
                            'electric_oven', 'electronics_workbench', 'garage',
                            'instant_hot_water', 'office', 'outside']}
TRAIN_START = 14
TRAIN_END = 35
DROP_AMPDs = 38
DAYS_AMPDs = 45

START_UKDALE_TS = {1: '2014-06-01',
                   2: '2013-05-22'}
END_UKDALE_TS = {1: '2015-02-01',
                 2: '2013-09-22'}
DAYS_UKDALE = {1: 37,
               2: 38}
DROP_UKDALE = {1: 30,
               2: 31}
TRAIN_START_UKDALE = {1: 9,
                      2: 9}
TRAIN_END_UKDALE = {1: 30,
                    2: 30}
NILMTK_TRAIN_START_UKDALE = {1: "2014-06-10",
                             2: "2013-06-01"}
NILMTK_TRAIN_END_UKDALE = {1: "2014-06-30",
                           2: "2013-06-21"}
SELECTED_APPLIANCES_UKDALE = {1: ['meter2', 'meter5', 'meter6', 'meter9', 
                                  'meter12', 'noise'],
                              2: ['meter5', 'meter8', 'meter13', 'meter14',
                                  'meter18', 'noise']}
NILMTK_TEST_START_UKDALE = {1: "2014-06-30",
                            2: "2013-06-22"}
NILMTK_TEST_END_UKDALE = {1: "2014-07-07",
                          2: "2013-06-29"}


SELECTED_APPLIANCES_SYND = [3, 4, 5, 9]
DROP_SYND = 40
DAYS_SYND = 47
TRAIN_START_SYND = 16
TRAIN_END_SYND = 37


DROP_REFIT = {3: 30,
              9: 356}
DAYS_REFIT = {3: 37,
              9: 363}
TRAIN_START_REFIT = {3: "2013-10-20",
                     9: "2014-11-19"}
TRAIN_END_REFIT = {3: "2013-11-09",
                   9: "2014-12-09"}
TEST_START_REFIT = {3: "2013-11-09",
                    9: "2014-12-09"}
TEST_END_REFIT = {3: "2013-11-16",
                  9: "2014-12-16"}
START_REFIT = {3: "2013-10-10 00:00:00",
               9: "2013-12-18 00:00:00"}
SELECTED_APPLIANCES_REFIT = {3: [2, 4, 5, 7],
                             9: [1, 3, 4, 7]}


ENERGY_BAND = [1, 5, 24]   # as hour

def plot_histogram(values, title, path=""):
    plt.clf()
    plt.hist(values)
    plt.title(title)
    plt.savefig(path + title + ".png", format="png")

def print_to_file(file, text):
    """
    Write to a file in append mode.

    :param file: path of the file
    :param text: message to append
    :return: None
    """
    f = open(file, 'a')
    f.write(text)
    f.close()


def delete_file(path):
    """
    Delete a file if exists.

    :param path: path of the file
    :return: None
    """

    if os.path.exists(path):
        os.remove(path)


def create_dir(path):
    """
    Create dir if it doesn't exist.

    :param path: path of  the directory
    :return: None
    """

    if not os.path.exists(path):
        os.makedirs(path)


def read_csv_AMPDs(path, ts_col='unix_ts', p_col='P'):
    """
    Read a csv from the given path returning the numpy arrays related to the
    two columns specified.
    This works only in this directory or parallel's ones.

    :param path: path where to find data
    :param ts_col: header name of the time column
    :param p_col: header name of the power column
    :return: numpy arrays of the specified columns
    """

    csv = pd.read_csv(Path(__file__).parent.parent / path)
    return np.array(csv[ts_col]), np.array(csv[p_col])


def plot_data(x, y, name, color='red', show=False, clean=False, path=None):
    """
    Plot data using pyplot.

    :param x: numpy array
    :param y: numpy array
    :param name: title
    :param color: color of the plot
    :param show: boolean, if set it calls the show function of the plot
    :param clean: boolean, if set it clear the current plot
    :param path: path where to store image, if not set the plot isn't saved
    :return: None
    """

    if clean:
        plt.clf()

    plt.plot(x, y, label=name, color=color)
    plt.legend()
    plt.xlabel("# samples")
    plt.ylabel("Power [Watt]")

    if show:
        plt.show()
    elif path is not None:
        plt.savefig(fname=path, format='png')


def random_rgb():
    """
    Get a random color, in such way each appliance has a different color.

    :return: the rgb color
    """
    color = (random.uniform(0, 1),
             random.uniform(0, 1),
             random.uniform(0, 1))
    return color


def import_data_AMPDs(downsample=1):
    unix_ts, clothes_dryer = read_csv_AMPDs('data/AMPDs/clothes_dryer.csv')
    _, fan_and_thermostat = read_csv_AMPDs('data/AMPDs/fan_and_thermostat.csv')
    _, entertainment = read_csv_AMPDs('data/AMPDs/entertainment.csv')
    _, fridge = read_csv_AMPDs('data/AMPDs/fridge.csv')
    _, heat_pump = read_csv_AMPDs('data/AMPDs/heat_pump.csv')
    _, security_and_network = read_csv_AMPDs(
        'data/AMPDs/security_and_network.csv')
    _, dish_washer = read_csv_AMPDs('data/AMPDs/dish_washer.csv')

    _, basement = read_csv_AMPDs('data/AMPDs/basement.csv')
    _, bedroom1 = read_csv_AMPDs('data/AMPDs/bedroom1.csv')
    _, bedroom2 = read_csv_AMPDs('data/AMPDs/bedroom2.csv')
    _, clothes_washer = read_csv_AMPDs('data/AMPDs/clothes_washer.csv')
    _, dining_room = read_csv_AMPDs('data/AMPDs/dining_room.csv')
    _, electric_oven = read_csv_AMPDs('data/AMPDs/electric_oven.csv')
    _, electronics_workbench = read_csv_AMPDs(
        'data/AMPDs/electronics_workbench.csv')
    _, garage = read_csv_AMPDs('data/AMPDs/garage.csv')
    _, instant_hot_water = read_csv_AMPDs(
        'data/AMPDs/instant_hot_water_unit.csv')
    _, office = read_csv_AMPDs('data/AMPDs/office.csv')
    _, outside = read_csv_AMPDs('data/AMPDs/outside.csv')

    _, main_meter = read_csv_AMPDs('data/AMPDs/total.csv')
    _, main = read_csv_AMPDs('data/AMPDs/total.csv')

    dictionary = {'clothes_dryer': clothes_dryer[::downsample],
                  'dish_washer': dish_washer[::downsample],
                  'fan_and_thermostat': fan_and_thermostat[::downsample],
                  'entertainment': entertainment[::downsample],
                  'fridge': fridge[::downsample],
                  'heat_pump': heat_pump[::downsample]}

    all_dictionary = {'clothes_dryer': clothes_dryer[::downsample],
                      'fan_and_thermostat': fan_and_thermostat[::downsample],
                      'entertainment': entertainment[::downsample],
                      'fridge': fridge[::downsample],
                      'heat_pump': heat_pump[::downsample],
                      'security_and_network':
                          security_and_network[::downsample],
                      'basement': basement[::downsample],
                      'bedroom1': bedroom1[::downsample],
                      'bedroom2': bedroom2[::downsample],
                      'clothes_washer': clothes_washer[::downsample],
                      'dining_room': dining_room[::downsample],
                      'dish_washer': dish_washer[::downsample],
                      'electric_oven': electric_oven[::downsample],
                      'electronics_workbench':
                          electronics_workbench[::downsample],
                      'garage': garage[::downsample],
                      'instant_hot_water': instant_hot_water[::downsample],
                      'office': office[::downsample],
                      'outside': outside[::downsample]}


    noise = None
    for k in AMPDS_NOISY_APP[NOISY_PERCENTAGE]:
        if noise is None:
            noise = [0.0 for _ in range(len(all_dictionary[k]))]
        noise += all_dictionary[k]
    dictionary['noise'] = noise

    return dictionary, all_dictionary, unix_ts[::downsample]

def import_data_SYND(sample=1, n_app=None):
    noise = ["meter2", "meter13", "meter21", "meter22", "meter11", "meter17",
             "meter20", "meter7", "meter10", "meter8", "meter6", "meter19",
             "meter15", "meter12", "meter14", "meter16", "meter18"]
    dictionary = dict()

    total = []
    if n_app is None:
        f = open(str(Path(__file__).parent.parent) + '/data/SYND/total.csv',
                 'r')
        for line in f.readlines():
            total.append(float(line.split('\n')[0]))
        f.close()
    else:
        for i in range(n_app):
            f = open(str(Path(__file__).parent.parent) +
                     '/data/SYND/' + noise[i] + '.csv', 'r')
            if not total:
                for line in f.readlines():
                    total.append(float(line.split('\n')[0]))
            else:
                for j, line in enumerate(f.readlines()):
                    total[j] += float(line.split('\n')[0])
            f.close()
        for i in SELECTED_APPLIANCES_SYND:
            f = open(str(Path(__file__).parent.parent) + '/data/SYND/meter' +
                     str(i) + '.csv', 'r')
            if not total:
                for line in f.readlines():
                    total.append(float(line.split('\n')[0]))
            else:
                for j, line in enumerate(f.readlines()):
                    total[j] += float(line.split('\n')[0])
            f.close()

    for i in SELECTED_APPLIANCES_SYND:
        f = open(str(Path(__file__).parent.parent) + '/data/SYND/meter' +
                 str(i) + '.csv', 'r')
        dictionary['meter' + str(i)] = []
        for line in f.readlines():
            dictionary['meter' + str(i)].append(float(line.split('\n')[0]))
        f.close()

    app = dict()
    for k, v in dictionary.items():
        app[k] = v[::sample]
    return app, total[::sample]


def import_data_REFIT(house):
    h = "house" + str(house) + ".csv"
    df = pd.read_csv(Path(__file__).parent.parent / "data/REFIT" / h)

    all_dictionary = dict()
    dictionary = dict()
    total = []
    ts = []

    t = int(mktime(datetime.strptime(START_REFIT[house], "%Y-%m-%d %H:%M:%S")
                   .timetuple())) + 7200
    for e in df.iloc[:, 1]:
        total.append(e)
        ts.append(t)
        t += 60
    for i in range(1, 10):
        all_dictionary['Appliance' + str(i)] = []
        for e in df.iloc[:, i+1]:
            all_dictionary['Appliance' + str(i)].append(e)

    for selected in SELECTED_APPLIANCES_REFIT[house]:
        dictionary["Appliance" + str(selected)] = \
            all_dictionary["Appliance" + str(selected)]

    plot_contributions(all_dictionary, "plots/REFIT/" + str(house) + "/pie.png",
                       noise=False)

    return dictionary, total, ts


def import_data_UKDALE2(start, end, sample=1):
    data = DataSet(Path(__file__).parent.parent / 'data/UKDALE/ukdale.h5')
    data.set_window(start=start, end=end)
    meter = data.buildings[2].elec.submeters()
    mains = data.buildings[2].elec.mains()

    return sample_and_align_group2(mains, meter.meters, sample=sample*60)


def import_data_UKDALE1(sample=1):
    apps = dict()
    list_dir = [2, 3, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18, 19, 21, 23, 26, 27, 28,
                29, 31, 32, 33, 34]
    for l in list_dir:
        f = open(str(Path(__file__).parent.parent) + '/data/UKDALE/1/meter' +
                 str(l) + '.csv', 'r')
        apps['meter' + str(l)] = []
        for line in f.readlines()[:1440 * 120:sample]:
            apps['meter' + str(l)].append(float(line.split('\n')[0]))

    total = [0 for _ in range(1440 * 120)]
    for k, v in apps.items():
        for i, e in enumerate(v[:len(total)]):
            total[i] += e
    return total, apps

def sample_and_align_group1(main, meters, sample=60):
    main = main.power_series_all_data(sample_period=sample)
    not_taken = []
    nt = [4, 10, 14, 20, 24, 30]
    for i in nt:
        not_taken.append("meter" + str(i))
    for i in range(35, 54):
        not_taken.append("meter" + str(i))
    apps = dict()
    metered = None
    last_ixs = None
    for m in meters:
        if m.key.split('/')[-1] in not_taken:
            continue
        app = m.power_series_all_data(sample_period=sample)
        ixs = main.index.intersection(app.index)
        main = main.loc[ixs]
        apps[m.key.split('/')[-1]] = app[ixs].values

        if last_ixs is not None:
            a = False
            if len(ixs) != len(last_ixs):
                print('WARNING\nLength different: ', m.key)
                a = True
            for i in range(len(ixs)):
                if ixs[i] != last_ixs[i] and not a:
                    print('WARNING\nDiffer: ', m.key, ixs[i], last_ixs[i])
                    a = True
        last_ixs = ixs
        if metered is None:
            metered = [0 for _ in range(len(app[ixs].values))]
        if len(app[ixs].values) == len(metered):
            metered += app[ixs].values

    for k, v in apps.items():
        f = open(str(Path(__file__).parent.parent) + "/data/UKDALE/1/" + k +
                 ".csv", "w")
        for e in v:
            f.write(str(e) + '\n')
        f.close()
    return apps

def sample_and_align_group2(main, meters, sample=60):
    main = main.power_series_all_data(sample_period=sample)
    apps = dict()
    metered = None
    last_ixs = None
    for m in meters:
        app = m.power_series_all_data(sample_period=sample)
        ixs = main.index.intersection(app.index)
        main = main.loc[ixs]
        apps[m.key.split('/')[-1]] = app[ixs].values

        if last_ixs is not None:
            if len(ixs) != len(last_ixs):
                print('WARNING\nLength different: ', m.key)
            for i in range(len(ixs)):
                if ixs[i] != last_ixs[i]:
                    print('WARNING\nDiffer: ', m.key, ixs[i], last_ixs[i])
        last_ixs = ixs
        if metered is None:
            metered = [0 for _ in range(len(app[ixs].values))]
        if len(app[ixs].values) == len(metered):
            metered += app[ixs].values

    total = main.values
    for i, e in enumerate(total):
        if e < metered[i]:
            total[i] = metered[i]
    apps["noise"] = total - metered
    return total, apps, last_ixs


def plot_contributions(dictionary, path, noise=False):
    labels = []
    counts = []
    for k, v in dictionary.items():
        if not noise and k == 'noise':
            continue
        labels.append(k)
        counts.append(np.sum(v))

    f = open("perc", "w")
    for i, c in enumerate(counts):
        s = str(labels[i]) + ": " + "{:.4f}".format(c / np.sum(counts)) + "\n"
        f.write(s)
    f.close()
    _, ax = plt.subplots()
    ax.pie(counts, labels=labels)
    plt.savefig(fname=path, format='png')


def scale_array(array):
    mn = np.min(array)
    mx = np.max(array)
    a = mn
    b = 1
    scaled = []
    for x in array:
        scaled_x = (b - a) * (x - mn) / (mx - mn) + a
        scaled.append(scaled_x)
    return scaled
