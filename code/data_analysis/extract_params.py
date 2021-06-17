import math

from data_analysis.analize_data import *

def evaluate_max_mean_consumptions(dictionary, percentile, plot_lib, start=0,
                                   end=365, T=1440):
    """
    Evaluate the maximum consumption over a MINUTES_IN_A_DAY interval of time
    using the percentile.

    :param T: time horizon
    :param plot_lib: path where to plot
    :param start: first day to consider for evaluation of the statistics
    :param end: last day to consider for evaluation of the statistics
    :param dictionary: dictionary containing all the time series
    :param percentile: number of the percentile to be chosen
    :return: a value (i.e. the percentile of the consumption) per appliance
    """
    perc = dict()
    for k, v in dictionary.items():
        if k == "noise":
            continue
        count = 0
        consumptions = []
        values = v[start*T:end*T]
        for i, elem in enumerate(values):
            count += elem
            if i % T == T - 1:
                consumptions.append(count)
                count = 0

        # Evaluate percentile
        perc[k] = int(np.percentile(consumptions, percentile))
    print("max: ", perc)


def evaluate_max_cons_per_band(dictionary, start, end, T, path, downsample=1,
                               plot_dir="", plot=False):
    print("######## Max per band consumption ########")
    # downsample in minute
    perc = dict()
    for k, v in dictionary.items():
        if k == "noise":
            continue

        perc[k] = []
        consumptions = []
        count = []
        for _ in ENERGY_BAND:
            count.append(0)
            consumptions.append([])
        values = v[start * T:end * T]

        for i, e in enumerate(values):
            # Find index
            index = 0
            t = i % T
            for j, b in enumerate(ENERGY_BAND):
                if t * downsample / 60 >= b:
                    index = j+1
                else:
                    break

            # Update correct counter
            count[index] += e

            # End of the day
            if t == T-1:
                for j, _ in enumerate(ENERGY_BAND):
                    consumptions[j].append(count[j])
                    count[j] = 0

        for i, _ in enumerate(ENERGY_BAND):
            perc[k].append(int(np.percentile(consumptions[i], PERCENTILE)))

        band_day = consumptions[0] + consumptions[2]
        band_night = consumptions[1]
        if plot:
            plot_histogram(band_day, plot_dir + "max_cons_" + k + "_band_day",
                           path)
            plot_histogram(band_night, plot_dir + "max_cons_" + k +
                           "_band_night", path)
        print("Appliance ", k, " band day:",
              math.ceil(np.percentile(band_day, PERCENTILE)))
        print("Appliance ", k, " band night:",
              math.ceil(np.percentile(band_night, PERCENTILE)))


def evaluate_l_and_s(dictionary, drop, days, T):
    """
    Evaluate the times that the appliance is on over the whole data set minute
    by minute.

    :param T: time horizon
    :param days: days to select for statistic evaluation
    :param drop: days to be dropped before beginning count
    :param dictionary: dictionary containing all the time series
    :return: an array per appliance of MINUTES_IN_A_DAY length and the parameter
             l of that appliance
    """

    threshold = 10  # 10 watt threshold
    s = dict()
    l = dict()
    total = 0
    for k, v in dictionary.items():
        counters = [0] * T
        for i, elem in enumerate(v[T*drop:T*drop + T*days]):
            index = i % T
            if elem > threshold:
                counters[index] += 1
        s[k] = counters
        total += np.sum(counters)

    for k, v in s.items():
        if k == 'noise':
            continue
        app = np.sum(v)
        if app <= 0:
            app = 1
        l[k] = (days - drop) * T / app
    return s, l

def evaluate_w(dictionary, path_config, path_states, drop, days, T):
    w = dict()
    read_ps = read_power_states(path_config, path_states)
    i = 0
    maximum = 0
    for k, v in dictionary.items():
        if k == "noise":
            continue

        # Fix power state
        ps = np.array([0] + read_ps[i])
        i += 1

        count = 0
        last_index = 0
        for elem in v[drop*T: days*T]:
            absolute_val_array = np.abs(ps - elem)
            smallest_difference_index = absolute_val_array.argmin()
            if last_index != smallest_difference_index:
                count += 1
                last_index = smallest_difference_index
        w[k] = count
        if count > maximum:
            maximum = count
    for k, v in w.items():
        if v <= 0:
            v = 1
        w[k] = ((days - drop) * T) / v
    return w

def evaluate_u_bound(dictionary, path_config, path_states, drop, days, T,
                     path="", plot=False):
    print("######## Max in state transition number ########")
    u_bound = dict()
    read_ps = read_power_states(path_config, path_states)
    i = 0
    for k, v in dictionary.items():
        if k == "noise":
            continue

        # Fix power state
        ps = np.array([0] + read_ps[i])
        bounds = [[] for _ in read_ps[i]]
        count = [0 for _ in read_ps[i]]
        last_index = 0
        for j, elem in enumerate(v[drop*T: days*T]):
            absolute_val_array = np.abs(ps - elem)
            smallest_difference_index = absolute_val_array.argmin()
            if last_index != smallest_difference_index:
                if smallest_difference_index != 0:
                    count[smallest_difference_index - 1] += 1
                last_index = smallest_difference_index
            if j % T == T - 1:
                for l in range(len(read_ps[i])):
                    bounds[l].append(count[l])
                count = [0 for _ in read_ps]
                last_index = 0
        u_bound[k] = bounds
        i += 1
        for j, array in enumerate(bounds):
            if plot:
                plot_histogram(array, "max_u_" + k + "_state" + str(j), path)
            print("Appliance ", k, " state ", j, ": ",
                  math.ceil(np.percentile(array, PERCENTILE)))


def evaluate_min_max_times(dictionary, power_states, start, end, T, path="",
                           plot=False):
    print("######## Min and max time in state ########")
    i = 0
    time_in_state = dict()
    for k, v in dictionary.items():
        if k == "noise":
            continue
        count = 1
        last_index = 0
        time_in_state[k] = [[] for _ in power_states[i]]
        adj_ps = np.concatenate(([0], power_states[i]), axis=0)
        for elem in v[start*T: end*T]:
            absolute_val_array = np.abs(adj_ps - elem)
            smallest_difference_index = absolute_val_array.argmin()
            if last_index == smallest_difference_index:
                count += 1
            else:
                if last_index != 0:
                    time_in_state[k][last_index - 1].append(min(T, count))
                count = 1
                last_index = smallest_difference_index
        for j in range(len(power_states[i])):
            if len(time_in_state[k][j]) != 0:
                if plot:
                    plot_histogram(time_in_state[k][j], "time_in_state_" + k +
                                   "_state" + str(j), path)
                print("Appliance ", k, " state ", power_states[i][j], ": ",
                      math.floor(np.percentile(time_in_state[k][j], 100 -
                                               PERCENTILE)), "\t",
                      math.ceil(np.percentile(time_in_state[k][j], PERCENTILE)))
        i += 1
