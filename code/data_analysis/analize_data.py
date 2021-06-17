import pandas as pd

from data_analysis.extract_params import *
from util.util import *

global N


def plot_days(dictionary, plot_dir, T, add_percentage=True, limit=None):
    for k, v in dictionary.items():
        color = random_rgb()
        if k == "noise" and add_percentage:
            k = k + str(NOISY_PERCENTAGE)

        # Plot data
        if limit is None:
            limit = LIMIT
        for i in range(limit):
            start = i * T
            create_dir(plot_dir + "/first_" + str(limit) + "days/" + k)
            path = plot_dir + "/first_" + str(limit) + "days/" + k
            path = path + "/day" + str(i) + ".png"
            if len(v[start:start + T]) < T:
                break
            plot_data(x=range(T),
                      y=v[start:start + T],
                      color=color,
                      name=k,
                      show=False,
                      clean=True,
                      path=path)


def plot_from_dictionary(dictionary, total, plot_dir, days, drop, T, limit=0):
    create_dir(plot_dir + "/total" + str(NOISY_PERCENTAGE))
    for i in range(drop, days + limit):
        start = i * T
        path = plot_dir + "/total" + str(NOISY_PERCENTAGE)
        path = path + "/day" + str(i - drop + 1) + ".png"
        plt.clf()
        color = random_rgb()
        plt.plot(range(T), total[start:start + T], label="total",
                 color=color)
        plt.legend()
        plt.title("Sequence " + str(i - drop + 1))
        plt.xlabel("# samples")
        plt.ylabel("Power [Watt]")
        plt.savefig(fname=path, format='png')

    for k, v in dictionary.items():
        color = random_rgb()

        # Plot data
        for i in range(drop, days + limit):
            start = i * T
            create_dir(plot_dir + "/" + k)
            path = plot_dir + "/" + k + "/day" + str(i - drop + 1) + ".png"
            plot_data(x=range(T),
                      y=v[start:start + T],
                      color=color,
                      name=k,
                      show=False,
                      clean=True,
                      path=path)
        if k != "noise":
            # Plot histograms
            plt.clf()
            plt.hist(v, bins=100, range=(10, 5010))
            plt.title("Power consumption histogram for " + k)
            path = plot_dir + "/" + k + "/hist.png"
            plt.savefig(path, format="png")

def evaluate_penalties(dictionary, plot_dir, drop, days, base_path, T,
                       downsample):
    delete_file(base_path / 's')
    f = open(base_path / 's', 'a')

    s, l = evaluate_l_and_s(dictionary, drop, days, T=T)
    order = []
    for k, v in s.items():
        if k == 'noise':
            continue
        order.append(k)

        # Append results to file
        normalized = []
        for elem in v[:T * days]:
            normalized.append(elem / days)

        for elem in normalized:
            f.write(str(elem) + "\t")
        f.write("\n")

    f.write("^^^ order = " + str(order) + "\n")
    f.close()

    delete_file(base_path / 'l')
    f = open(base_path / 'l', 'a')
    order = []
    for k, v in l.items():
        order.append(k)
        f.write(str(v) + "\n")
    f.write("^^^ order = " + str(order) + "\n")
    f.close()

    if downsample == 1:
        w = evaluate_w(dictionary, base_path / 'config',
                       base_path / 'power_states_power',  drop, days, T=T)
    else:
        w = evaluate_w(dictionary, base_path / 'config',
                       base_path / 'power_states_power_15', drop, days, T=T)
    delete_file(base_path / 'w')
    f = open(base_path / 'w', 'a')
    order = []
    for k, v in w.items():
        order.append(k)
        f.write(str(v) + "\n")
    f.write("^^^ order = " + str(order) + "\n")
    f.close()

def evaluate_total(dictionary):
    total = None
    for k, v in dictionary.items():
        if total is None:
            total = [0.0 for _ in range(len(v))]
        total += v
    noise = dictionary["noise"]
    print("Noise percentage: ", np.sum(noise) / np.sum(total))
    return total


def read_power_states(path_config, path_states):
    global N
    f = open(path_config, 'r')
    N = int(f.readline())
    f.close()
    f = open(path_states, 'r')
    power_states = []
    for i in range(N):
        power_states.append(
            [int(s) for s in f.readline().split() if s.isdigit()])
    return power_states


def evaluate_clipped(total, power_states, base_path, days, drop, T):
    for day in range(days - drop):
        path = base_path + 'clipped_variables' + str(day + 1) + '.dat'
        delete_file(Path(__file__).parent.parent / path)
        f = open(Path(__file__).parent.parent / path, 'a')
        count = 0
        f.write("let clipped := {")
        string = ""
        for t in range(day * T, (day + 1) * T):
            for i in range(len(power_states)):
                for j in range(len(power_states[i])):
                    if total[T * drop + t] < power_states[i][j] * \
                            CLIPPED_PARAM:
                        string = string + '(' + str(t - day * T + 1) + ',' + \
                                 str(i + 1) + ',' + str(j + 1) + '),\n'
                        count = count + 1
        f.write(string[:-2])
        f.write("};")
        f.close()
        print("Variables clipped to zero in day ", day + 1, ": ", count)


def plot_noise_percentage(dictionary, plot_dir, total, T, limit=30):
    perc = []
    for i, elem in enumerate(dictionary['noise'][:limit * T]):
        if total[i] > 0:
            perc.append(elem / total[i])
        else:
            perc.append(0.0)

    plt.clf()
    color = random_rgb()
    plt.plot(range(len(perc)), perc, label="noise_percentage", color=color)
    plt.legend()
    plt.xlabel("# samples")
    plt.ylabel("percentage")
    plt.savefig(plot_dir + "/noise_percentage.png", format="png")

def evaluate_delta(dictionary, power_states, start, size):
    i = 0
    delta = dict()
    for k, v in dictionary.items():
        to_print = []
        if k == 'noise' or k == 'main':
            continue
        ps = np.array([0] + power_states[i])
        for elem in v[start[k]:start[k] + size]:
            absolute_val_array = np.abs(ps - elem)
            smallest_difference_index = absolute_val_array.argmin()
            found = False

            for j, _ in enumerate(ps):
                if j == smallest_difference_index and not found:
                    found = True
                    to_print.append(1)
                else:
                    to_print.append(0)
        delta[k] = to_print
        i += 1
    return delta


def plot_downsample(dictionary, start, end, T, downscale_rate=None):
    if downscale_rate is None:
        downscale_rate = [1, 10, 15]
    for downscale in downscale_rate:
        for k, v in dictionary.items():
            if k != "noise":
                continue
            else:
                x = (end - start) * T // downscale
                f_x = v[start * T:end * T:downscale]
                plt.clf()
                plt.plot(range(x),
                         f_x)
                plt.title(k + " downsampled at " + str(downscale) + " minutes")
                plt.xlabel("# sample")
                plt.ylabel("Watt [W]")
                plt.show()


def plot_noise(noise, T, start, end, downsample):
    perc = [[] for _ in range(T)]
    for i in range(start, end):
        values = noise[(start+i)*T:(start+i+1)*T]
        for j, e in enumerate(values):
            perc[j].append(e / np.max(values))
        plt.clf()
        plt.plot(range(T), values)
        plt.title("Day " + str(i - start))
        plt.savefig("plots/AMPDS/noise/ds_" + str(downsample) + "_perc_"
                    + str(NOISY_PERCENTAGE) + "_day" + str(i - start) + ".png",
                    format="png")

    avg_perc = []
    for array in perc:
        avg_perc.append(np.mean(array))

    plt.clf()
    plt.plot(range(T), avg_perc)
    plt.title("Average percentage")
    plt.savefig("plots/AMPDS/noise/avg_perc.png", format="png")


def evaluate_sparse_parameters(apps, path, start, end, T):
    f = open(str(path) + "/power_states_power", 'r')
    ps = []
    for i in range(N):
        ps.append([int(s) for s in f.readline().split() if s.isdigit()])
    f.close()

    f = open(str(path) + "/w", "w")
    param = dict()
    i = 0
    for k, v in apps.items():
        if k == "noise":
            continue
        param[k] = [[0 for _ in range(T)] for _ in ps[i]]
        for d in range(end - start):
            for t in range(T):
                absolute_val_array = np.abs(np.add(ps[i],
                                                   -v[(start + d) * T + t]))
                j_star = absolute_val_array.argmin()
                param[k][j_star][t] += 1
        for j in range(len(ps[i])):
            for t in range(T):
                if param[k][j][t] < 1:
                    param[k][j][t] = 1
                f.write(str(1 / param[k][j][t]) + " ")
            f.write('\n')
        i += 1
    f.close()




def analize_AMPDS_data(downsample=1):
    T = 1440 // downsample
    # Import data from csv
    dictionary, all_dictionary, _ = import_data_AMPDs(downsample=downsample)
    plot_dir = "plots/AMPDS"

    ############################################################################
    # Plot pie chart of contribution of each appliance                         #
    ############################################################################
    plot_contributions(all_dictionary, plot_dir + "/pie.png", noise=False)

    ############################################################################
    # Evaluate aggregated value for selected appliances                        #
    ############################################################################
    total = evaluate_total(dictionary)

    f1 = open("ampds_aggr.csv", "w")
    f1.write("Time,aggregate\n")
    f2 = open("ampds_disaggr.csv", "w")
    f2.write("Time")
    for k in dictionary.keys():
        if k == "noise":
            continue
        f2.write("," + k)
    f2.write("\n")
    ts = datetime.fromtimestamp(1333404000)
    for i in range(DROP_AMPDs * T, DAYS_AMPDs * T):
        f1.write(ts.strftime("%Y-%m-%d %H:%M:%S") + "," + str(int(total[i])) +
                 "\n")
        f2.write(ts.strftime("%Y-%m-%d %H:%M:%S"))
        for k in dictionary.keys():
            if k == "noise":
                continue
            f2.write("," + str(int(dictionary[k][i])))
        f2.write("\n")
        ts = ts + timedelta(minutes=1)
    f1.close()
    f2.close()

    # Write total to file
    delete_file(Path(__file__).parent.parent / 'bqp_formulation/AMPDS/total')
    f = open(Path(__file__).parent.parent / 'bqp_formulation/AMPDS/total',
             'a')

    for elem in total:
        f.write(str(elem) + '\n')
    f.close()

    for i in range(DROP_AMPDs, DAYS_AMPDs):
        plt.clf()
        plt.plot(total[i * T:(i + 1) * T], label="total")
        for k, v in dictionary.items():
            if k == "noise":
                continue
            plt.plot(v[i * T:(i + 1) * T], label=k)
        plt.legend()
        plt.show()

    # Pre-processing stage
    if downsample == 1:
        power_states = read_power_states(
            Path(__file__).parent.parent / 'bqp_formulation/AMPDS/config',
            Path(__file__).parent.parent /
            'bqp_formulation/AMPDS/power_states_power')
    else:
        power_states = read_power_states(
            Path(__file__).parent.parent / 'bqp_formulation/AMPDS/config',
            Path(__file__).parent.parent /
            'bqp_formulation/AMPDS/power_states_power_15')
    evaluate_clipped(total, power_states, 'bqp_formulation/AMPDS/',
                     DAYS_AMPDs, DROP_AMPDs, T)

    # Evaluate min max time in state
    evaluate_min_max_times(dictionary, power_states, TRAIN_START, TRAIN_END, T)

    # Write appliances and delta to file
    if downsample == 1:
        ts = 600
        start = {'clothes_dryer': 12 * T + 800,
                 'dish_washer': 10 * T + 800,
                 'fan_and_thermostat': 9 * T + 800,
                 'entertainment': 10 * T + 1200,
                 'fridge': 9 * T,
                 'heat_pump': 11 * T}
        i = 1
        for k, v in dictionary.items():
            if k == 'noise':
                continue
            p = 'bqp_formulation/AMPDS_PP/appliance' + str(i)
            delete_file(Path(__file__).parent.parent / p)
            f = open(Path(__file__).parent.parent / p, 'a')
            for elem in v[start[k]:start[k] + ts]:
                f.write(str(elem) + '\n')
            f.close()
            i += 1

        delta = evaluate_delta(dictionary, power_states, start, ts)
        i = 1
        for k, v in delta.items():
            if k == 'noise':
                continue
            p = 'bqp_formulation/AMPDS_PP/delta' + str(i)
            delete_file(Path(__file__).parent.parent / p)
            f = open(Path(__file__).parent.parent / p, 'a')
            for idx, elem in enumerate(v):
                f.write(str(elem) + ' ')
                if idx % (len(power_states[i - 1]) + 1) == \
                        len(power_states[i - 1]):
                    f.write('\n')
            i += 1
            f.close()

    evaluate_max_cons_per_band(dictionary, TRAIN_START, TRAIN_END, T,
                               plot_dir + "/", downsample)

    evaluate_penalties(dictionary, plot_dir, TRAIN_START, TRAIN_END,
                       Path(__file__).parent.parent /
                       'bqp_formulation/AMPDS', T=T, downsample=downsample)

    evaluate_u_bound(dictionary, Path(__file__).parent.parent /
                     'bqp_formulation/AMPDS/config',
                     Path(__file__).parent.parent /
                     'bqp_formulation/AMPDS/power_states_power',
                     TRAIN_START, TRAIN_END, T)

def analize_UKDALE_data(house, sample=1):
    T = 1440 // sample

    if house == 1:
        total, apps = import_data_UKDALE1(sample)
    else:
        total, apps, _ = import_data_UKDALE2(START_UKDALE_TS[house],
                                             END_UKDALE_TS[house],
                                             sample=sample)

    plot_dir = "plots/UKDALE/" + str(house)
    plot_contributions(apps, plot_dir + "/pie.png", False)

    dictionary = dict()
    metered = [0.0 for _ in total]
    for k, v in apps.items():
        if k == "noise":
            continue
        if k in SELECTED_APPLIANCES_UKDALE[house]:
            dictionary[k] = v
            for i, e in enumerate(v):
                metered[i] += e
    noise = []
    for i in range(len(total)):
        noise.append(total[i] - metered[i])
    dictionary['noise'] = noise

    f = open("BCH_ukdale1_train.csv", "w")
    f.write("TimeStamp,MAIN")
    for k in dictionary.keys():
        if k == "noise":
            continue
        f.write("," + k)
    f.write("\n")
    ts = 1333404000
    for i in range(TRAIN_START_UKDALE[house] * T, TRAIN_END_UKDALE[house] * T):
        f.write(str(int(ts)) + "," + str(int(total[i])))
        for k in dictionary.keys():
            if k == "noise":
                continue
            f.write("," + str(int(dictionary[k][i])))
        f.write("\n")
        ts += 60
    f.close()

    f = open("BCH_ukdale1_test.csv", "w")
    f.write("TimeStamp,MAIN")
    for k in dictionary.keys():
        if k == "noise":
            continue
        f.write("," + k)
    f.write("\n")
    for i in range(DROP_UKDALE[house] * T, DAYS_UKDALE[house] * T):
        f.write(str(int(ts)) + "," + str(int(total[i])))
        for k in dictionary.keys():
            if k == "noise":
                continue
            f.write("," + str(int(dictionary[k][i])))
        f.write("\n")
        ts += 60
    f.close()

    for i in range(DROP_UKDALE[house], DAYS_UKDALE[house]):
        plt.clf()
        plt.plot(total[i * T:(i + 1) * T], label="total")
        for k, v in dictionary.items():
            if k == "noise":
                continue
            plt.plot(v[i * T:(i + 1) * T], label=k)
        plt.legend()
        plt.title("Day " + str(i))
        plt.show()

    # Write total to file
    delete_file(Path(__file__).parent.parent /
                'bqp_formulation/UKDALE' / str(house) / 'total')
    f = open(Path(__file__).parent.parent / 'bqp_formulation/UKDALE' /
             str(house) / 'total', 'a')

    for elem in total:
        f.write(str(elem) + '\n')
    f.close()

    # Pre-processing stage
    power_states = read_power_states(
        Path(__file__).parent.parent / 'bqp_formulation/UKDALE' /
        str(house) / 'config', Path(__file__).parent.parent /
        'bqp_formulation/UKDALE' / str(house) / 'power_states_power')

    evaluate_clipped(total, power_states, 'bqp_formulation/UKDALE/' +
                     str(house) + '/', DAYS_UKDALE[house], DROP_UKDALE[house],
                     T)

    # Evaluate min max time in state
    evaluate_min_max_times(dictionary, power_states, TRAIN_START_UKDALE[house],
                           TRAIN_END_UKDALE[house], T)

    # Write appliances and delta to file
    ts = {1: 600,
          2: 1440}
    start = {1: {'meter2': 29 * T + 400,
                 'meter5': 26 * T + 400,
                 'meter6': 29 * T + 400,
                 'meter9': 26 * T + 400,
                 'meter12': 28 * T + 200},
             2: {'meter5': 28 * T,
                 'meter8': 28 * T,
                 'meter13': 28 * T,
                 'meter14': 28 * T,
                 'meter18': 28 * T}}
  
    evaluate_max_cons_per_band(dictionary, TRAIN_START_UKDALE[house],
                               TRAIN_END_UKDALE[house],
                               T, plot_dir + "/", sample)

    evaluate_penalties(dictionary, plot_dir, TRAIN_START_UKDALE[house],
                       TRAIN_END_UKDALE[house], Path(__file__).parent.parent /
                       'bqp_formulation/UKDALE' / str(house), T=T,
                       downsample=sample)

    evaluate_u_bound(dictionary, Path(__file__).parent.parent /
                     'bqp_formulation/UKDALE' / str(house) / 'config',
                     Path(__file__).parent.parent /
                     'bqp_formulation/UKDALE' / str(house) /
                     'power_states_power', TRAIN_START_UKDALE[house],
                     TRAIN_END_UKDALE[house], T)

def analize_SYND_data(sample=60):
    napp = 0
    T = 1440 * 60 // sample
    # Import data from csv
    dictionary, total = import_data_SYND(sample // 60, n_app=napp)
    plot_dir = "plots/SYND"

    # Print noise percentage
    count = 0
    for array in dictionary.values():
        count += np.sum(array)
    print("Noise percentage: ", 1 - count / np.sum(total))

    # Write total to file
    delete_file(Path(__file__).parent.parent / 'bqp_formulation/SYND/total')
    f = open(Path(__file__).parent.parent / 'bqp_formulation/SYND/total',
             'a')
    for elem in total:
        f.write(str(max(0.0, elem)) + '\n')
    f.close()

    # Read power states
    power_states = read_power_states(
        Path(__file__).parent.parent / 'bqp_formulation/SYND/config',
        Path(__file__).parent.parent /
        'bqp_formulation/SYND/power_states_power')

    evaluate_clipped(total, power_states, 'bqp_formulation/SYND/',
                     DAYS_SYND, DROP_SYND, T)

    # Evaluate parameters
    evaluate_min_max_times(dictionary, power_states, TRAIN_START_SYND,
                           TRAIN_END_SYND, T, path=plot_dir + "/", plot=False)

    evaluate_max_cons_per_band(dictionary, TRAIN_START_SYND, TRAIN_END_SYND, T,
                               plot_dir + "/", 1, plot=False)

    evaluate_penalties(dictionary, plot_dir, TRAIN_START_SYND, TRAIN_END_SYND,
                       Path(__file__).parent.parent /
                       'bqp_formulation/SYND', T=T, downsample=1)

    evaluate_u_bound(dictionary, Path(__file__).parent.parent /
                     'bqp_formulation/SYND/config',
                     Path(__file__).parent.parent /
                     'bqp_formulation/SYND/power_states_power',
                     TRAIN_START_SYND, TRAIN_END_SYND, T,
                     path=plot_dir + "/", plot=False)

    # Write appliances and delta to file
    if sample == 60:
        ts = 2880
        start = {'meter3': 0,
                 'meter4': 1440,
                 'meter5': 0,
                 'meter9': 0}
        i = 1
        for k, v in dictionary.items():
            if k == 'noise':
                continue
            p = 'bqp_formulation/SYND_PP/appliance' + str(i)
            delete_file(Path(__file__).parent.parent / p)
            f = open(Path(__file__).parent.parent / p, 'a')
            for elem in v[start[k]:start[k] + ts]:
                f.write(str(elem) + '\n')
            f.close()
            i += 1

        delta = evaluate_delta(dictionary, power_states, start, ts)
        i = 1
        for k, v in delta.items():
            if k == 'noise':
                continue
            p = 'bqp_formulation/SYND_PP/delta' + str(i)
            delete_file(Path(__file__).parent.parent / p)
            f = open(Path(__file__).parent.parent / p, 'a')
            for idx, elem in enumerate(v):
                f.write(str(elem) + ' ')
                if idx % (len(power_states[i - 1]) + 1) == \
                        len(power_states[i - 1]):
                    f.write('\n')
            i += 1
            f.close()

def analize_REFIT(downsample, house):
    T = 1440 // downsample
    # Import data from csv
    dictionary, total, ts = import_data_REFIT(house=house)
    plot_dir = "plots/REFIT"

    for i in range(DROP_REFIT[house], DAYS_REFIT[house]):
        plt.clf()
        plt.plot(total[i * T:(i + 1) * T], label="total")
        for k, v in dictionary.items():
            plt.plot(v[i * T:(i + 1) * T], label=k)
        plt.legend()
        plt.title("Day " + str(i))
        plt.show()

    # Print noise percentage
    count = 0
    for array in dictionary.values():
        count += np.sum(array)
    print("Noise percentage: ", 1 - count / np.sum(total))

    # Write total to file
    delete_file(Path(__file__).parent.parent / 'bqp_formulation/REFIT' /
                str(house) / 'total')
    f = open(Path(__file__).parent.parent / 'bqp_formulation/REFIT' /
             str(house) / 'total', 'a')
    for elem in total:
        f.write(str(max(0.0, elem)) + '\n')
    f.close()

    # Read power states
    power_states = read_power_states(Path(__file__).parent.parent /
                 'bqp_formulation/REFIT' / str(house) / 'config',
                 Path(__file__).parent.parent / 'bqp_formulation/REFIT' /
                 str(house) / 'power_states_power')

    evaluate_clipped(total, power_states, 'bqp_formulation/REFIT/' +
                     str(house) + '/', DAYS_REFIT[house], DROP_REFIT[house], T)

    # Evaluate parameters
    evaluate_min_max_times(dictionary, power_states, DROP_REFIT[house] - 30,
                           DROP_REFIT[house], T, path=plot_dir + "/", plot=False)

    evaluate_max_cons_per_band(dictionary, DROP_REFIT[house] - 30,
                               DROP_REFIT[house], T, plot_dir + "/", 1,
                               plot=False)

    evaluate_penalties(dictionary, plot_dir, DROP_REFIT[house] - 30,
                       DROP_REFIT[house], Path(__file__).parent.parent /
                       'bqp_formulation/REFIT' / str(house), T=T,
                       downsample=1)

    evaluate_u_bound(dictionary, Path(__file__).parent.parent /
                     'bqp_formulation/REFIT' / str(house) / 'config',
                     Path(__file__).parent.parent /
                     'bqp_formulation/REFIT' / str(house) /
                     'power_states_power',
                     DROP_REFIT[house] - 30,  DROP_REFIT[house], T,
                     path=plot_dir + "/", plot=False)

    ts = 600
    i = 1
    start = {3: {'Appliance2': 27 * T + 650,
                 'Appliance4': 27 * T + 450,
                 'Appliance5': 27 * T + 450,
                 'Appliance7': 27 * T + 1000},
             9: {'Appliance1': 353 * T + 600,
                 'Appliance3': 346 * T,
                 'Appliance4': 346 * T + 600,
                 'Appliance7': 347 * T + 700}}
    for k, v in dictionary.items():
        if k == 'noise':
            continue
        p = 'bqp_formulation/REFIT_PP/' + str(house) + '/appliance' + str(i)
        delete_file(Path(__file__).parent.parent / p)
        f = open(Path(__file__).parent.parent / p, 'a')
        for elem in v[start[house][k]:start[house][k] + ts]:
            f.write(str(elem) + '\n')
        f.close()
        i += 1

    delta = evaluate_delta(dictionary, power_states, start[house], ts)
    i = 1
    for k, v in delta.items():
        if k == 'noise':
            continue
        p = 'bqp_formulation/REFIT_PP/' + str(house) + '/delta' + str(i)
        delete_file(Path(__file__).parent.parent / p)
        f = open(Path(__file__).parent.parent / p, 'a')
        for idx, elem in enumerate(v):
            f.write(str(elem) + ' ')
            if idx % (len(power_states[i - 1]) + 1) == \
                    len(power_states[i - 1]):
                f.write('\n')
        i += 1
        f.close()


if __name__ == "__main__":
    #analize_AMPDS_data(downsample=1)
    #analize_UKDALE_data(house=2, sample=1)
    #analize_SYND_data(60)
    #analize_REFIT(downsample=1, house=9)
