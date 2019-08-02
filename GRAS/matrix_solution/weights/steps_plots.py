import numpy as np
import matplotlib.pyplot as plt
from matrix_solution.plot_results import draw_slice_borders

muscle = '1_OM1_2_E'
save_folder = f"/home/yuliya/Desktop/STDP/GRAS/matrix_solution/weights/{muscle}_plots"


def plot(title, skin_stim_time, voltage, g_exc, g_inh, spikes):
    step = 0.025
    sim_time = 275
    shared_x = np.arange(len(voltage)) * step

    plt.figure(figsize=(16, 9))
    plt.suptitle(title)

    # 1
    ax1 = plt.subplot(311)
    draw_slice_borders(sim_time, skin_stim_time)

    plt.plot(shared_x, voltage, color='b')
    plt.plot(spikes, [0] * len(spikes), '.', color='r', alpha=0.7)

    xticks = range(0, sim_time + 1, 5)

    plt.xlim(0, sim_time)
    plt.ylim(-100, 60)
    plt.xticks(xticks, [""] * sim_time, color='w')
    plt.ylabel("Voltage, mV")
    plt.grid()

    # 2
    ax2 = plt.subplot(312, sharex=ax1)
    draw_slice_borders(sim_time, skin_stim_time)

    plt.plot(shared_x, g_exc, color='r')
    plt.plot(shared_x, g_inh, color='b')
    plt.ylabel("Currents, pA")

    plt.ylim(bottom=0)
    plt.xlim(0, sim_time)
    plt.xticks(xticks, [""] * sim_time, color='w')
    plt.grid()

    # 3
    plt.subplot(313, sharex=ax1)
    draw_slice_borders(sim_time, skin_stim_time)

    plt.hist(spikes, bins=range(sim_time))  # bin is equal to 1ms
    plt.xlim(0, sim_time)
    plt.ylim(bottom=0)
    plt.grid()
    plt.ylabel("Spikes, n")
    plt.ylim(bottom=0)
    plt.xticks(xticks, xticks, rotation=90)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.05, hspace=0.08)

    plt.savefig(f"{save_folder}/{title}.png", format="png", dpi=200)
    plt.close()


def process_data():
    with open(f'{muscle}.dat') as file:
        all_data = file.readlines()
        volt = np.array(list(map(float, all_data[0].split())))
        g_exc = np.array(list(map(float, all_data[1].split())))
        g_inh = np.array(list(map(float, all_data[2].split())))
        spikes = np.array(list(map(float, all_data[3].split())))

    sim_step = 0.025
    leg_step_time = 275
    leg_step_time_in_steps = int(leg_step_time / sim_step)

    for start_i in range(0, len(volt), leg_step_time_in_steps):
        start_ms = start_i * sim_step
        end_ms = (start_i + leg_step_time_in_steps) * sim_step

        v = volt[start_i:start_i + leg_step_time_in_steps]
        g_e = g_exc[start_i:start_i + leg_step_time_in_steps]
        g_i = g_inh[start_i:start_i + leg_step_time_in_steps]
        s = np.array(list(filter(lambda spike: start_ms <= spike <= end_ms, spikes))) - start_ms

        title = f"{start_ms} - {end_ms} ms"
        skin_stim_time = 25

        plot(title, skin_stim_time, v, g_e, g_i, s)

        print(title)


if __name__ == "__main__":
    process_data()


def run():
    process_data()
