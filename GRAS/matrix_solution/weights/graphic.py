import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

n = 0
a = 33000

with open('id.dat') as file:
	id_arr = list(map(int, file.read().split()))

with open('spikes_pre.dat') as file:
	spikes_pre_arr = np.array(list(map(float, file.read().split())))

with open('spikes_post.dat') as file:
	spikes_post_arr = np.array(list(map(float, file.read().split())))

with open('weights.dat') as file:
	weight = list(map(float, file.read().split()))


times = np.array(range(len(spikes_pre_arr))) * 0.025

fig = plt.figure()

ax1 = fig.add_subplot(211)
ax1.xaxis.set_major_locator(ticker.MultipleLocator())
ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

ax1.scatter(times[spikes_pre_arr == -1], spikes_pre_arr[spikes_pre_arr == -1], label="pre")
ax1.scatter(times[spikes_post_arr == 1], spikes_post_arr[spikes_post_arr == 1], label="post")
plt.legend()

ax2 = fig.add_subplot(212, sharex=ax1)
ax2.plot(times, weight)

plt.text(17, -0.7, f'pre_ID = {id_arr[0]}')
plt.text(17, 1.3, f'post_ID = {id_arr[1]}')

fig.subplots_adjust(top=0.8)

plt.savefig('graphic.png', dpi = 800)
# plt.show()
