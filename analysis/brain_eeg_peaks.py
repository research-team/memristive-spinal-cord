import logging
import scipy.io
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import article_plotting

"""
The script to read mat files of the rats brains and mark peaks.
KDE analysis of the peaks.
"""

logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
log = logging.getLogger()

def draw_channel(batch, color):
    """
    Parameters
    ----------
    batch the numpy 1D (flattened) array to draw.
    color the graph color.

    Returns nothing, draws the array as figure.
    -------

    """

    plt.rcParams['figure.figsize'] = [30, 7]
    batch_length = len(batch)
    x = np.linspace(0, batch_length, batch_length)
    # plot
    fig, ax = plt.subplots()
    ax.plot(x, batch, linewidth=1, color=color)
    plt.show()

file_prefix = '../../data/rats/'
#file_path = '2011_05_03_0011.mat'
file_path = '2011_05_03_0003.mat'
#file_path = '2011_05_03_0023.mat'

mat = scipy.io.loadmat(file_prefix+file_path)
log.info("File loaded")

log.info("len lfp[0] %s", len(mat['lfp'][0]))

log.info("len lfp %s", len(mat['lfp']))

lfp = mat['lfp']
"""
Channel to draw
"""
ch = lfp[:,0,:50]
log.debug("Shape of the channel")
log.debug("Shape %s", ch.shape)
log.debug("Flatten len %s", len(ch.flatten()))

ch = lfp[:,0,:50]
"""
slice of the channel 
"""
sl = ch.flatten()[60000:70000]
draw_channel(sl, "teal")

analyser = article_plotting.Analyzer()
## sliced_datasets, dstep, borders, filter_val, tails=False, debug=False
#TODO correct ch to array of channels
dstep = 0.025 #TODO clarify this
borders = 250 #TODO clarify this
filter_val = 0.028 #TODO clarify this
tails = False
debug = True
analyser._get_peaks(ch, dstep, borders, filter_val, tails, debug)
