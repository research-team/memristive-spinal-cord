import numpy as np
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys

red = None
black = None

def get_mean(data):
    meaned = []
    for i, a in enumerate(data):
        if i < 20:
            meaned.append(a)
        else:
            s = 0
            for j in range(1, 21):
                s += data[i - j]
            meaned.append(s / 20)

    return meaned


def check_argv(insole_name):
    if insole_name == "red":
        # first - red, second - black
        return [None, 'legendonly']
    elif insole_name == "black":
        return ['legendonly', None]
    elif insole_name == "all":
        return [None, None]



if __name__ == "__main__":
    red = []
    black = []

    with open("/home/ann/Downloads/Telegram Desktop/red.txt") as f:
        for line in f:
            red.append(list(map(float, line.split())))

    with open("/home/ann/Downloads/Telegram Desktop/black.txt") as f:
        for line in f:
            black.append(list(map(float, line.split())))

    red = np.array(red).T
    black = np.array(black).T

    ankle_right = black[11][100:]
    knee_right = black[3][100:]
    hip_right = black[10][100:]
    united_right = black[16][100:]

    min_angles_right = black[17]
    max_angles_right = black[18]

    ankle_left = red[14][100:]
    knee_left = red[10][100:]
    hip_left = red[15][100:]
    united_left = red[16][100:]

    min_angles_left = red[17]
    max_angles_left = red[18]

    try:
        red, black = check_argv(insole_name=sys.argv[1])
    except(IndexError):
        print("red? black? all?")
        sys.exit()

    fig = make_subplots(rows=3, cols=1, subplot_titles=["Ankle", "Knee", "Hip"])

    fig.add_trace(go.Scatter(x=np.arange(len(ankle_left)), y=get_mean(ankle_left), name="ankle_left",
                             visible=red), 1, 1)
    fig.add_trace(go.Scatter(x=np.arange(len(ankle_right)), y=get_mean(ankle_right), name="ankle_right",
                             visible=black), 1, 1)

    fig.add_trace(go.Scatter(x=np.arange(len(knee_left)), y=get_mean(knee_left), name="knee_left",
                             visible=red), 2, 1)
    fig.add_trace(go.Scatter(x=np.arange(len(knee_right)), y=get_mean(knee_right), name="knee_right",
                             visible=black), 2, 1)

    fig.add_trace(go.Scatter(x=np.arange(len(hip_left)), y=get_mean(hip_left), name="hip_left",
                             visible=red), 3, 1)
    fig.add_trace(go.Scatter(x=np.arange(len(hip_right)), y=get_mean(hip_right), name="hip_right",
                             visible=black), 3, 1)

    fig.update_layout(legend_orientation="h",
                      margin=dict(l=0, r=0, t=30, b=0),
                      legend=dict(x=.5, xanchor="center"))
    fig.update_traces(hoverinfo="all", hovertemplate="x: %{x}<br>y: %{y}")
    fig.show()
