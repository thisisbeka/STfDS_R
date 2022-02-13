from quantiles import MRL98_algo
import matplotlib.pyplot as plt

def plot_fig(x, y, name, xlabel, ylabel, title):
    for i in range(len(x)):
        plt.plot(x[i], y[i], label = name[i], color = 'red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    locs, labels=plt.xticks()
    x_ticks = []
    new_xticks=[0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    plt.xticks(locs, new_xticks)
    locs, labels=plt.yticks()
    y_ticks = []
    new_yticks=[4, 5, 6, 7, 8, 9, 10, 11]
    plt.yticks(locs, new_yticks)
    plt.show()

samples = []
for x in range(1, 3):
    samples.append({"n": 10**5, "b": 10, "k":500, "q": x/100.0})

x = []
y = []
for smpl in samples:
    quantile, eps, time, memory = MRL98_algo(smpl["b"], smpl["k"], smpl["q"])
    x.append(smpl["q"])
    y.append(quantile)

plot_fig( x, y, ["Threshold"], "epsilon", "log (N) to base 10", "Threshold value for N for 99.99% confidence")
