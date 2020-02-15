import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
import numpy as np


# plot feature layer
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize Feature Layer using t-SNE')
    plt.show()
    plt.pause(10000)


def plot_tsne(gf, g_pids):
    new_gf = np.asarray(gf)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    X = []
    Y = []
    for index in range(new_gf.shape[0]):
        if g_pids[index] in [1, 721, 156, 1317, 727, 521, 731, 92, 218, 94]:
            X.append(new_gf[index, :])
            Y.append(g_pids[index])
    Y = np.asarray(Y)
    low_dim_embs = tsne.fit_transform(X)
    plot_with_labels(low_dim_embs, Y)
