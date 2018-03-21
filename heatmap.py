import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# a and b requires 1 x 36 1d numpy array
def plot_heatmap(a, b, title='title', saveLoc='temp'):

    a = a.reshape((6,6))
    b = b.reshape((6,6))

    fig, (ax1, ax2) = plt.subplots(1, 2)

    h1 = sns.heatmap(a,cmap="magma",cbar=False,ax=ax1)
    h1.set_title("Attention 1")
    h1.invert_yaxis()
    h1.set_xlabel('')
    h1.set_ylabel('')

    h2 = sns.heatmap(b,cmap="magma",ax=ax2)
    h2.set_title("Attention 1")
    h2.invert_yaxis()
    h2.set_xlabel('')
    h2.set_ylabel('')

    plt.show()

if __name__ == '__main__':
    heatLoc = "example"
    a = np.random.random(36)
    b = np.random.random(36)

    plot_heatmap(a, b, "example", heatLoc)
