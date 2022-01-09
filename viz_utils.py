
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def bar_plot(title, labels, data):
    ind = np.arange(len(labels))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(ind, data)
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(labels)


def viz_pca_components(pca):
    fig = plt.figure()
    fig.suptitle('MNIST Principal components')
    plot_dim = int(np.ceil(np.sqrt(pca.n_components_)))
    idx = 1
    for component in pca.components_:
        img = 255*np.reshape(component, (28,28))
        ax = fig.add_subplot(plot_dim, plot_dim, idx)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.imshow(img)
        idx=idx+1


def viz_cm_examples(cm_examples):
    fig = plt.figure()
    fig.supxlabel('predicted classes')
    fig.supylabel('actual classes')
    for y in range(10):
        for pred in range(10):
            img = cm_examples[y][pred]
            i = y*10+pred
            ax = fig.add_subplot(10,10,i+1)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            img = 255*img.reshape(28,28)
            ax.imshow(img)


def viz_img_block(imgs):
    fig = plt.figure()
    plt.subplots_adjust(wspace=0, hspace=0)


    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            img = imgs[i,j]
            idx = i*imgs.shape[1]+j
            ax = fig.add_subplot(imgs.shape[0], imgs.shape[1], idx+1)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.set_axis_off()
            ax.imshow(255*img, cmap='gray_r')


def animate_imgs(imgs, filename='image_generation.gif'):
    fig, ax = plt.subplots(figsize=(2,2))
    fig.tight_layout(pad=0.05)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    aximg = ax.imshow(255*imgs[0], cmap='gray', animated=True)

    def update(i):
        aximg.set_array(255*imgs[i])
        return aximg,

    ani = FuncAnimation(fig, update, frames=imgs.shape[0], interval=40, blit=True)
    ani.save(filename=filename)
    plt.show()

