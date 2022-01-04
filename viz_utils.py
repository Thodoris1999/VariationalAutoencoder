
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

    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            img = imgs[i,j]
            idx = i*imgs.shape[1]+j
            ax = fig.add_subplot(imgs.shape[0], imgs.shape[1], idx+1)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.imshow(255*img, cmap='gray')


def animate_imgs(imgs):
    fig, ax = plt.subplots()
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    aximg = ax.imshow(255*imgs[0], cmap='gray', animated=True)

    def update(i):
        aximg.set_array(255*imgs[i])
        print(i)
        return aximg,

    ani = FuncAnimation(fig, update, frames=imgs.shape[0], interval=40, blit=True)
    plt.show()



def plot_cv_results_with_stdev(cv_results):
    pass
    Cs_rbf = [1, 10, 100, 1000]
    Cs_linear = [0.01, 0.1, 1, 10, 100, 1000]
    # plot test scores
    gammae_1_score = cv_results['mean_test_score'][0:4]
    gammae_1_score_std = cv_results['std_test_score'][0:4]
    gammae_2_score = cv_results['mean_test_score'][4:8]
    gammae_2_score_std = cv_results['std_test_score'][4:8]
    gammae_3_score = cv_results['mean_test_score'][8:12]
    gammae_3_score_std = cv_results['std_test_score'][8:12]
    gammae_4_score = cv_results['mean_test_score'][12:16]
    gammae_4_score_std = cv_results['std_test_score'][12:16]
    linear_score = cv_results['mean_test_score'][16:22]
    linear_score_std = cv_results['std_test_score'][16:22]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    plt.xlabel('C (regularization parameter)')
    plt.ylabel('accuracy')
    plt.errorbar(Cs_rbf, gammae_1_score, yerr=gammae_1_score_std, label='RBF, gamma=0.1')
    plt.errorbar(Cs_rbf, gammae_2_score, yerr=gammae_2_score_std, label='RBF, gamma=0.01')
    plt.errorbar(Cs_rbf, gammae_3_score, yerr=gammae_3_score_std, label='RBF, gamma=0.001')
    plt.errorbar(Cs_rbf, gammae_4_score, yerr=gammae_4_score_std, label='RBF, gamma=0.0001')
    plt.errorbar(Cs_linear, linear_score, yerr=linear_score_std, label='linear kerneal')
    plt.legend(loc='best')

    # plot fit time
    gammae_1_score = cv_results['mean_fit_time'][0:4]
    gammae_1_score_std = cv_results['std_fit_time'][0:4]
    gammae_2_score = cv_results['mean_fit_time'][4:8]
    gammae_2_score_std = cv_results['std_fit_time'][4:8]
    gammae_3_score = cv_results['mean_fit_time'][8:12]
    gammae_3_score_std = cv_results['std_fit_time'][8:12]
    gammae_4_score = cv_results['mean_fit_time'][12:16]
    gammae_4_score_std = cv_results['std_fit_time'][12:16]
    linear_score = cv_results['mean_fit_time'][16:22]
    linear_score_std = cv_results['std_fit_time'][16:22]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    plt.xlabel('C (regularization parameter)')
    plt.ylabel('train time (seconds)')
    plt.errorbar(Cs_rbf, gammae_1_score, yerr=gammae_1_score_std, label='RBF, gamma=0.1')
    plt.errorbar(Cs_rbf, gammae_2_score, yerr=gammae_2_score_std, label='RBF, gamma=0.01')
    plt.errorbar(Cs_rbf, gammae_3_score, yerr=gammae_3_score_std, label='RBF, gamma=0.001')
    plt.errorbar(Cs_rbf, gammae_4_score, yerr=gammae_4_score_std, label='RBF, gamma=0.0001')
    plt.errorbar(Cs_linear, linear_score, yerr=linear_score_std, label='linear kerneal')
    plt.legend(loc='best')

    # plot test time
    gammae_1_score = cv_results['mean_score_time'][0:4]
    gammae_1_score_std = cv_results['std_score_time'][0:4]
    gammae_2_score = cv_results['mean_score_time'][4:8]
    gammae_2_score_std = cv_results['std_score_time'][4:8]
    gammae_3_score = cv_results['mean_score_time'][8:12]
    gammae_3_score_std = cv_results['std_score_time'][8:12]
    gammae_4_score = cv_results['mean_score_time'][12:16]
    gammae_4_score_std = cv_results['std_score_time'][12:16]
    linear_score = cv_results['mean_score_time'][16:22]
    linear_score_std = cv_results['std_score_time'][16:22]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    plt.xlabel('C (regularization parameter)')
    plt.ylabel('test time (seconds)')
    plt.errorbar(Cs_rbf, gammae_1_score, yerr=gammae_1_score_std, label='RBF, gamma=0.1')
    plt.errorbar(Cs_rbf, gammae_2_score, yerr=gammae_2_score_std, label='RBF, gamma=0.01')
    plt.errorbar(Cs_rbf, gammae_3_score, yerr=gammae_3_score_std, label='RBF, gamma=0.001')
    plt.errorbar(Cs_rbf, gammae_4_score, yerr=gammae_4_score_std, label='RBF, gamma=0.0001')
    plt.errorbar(Cs_linear, linear_score, yerr=linear_score_std, label='linear kerneal')
    plt.legend(loc='best')