from time import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from time import time
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize, maxabs_scale
from sklearn.metrics import confusion_matrix


def t_sne(input_data, input_label, classes, name, n_dim=2):
    """
    :param input_label:(n,)
    :param input_data:  (n, dim)
    :param name: name
    :param classes: number of classes
    :param n_dim: 2d or 3d
    :return: figure
    """
    input_label = input_label.astype(dtype=int)
    t0 = time()
    da = TSNE(n_components=n_dim, init='pca', random_state=0).fit_transform(input_data)  # (n, n_dim)
    # x_min, x_max = np.min(da, 0), np.max(da, 0)
    # da = (da - x_min) / (x_max - x_min)
    da = MinMaxScaler().fit_transform(da)

    figs = plt.figure()
    if n_dim == 3:
        ax = figs.add_subplot(111, projection='3d')
        ax.set_zlim(-0.1, 1.1)
        for i in range(da.shape[0]):
            ax.text(da[i, 0], da[i, 1], da[i, 2], str(input_label[i]),
                    backgroundcolor=plt.cm.Set1(input_label[i] / (classes + 1)),
                    fontdict={'weight': 'bold', 'size': 9})

    else:
        ax = figs.add_subplot(111)
        for i in range(da.shape[0]):
            ax.text(da[i, 0], da[i, 1], str(input_label[i]),
                    backgroundcolor=plt.cm.Set1(input_label[i] / (classes + 1)),
                    fontdict={'weight': 'bold', 'size': 9})
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    title = 't-SNE embedding of %s (time %.2fs)' % (name, (time() - t0))
    plt.title(title)
    return figs


def vis_tSNE(input_data, input_label, classes, vis, name, n_dim=2):
    """
    :param vis: visdom.Visdom()
    :param input_data: (n, m)
    :param input_label: (n,)
    :param classes:
    :param n_dim: int, 2d or 3d
    :param name: str, name of figure
    """
    input_label = input_label.astype(dtype=int)
    if np.min(input_label, axis=-1) < 1:
        input_label += 1
    t0 = time()
    da = TSNE(n_components=n_dim, init='pca', random_state=0).fit_transform(input_data)
    # random_state一定要固定下来，否则每次运行结果都不相同
    da = MinMaxScaler().fit_transform(da)
    y = np.arange(classes)  # or (1, classes+1)
    legends = [str(i) for i in y]
    vis.scatter(X=da, Y=input_label, win=name,
                opts=dict(legend=legends,
                          xtickmin=-0.2,
                          xtickmax=1.2,
                          ytickmin=-0.2,
                          ytickmax=1.2,
                          title='%s-tSNE(time %.2f s)' % (name, time() - t0)))
    print(name+'-tSNE done through visdom!')


def Euclidean_Distance(x, y):
    """
    :param x: n x p   N-example, P-dimension for each example; zq
    :param y: m x p   M-Way, P-dimension for each example, but 1 example for each Way; z_proto
    :return: [n, m]
    """
    n = x.size(0)
    m = y.size(0)
    p = x.size(1)
    assert p == y.size(1)
    x = x.unsqueeze(dim=1)  # [n,p]==>[n, 1, p]
    y = y.unsqueeze(dim=0)  # [n,p]==>[1, m, p]
    # 或者不依靠broadcast机制，使用expand 沿着1所在的维度进行复制
    # x = x.unsqueeze(dim=1).expand([n, m, p])  # [n,p]==>[n, 1, p]==>[n, m, p]
    # y = y.unsqueeze(dim=0).expand([n, m, p])  # [n,p]==>[1, m, p]==>[n, m, p]

    return torch.pow(x - y, 2).mean(dim=2)


def my_normalization1(x):  # 针对二维array
    # x = x.astype(np.float)
    # x_max = np.max(abs(x), axis=1)
    # for i in range(len(x)):
    #     x[i] /= x_max[i]
    x = maxabs_scale(x.astype(np.float), axis=1)
    return x


def my_normalization2(x):  # 针对二维array
    x = x.astype(np.float)
    x_min, x_max = np.min(x, 1), np.max(x, axis=1)
    for i in range(len(x)):
        x[i] = (x[i] - x_min[i])/(x_max[i] - x_min[i])
    return x


def my_normalization3(x):  # 针对二维array
    x = normalize(x.astype(np.float))  # default=>axis=1)
    return x


def plot_confusion_matrix(y_pre, y_true, disp_acc=True):
    """
    :param y_pre: (nc*nq, )
    :param y_true: (nc*nq, )
    :param disp_acc:
    """
    f, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pre)
    if disp_acc:  # 归一化,可显示准确率accuracy,默认显示准确率
        cm = cm.astype('float32') / (cm.sum(axis=1)[:, np.newaxis])
        sns.heatmap(cm, annot=True, ax=ax, cmap='plasma', fmt='.2f')
        # cmap如: plasma, viridis, magma, inferno; fmt: default=>'.2g'
    else:
        sns.heatmap(cm, annot=True, ax=ax, cmap='plasma')
        # cmap如: plasma, viridis, magma, inferno; fmt: default=>'.2g'
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')
    # 注意在程序末尾加 plt.show()


if __name__ == '__main__':
    '''
    import visdom
    from Data_generator_normalize import data_generate
    generator = data_generate()
    
    vis = visdom.Visdom(env='yancy_env')

    test_set, label = generator.SQ_data_generator(train=False, examples=20, normalize=True)
    data = np.squeeze(test_set, axis=-1).reshape([-1, test_set.shape[-2]])  # (n, 2048)
    label = label.reshape([-1])  # (n,)

    cls = 3
    # try vis_tSNE
    vis_tSNE(data, label, cls, vis, name='test')
    # try t_sne
    fig = t_sne(data, label, classes=cls, name='ta')
    plt.show()
    '''
    a0 = [[1, 2, 1], [1, 3, 4], [5, 8, -10]]
    a = np.array(a0)
    b = maxabs_scale(a.astype(np.float), axis=1)
    c = my_normalization1(a)
    d = my_normalization2(a)
    e = my_normalization3(a)
    print(b)
    print(c)
    print(d)
    print(e)
