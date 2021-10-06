import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def fitting(simple_points, view=0):
    """曲线拟合
    """
    # 预拟合
    ff_pre = fit_pre(simple_points, view=view)
    # 滤波
    x_, y_ = simple_filter(simple_points, ff_pre, view=0)
    # 局部拟合
    ff_ = RandomForest(x_, y_, view=view)
    # ff_ = Polynomial_fit(x_, y_, deg=15)

    if view:
        x_view = np.linspace(min(x_), max(x_), 100)
        y_pred = ff_(x_view)
        plt.plot(x_, y_, 'o', markersize=3)
        plt.plot(x_view, y_pred, '-r')
        # plt.text(-15, 205, 'time=' + str(ctime), fontdict={'size': 15, 'color': 'red'})
        plt.show()
    return ff_


def fit_pre(simple_points, view=0):
    x_ = simple_points[:, 0]
    y_ = simple_points[:, 1]
    predict_y = sklearn_fit(x_, y_, degree=3)
    ff_ = Polynomial_fit(x_, predict_y, deg=5)

    if view:
        plt.scatter(x_, y_, s=5, color='blue', alpha=0.8)
        # plt.plot(x_, predict_y, linewidth=2, label='line')
        # x_view = np.linspace(min(x_), max(x_), simple_points.shape[0])
        # plt.plot(x_view, ff_(x_view), linewidth=2)
        # plt.legend()
        # plt.show()
    return ff_


def rmse(predictions, targets):
    # 均方根差
    return np.sqrt(((predictions - targets) ** 2).mean())


def simple_filter(simple_points, ff_pre, view=0):
    # 利用均方根差与正态分布特性（满足 “68-95-99.7”规则：
    # 大约 68% 的值落在1σ内，95% 的值落在2σ内，99.7% 的值落在3σ内）
    k = rmse(simple_points[:, 1], ff_pre(simple_points[:, 0]))  # 代码运行时长：0.000069s
    # k = np.sqrt(metrics.mean_squared_error(simple_points[:, 1], ff_pre(simple_points[:, 0])))#代码运行时长：0.000198s
    k = 3 * k
    # 计算自适应k

    index = []
    for i in range(np.shape(simple_points)[0]):
        if abs(simple_points[i, 1] - ff_pre(simple_points[i, 0])) < k:
            index.append(i)
    new_point = np.array([simple_points[i] for i in index])

    if view:
        x_ = simple_points[:, 0]
        y_ = simple_points[:, 1]
        x_view = np.linspace(min(x_), max(x_), 100)
        plt.scatter(x_, y_, s=5, color='red', alpha=0.8)
        plt.plot(x_view, ff_pre(x_view), linewidth=2)
        plt.plot(x_view, ff_pre(x_view) + k, '-g')
        plt.plot(x_view, ff_pre(x_view) - k, '-g')
        plt.show()
    return new_point[:, 0], new_point[:, 1]


def Polynomial_fit(x_, y_, deg=16):
    # 多项式拟合
    z1 = np.polyfit(x_, y_, deg)  # 曲线拟合，返回值为多项式的各项系数
    p1 = np.poly1d(z1)  # 返回值为多项式的表达式，也就是函数式子
    return p1


def least_square_method(simple_points, deg=7):
    # 最小二乘法拟合
    x = simple_points[:, 0]
    y = simple_points[:, 1]

    m = []
    for i in range(deg + 1):  # 这里选的最高次为x^7的多项式
        a = x ** (deg - i)
        m.append(a)
    X = np.array(m).T
    C, resid, rank, s = np.linalg.lstsq(X, y, rcond=None)
    p1 = np.poly1d(C)  # 返回值为多项式的表达式，也就是函数式子
    return p1


def RandomForest(x_, y_, step=800, view=0):
    x_fit = np.linspace(min(x_), max(x_), step)

    regr = RandomForestRegressor()
    regr.fit(x_[:, None], y_)
    y_fit = regr.predict(x_fit[:, None])
    # 二次拟合
    ff = Polynomial_fit(x_fit, y_fit, 16)

    if view:
        # plt.plot(x_, y_, '*', markersize=3)
        plt.plot(x_fit, y_fit, '-b')
        # plt.plot(x_fit, ff(x_fit), '-r')
        # plt.show()
    return ff


'''def fourier(x, a0, a1, n1, b1):
    ret = a1 * np.cos(n1 * np.pi * x + b1)
    return ret


def fourier_fit(x_, y_):
    # Fit with 15 harmonics
    popt, pcov = curve_fit(fourier, x_, y_)
    print(popt)
    # Plot data, 15 harmonics, and first 3 harmonics
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    _ = plt.plot(x_, y_, '*', markersize=3)
    _ = plt.plot(x_, fourier(x_, *popt), '-r')
    # _ = plt.plot(x_, fourier(x_, popt[0], popt[1], popt[2]))
    plt.show()
'''


# def dl_fit(simple_points):
#     """
#         Pytorch是一个拥有强力GPU加速的张量和动态构建网络的库，其主要构建是张量，所以可以把PyTorch当做Numpy
#         来用,Pytorch的很多操作好比Numpy都是类似的，但是其能够在GPU上运行，所以有着比Numpy快很多倍的速度。
#         训练完了，发现隐层越大，拟合的速度越是快，拟合的效果越是好
#     """
#
#     print('------      构建数据集      ------')
#     flag = 0
#     if flag:
#         x = simple_points[:, 0].reshape(simple_points.shape[0], 1).astype(np.float32)
#         # x = x.reshape[x.shape,1].reshape(simple_points.shape[0], 1)
#         y = simple_points[:, 1].reshape(simple_points.shape[0], 1).astype(np.float32)
#     else:
#         x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
#         # torch.rand返回的是[0,1]之间的均匀分布   这里是使用一个计算式子来构造出一个关联结果，当然后期要学的也就是这个式子
#         y = x.pow(2) + 0.2 * torch.rand(x.size())
#         # Variable是将tensor封装了下，用于自动求导使用
#         x, y = Variable(x), Variable(y)
#
#     x = torch.tensor(x)
#     y = torch.tensor(y)
#     # Variable是将tensor封装了下，用于自动求导使用
#     x, y = Variable(x), Variable(y)
#     # 绘图展示
#     # plt.scatter(x.data.numpy(), y.data.numpy())
#     # plt.show()
#
#     print('------      搭建网络      ------')
#
#     # 使用固定的方式继承并重写 init和forword两个类
#     class Net(torch.nn.Module):
#         def __init__(self, n_feature, n_hidden, n_hidden1, n_output):
#             # 初始网络的内部结构
#             super(Net, self).__init__()
#             self.hidden = torch.nn.Linear(n_feature, n_hidden)
#             self.hidden1 = torch.nn.Linear(n_hidden, n_hidden1)
#             self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden1)
#             self.predict = torch.nn.Linear(n_hidden1, n_output)
#
#         def forward(self, x_):
#             # 一次正向行走过程
#             x_ = fun.relu(self.hidden(x_))
#             x_ = fun.relu(self.hidden1(x_))
#             x_ = fun.relu(self.hidden2(x_))
#             # x_ = fun.tanh(self.hidden2(x_))
#             x_ = self.predict(x_)
#             return x_
#
#     net = Net(n_feature=1, n_hidden=1000, n_hidden1=1000, n_output=1)
#     print('网络结构为：', net)
#
#     print('------      启动训练      ------')
#     loss_func = fun.mse_loss
#     optimizer = torch.optim.SGD(net.parameters(), lr=0.0005)
#
#     # 使用数据 进行正向训练，并对Variable变量进行反向梯度传播  启动100次训练
#     for t in range(1000):
#         # 使用全量数据 进行正向行走
#         prediction = net(x)
#         loss = loss_func(prediction, y)
#         optimizer.zero_grad()  # 清除上一梯度
#         loss.backward()  # 反向传播计算梯度
#         optimizer.step()  # 应用梯度
#
#         # 间隔一段，对训练过程进行可视化展示
#         if t % 5 == 0:
#             plt.cla()
#             plt.scatter(x.data.numpy(), y.data.numpy())  # 绘制真实曲线
#             plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
#             plt.text(-1, 0, 'Loss=' + str(loss.data), fontdict={'size': 20, 'color': 'red'})
#             plt.text(-1, 5, 'num=' + str(t), fontdict={'size': 20, 'color': 'red'})
#             plt.pause(0.1)
#     plt.ioff()
#     plt.show()
#     print('------      预测和可视化      ------')
def sklearn_fit(x_, y_, degree=4):
    clf = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                    ('linear', linear_model.LinearRegression(fit_intercept=False))])
    clf.fit(x_[:, np.newaxis], y_)  # 自变量需要二维数组
    predict_y = clf.predict(x_[:, np.newaxis])
    return predict_y


def LWLR():
    # 局部加权线性回归（Locally Weighted Linear Regression，LWLR）
    # 比较不同k值得回归效果
    pass


if __name__ == "__main__":
    xy_ = np.loadtxt('simple_points.txt')
    xArr = xy_[:, 0]
    yArr = xy_[:, 1]
    # ff = fit_filter_pre(xy_, view=1)
    fitting(xy_, view=0)
