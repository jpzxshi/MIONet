"""
@author: jpzxshi
"""
import os
import numpy as np
from sklearn import gaussian_process as gp
from itertools import product
import matplotlib.pyplot as plt
import learner as ln
#from pathos.pools import ProcessPool

class Gaussian_process:
    '''Generate Gaussian process.
    '''
    def __init__(self, intervals, mean, std, length_scale, features, e=1e-13):
        self.intervals = intervals # e.g. [0, 1]
        self.mean = mean # e.g. 0
        self.std = std # e.g. 1
        self.length_scale = length_scale # e.g. 0.3
        self.features = features # e.g. 1000
        self.e = e

    @ln.utils.timing
    def generate(self, num):
        if isinstance(self.intervals[0], list):
            itvs = []
            for interval in self.intervals:
                itvs.append(np.linspace(interval[0], interval[1], num=self.features))
            x = np.array(list(product(*itvs)))
            d = len(self.intervals)
        else:
            x = np.linspace(self.intervals[0], self.intervals[1], num=self.features)[:, None]
            d = 1
        A = gp.kernels.RBF(length_scale=self.length_scale)(x)
        L = np.linalg.cholesky(A + self.e * np.eye(x.shape[0]))
        res = (L @ np.random.randn(x.shape[0], num)).transpose() * self.std + self.mean # [num, features ** d]
        return res.reshape([num] + [self.features] * d)

def solve_Poisson_2d(k, f):
    h = 1 / (k.shape[0] - 1)
    n = k.shape[0] - 2
    A0 = ((4 / 3) * k[1:-1, 1:-1]
          + (1 / 3) * (k[:-2, 2:] + k[2:, :-2])
          + (1 / 2) * (k[1:-1, 2:] + k[1:-1, :-2] + k[2:, 1:-1] + k[:-2, 1:-1])).ravel() * np.eye(n ** 2)
    A1 = ((- 1 / 3) * (k[1:-1, 1:-2] + k[1:-1, 2:-1]) 
          + (- 1 / 6) * (k[:-2, 2:-1] + k[2:, 1:-2]))
    A1 = np.hstack([A1, np.zeros([n, 1])]).ravel()[:, None] * np.eye(n ** 2, k=1)
    A2 = ((- 1 / 3) * (k[1:-2, 1:-1] + k[2:-1, 1:-1])
          + (- 1 / 6) * (k[1:-2, 2:] + k[2:-1, :-2]))
    A2 = np.vstack([A2, np.zeros([1, n])]).ravel()[:, None] * np.eye(n ** 2, k=n)
    A = A0 + A1 + A2 + A1.T + A2.T
    b = h ** 2 * ((1 / 2) * f[1:-1, 1:-1]
         + (1 / 12) * (f[2:, 1:-1] + f[1:-1, 2:] + f[:-2, 2:] + f[:-2, 1:-1] + f[1:-1, :-2] + f[2:, :-2])).ravel()
    u = np.linalg.solve(A, b).reshape(n, n)
    return np.hstack([np.zeros([n + 2, 1]), np.vstack([np.zeros([1, n]), u, np.zeros([1, n])]), np.zeros([n + 2, 1])])
    
def generate_data_from_kf(gps_k, gps_f):
    sensors_per_dim = gps_k.shape[-1]
    def generate(k, f, i):
        print('Solving No. {} ...'.format(i), flush=True)
        u = solve_Poisson_2d(k, f)
        return np.hstack([k.ravel(), f.ravel(), u.ravel()])
    
    #### multi-thread
    #p = ProcessPool(nodes=4)
    #res = np.vstack(list(p.map(generate, gps_k, gps_f)))
    #### single thread
    res = np.vstack(list(map(generate, gps_k, gps_f, np.arange(gps_k.shape[0]))))
    
    x1 = np.linspace(0, 1, num=sensors_per_dim)
    x2 = np.linspace(0, 1, num=sensors_per_dim)
    x = np.array(list(product(x1, x2)))
    data_k = res[..., :sensors_per_dim ** 2]
    data_f = res[..., sensors_per_dim ** 2:sensors_per_dim ** 2 * 2]
    data_u = res[..., -sensors_per_dim ** 2:]
    return (data_k, data_f, x), data_u

def generate_and_save_data(train_num, test_num, path='./data/'):
    if not os.path.isdir(path): os.makedirs(path)

    gp_k = Gaussian_process([[0, 1]] * 2, 1, 0.2, 0.2, 100)
    gp_f = Gaussian_process([[0, 1]] * 2, 0, 1, 0.2, 100)
    
    gps_train_k, gps_train_f = gp_k.generate(train_num), gp_f.generate(train_num)
    gps_test_k, gps_test_f = gp_k.generate(test_num), gp_f.generate(test_num)

    print('Generating training data ...', flush=True)
    X_train, y_train = generate_data_from_kf(gps_train_k, gps_train_f)
    print('Generating test data ...', flush=True)
    X_test, y_test = generate_data_from_kf(gps_test_k, gps_test_f)
    ####
    np.savez_compressed(path + 'X_train', *X_train)
    np.save(path + 'y_train', y_train)
    np.savez_compressed(path + 'X_test', *X_test)
    np.save(path + 'y_test', y_test)
    
def test_data(path):
    X_train = np.load(path + 'X_train.npz')
    y_train = np.load(path + 'y_train.npy')
    X_test = np.load(path + 'X_test.npz')
    y_test = np.load(path + 'y_test.npy')
    
    print(X_train['arr_0'].shape, X_train['arr_1'].shape, X_train['arr_2'].shape)
    print(y_train.shape)
    print(X_test['arr_0'].shape, X_test['arr_1'].shape, X_test['arr_2'].shape)
    print(y_test.shape)
    
    n = np.random.choice(X_test['arr_0'].shape[0])
    
    k = X_test['arr_0'][n].reshape(100, 100)
    f = X_test['arr_1'][n].reshape(100, 100)
    u_data = y_test[n].reshape(100, 100)
    u_solve = solve_Poisson_2d(k, f)
    
    print('{}-th point from test data'.format(n))
    print('max error: ', np.max(np.abs(u_data - u_solve)))
    
    plt.figure(figsize=[6.4 * 2, 4.8 * 1])
    plt.subplot(121)
    plt.imshow(np.rot90(u_data), cmap='rainbow')
    plt.xticks([0, 49.5, 99], [0, 0.5, 1])
    plt.yticks([0, 49.5, 99], [1, 0.5, 0])
    plt.colorbar()
    plt.title('u (data)')
    plt.subplot(122)
    plt.imshow(np.rot90(u_solve), cmap='rainbow')
    plt.xticks([0, 49.5, 99], [0, 0.5, 1])
    plt.yticks([0, 49.5, 99], [1, 0.5, 0])
    plt.colorbar()
    plt.title('u (solve)')
    
class Poisson_2d_data(ln.data.Data_MIONet_Cartesian):
    '''Data for 2d Poisson equation.
    '''
    def __init__(self, path):
        super(Poisson_2d_data, self).__init__()
        X_train, X_test = np.load(path + '/X_train.npz'), np.load(path + '/X_test.npz')
        self.X_train = (X_train['arr_0'], X_train['arr_1'], X_train['arr_2'])
        self.y_train = np.load(path + '/y_train.npy')
        self.X_test = (X_test['arr_0'], X_test['arr_1'], X_test['arr_2'])
        self.y_test = np.load(path + '/y_test.npy')

def main():
    path = './data_poisson/'
    train_num = 50 # 5000
    test_num = 10 # 500
    generate_and_save_data(train_num, test_num, path)
    test_data(path)

if __name__ == '__main__':
    main()