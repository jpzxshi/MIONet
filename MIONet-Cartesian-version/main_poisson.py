"""
@author: jpzxshi
"""
import numpy as np
import matplotlib.pyplot as plt
import learner as ln
from data_poisson import Poisson_2d_data

def postprocessing_Poisson_2d(data, net):
    k = data.X_test[0][0]
    f = data.X_test[1][0]
    x = data.X_test[2]
    u = data.y_test[0]
    
    u_pred = net.predict((k, f, x), returnnp=True).reshape(100, 100)
    k = data.tc_to_np(k).reshape(100, 100)
    f = data.tc_to_np(f).reshape(100, 100)
    u = data.tc_to_np(u).reshape(100, 100)
    
    plt.figure(figsize=[6.4 * 2, 4.8 * 2])
    plt.subplot(221)
    plt.imshow(np.rot90(k), cmap='rainbow')
    plt.xticks([0, 49.5, 99], [0, 0.5, 1])
    plt.yticks([0, 49.5, 99], [1, 0.5, 0])
    plt.colorbar()
    plt.title('k')
    
    plt.subplot(222)
    plt.imshow(np.rot90(f), cmap='rainbow')
    plt.xticks([0, 49.5, 99], [0, 0.5, 1])
    plt.yticks([0, 49.5, 99], [1, 0.5, 0])
    plt.colorbar()
    plt.title('f')
    
    plt.subplot(223)
    plt.imshow(np.rot90(u), cmap='rainbow')
    plt.xticks([0, 49.5, 99], [0, 0.5, 1])
    plt.yticks([0, 49.5, 99], [1, 0.5, 0])
    plt.colorbar()
    plt.title('u')
    
    plt.subplot(224)
    plt.imshow(np.rot90(u_pred), cmap='rainbow')
    plt.xticks([0, 49.5, 99], [0, 0.5, 1])
    plt.yticks([0, 49.5, 99], [1, 0.5, 0])
    plt.colorbar()
    plt.title('MIONet')
    
    plt.savefig('Poisson_2d_MIONet_prediction.pdf')

def main():
    device = 'gpu' # 'cpu' or 'gpu'
    #### data
    path = './data_poisson/'
    ##### MIONet
    sizes = [
        [100 ** 2] + [500] * 3, # Branch net for $k$
        [100 ** 2, -500], # Branch net for $f$, a linear mapping, and -500 means this layer is without bias
        # Because the solution operator of Poisson equation is linear w.r.t. the right-hand side $f$
        [2] + [500] * 3, # Trunk net
        ]
    activation = 'relu'
    initializer = 'default'
    ##### training
    lr = 1e-5
    iterations = 10000
    batch_size = 10
    print_every = 1000
    
    training_args = {
        'criterion': 'MSE',
        'optimizer': 'Adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': batch_size,
        'print_every': print_every,
        'save': 'best_only',
        'callback': None,
        'dtype': 'float',
        'device': device,
    }
    
    ln.Brain.Start()
    data = Poisson_2d_data(path)
    net = ln.nn.MIONet_Cartesian(sizes, activation, initializer, bias=False)
    ln.Brain.Init(data, net)
    ln.Brain.Run(**training_args)
    ln.Brain.Restore()
    ln.Brain.Output(data=False)
    postprocessing_Poisson_2d(data, ln.Brain.Best_model())
    ln.Brain.End()

if __name__ == '__main__':
    main()