from typing import *
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_graph(history, project_name: Optional[str]=None):
    train_loss = history['train_loss']
    test_loss = history['test_loss']
    
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(train_loss)), train_loss, label='train')
    plt.plot(np.arange(len(test_loss)), test_loss, label='test')
    plt.legend(loc='best')
    if project_name is not None:
        plt.savefig(f'./{project_name}/loss_graph.png')
    else:
        plt.show()


def plot_generated_images(model, project_name: Optional[str] = None):
    decoder = model.decoder

    with torch.no_grad():
        z = torch.randn(64, 2).cuda()
        sample = decoder(z).cuda()

    plt.figure(figsize=(15, 15))
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        plt.imshow(sample[i].reshape(28, 28, -1))
    
    if project_name is not None:
        plt.savefig(f'./{project_name}/generated.png')
    else:
        plt.show()