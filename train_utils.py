import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_losses(inputs, outputs, losses, val_losses, title, nfe=None, net=None):
    # plot statistics
    if nfe is not None:
        nfe[0].append(net.odeblock_down1.odefunc.nfe)
        nfe[1].append(net.odeblock_down2.odefunc.nfe)
        nfe[2].append(net.odeblock_down3.odefunc.nfe)
        nfe[3].append(net.odeblock_down4.odefunc.nfe)
        nfe[4].append(net.odeblock_embedding.odefunc.nfe)
        nfe[5].append(net.odeblock_up1.odefunc.nfe)
        nfe[6].append(net.odeblock_up2.odefunc.nfe)
        nfe[7].append(net.odeblock_up3.odefunc.nfe)
        nfe[8].append(net.odeblock_up4.odefunc.nfe)

    if nfe is not None: cols = 4
    else: cols = 3
    fig, ax = plt.subplots(nrows=1, ncols=cols, figsize=(15, 5))

    fig.suptitle(title, fontsize=16)

    ax[0].plot(np.arange(len(losses)), losses, label="loss")
    ax[0].plot(np.arange(len(val_losses)), val_losses, label="val_loss")

    if nfe is not None:
        ax[3].plot(np.arange(len(nfe[0])), nfe[0], label="down1")
        ax[3].plot(np.arange(len(nfe[0])), nfe[1], label="down2")
        ax[3].plot(np.arange(len(nfe[0])), nfe[2], label="down3")
        ax[3].plot(np.arange(len(nfe[0])), nfe[3], label="down4")
        ax[3].plot(np.arange(len(nfe[0])), nfe[4], label="embed")
        ax[3].plot(np.arange(len(nfe[0])), nfe[5], label="up1")
        ax[3].plot(np.arange(len(nfe[0])), nfe[6], label="up2")
        ax[3].plot(np.arange(len(nfe[0])), nfe[7], label="up3")
        ax[3].plot(np.arange(len(nfe[0])), nfe[8], label="up4")
        ax[3].legend()

    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)[0]
    outputs = outputs.detach().cpu()
    outputs = outputs.numpy()

    ax[0].legend()
    ax[1].imshow(outputs)
    ax[2].imshow(inputs.detach().cpu()[0].numpy().transpose(1, 2, 0))

    plt.show()
