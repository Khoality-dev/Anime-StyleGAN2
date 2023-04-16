import argparse
import torch
import sys
from IPython import get_ipython

from utils.dataset_torch_utils import *
from utils.plotlib import *
from model.models import Generator, Discriminator
from model.configs import *
from model.losses import *
from model.utils import G_large_batch, save_models, load_models
from utils.mainwindow import MainWindow

# if PyQt5 is available, enable interactive mode
try:
    from utils.mainwindow import QMainWindow
    from PyQt5.QtWidgets import QApplication
except ImportError:
    ...

import threading
from collections import deque

def train(mainWindow, args):
    mini_batch_size = int(BATCH_SIZE / GRAD_ACCUMULATE_FACTOR)
    dataset = Torch_Dataset(args.data_src)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = mini_batch_size, shuffle = True)
    H, W = dataset.resolution
    batch_iter = iter(dataloader)

    if not(args.train_new) and os.path.exists(args.cp_src):
         G, optimizer_G, D, optimizer_D, visual_z = load_models(args.cp_src)
    else:
        random_seed = 1.048596
        torch.manual_seed(random_seed)
        G = Generator(LATENT_SIZE, NUM_MAPPING_LAYER, H)
        D = Discriminator(H)
        optimizer_G = torch.optim.Adam(G.parameters(), lr = LEARNING_RATE, betas = [0, 0.99])
        optimizer_D = torch.optim.Adam(D.parameters(), lr = LEARNING_RATE, betas = [0, 0.99])
        visual_z = torch.randn(size = (VISUALIZATION_BATCH_SIZE, LATENT_SIZE))

    G.to(DEVICE)
    D.to(DEVICE)
    while (True):
        real_samples = None
        for _ in range(N_CRITICS):
            D.zero_grad()
            for _ in range(GRAD_ACCUMULATE_FACTOR):
                real_samples = next(batch_iter).to(DEVICE)
                z = torch.randn(size = (mini_batch_size, LATENT_SIZE)).to(DEVICE)

                if (G.iteration % LAZY_REG_FACTOR == 0):
                    d_loss = D_loss_r1(G, D, z, real_samples, regularization=True)
                else:
                    d_loss = D_loss_r1(G, D, z, real_samples, regularization=False)

                d_loss = d_loss / GRAD_ACCUMULATE_FACTOR
                d_loss.backward()

            optimizer_D.step()
       
        G.zero_grad()
        for _ in range(GRAD_ACCUMULATE_FACTOR):
            z = torch.randn(size = (mini_batch_size, LATENT_SIZE)).to(DEVICE)

            g_Loss = None
            if (G.iteration % LAZY_REG_FACTOR == 0):
                g_Loss = G_loss_pl(G, D, z, regularization=True)
            else:
                g_Loss = G_loss_pl(G, D, z, regularization=False)

            g_Loss = g_Loss / GRAD_ACCUMULATE_FACTOR
            g_Loss.backward()

        optimizer_G.step()

        if (G.iteration % args.log_iter == 0):
            print("Iteration: ", G.iteration, "Loss G", g_Loss, "Loss D", d_loss)
        
        if (G.iteration % 100 == 0):
            mainWindow.update = True

        if (mainWindow.update_flag):
            with torch.no_grad():
                zs = torch.randn(size = (VISUALIZATION_BATCH_SIZE, LATENT_SIZE))
                fakes_list = list((G_large_batch(G, zs, mini_batch_size, device = 'cpu').permute(0,2,3,1).cpu().numpy() + 1) * 127.5)
                reals_list = list((real_samples.permute(0,2,3,1).cpu().numpy() + 1) * 127.5)
                static_fakes_list = list((G_large_batch(G, visual_z, mini_batch_size, device='cpu').permute(0,2,3,1).cpu().numpy() + 1) * 127.5)
            mainWindow.updatePreviewImage(fakes_list, reals_list, static_fakes_list)
            mainWindow.updateDisplay()
            mainWindow.update_flag = False

        if (G.iteration % args.cp_iter == 0):
            save_models(args.cp_src, G, D, optimizer_G, optimizer_D, visual_z)

        if mainWindow.save_flag:
            save_models(args.cp_src, G, D, optimizer_G, optimizer_D, visual_z)
            mainWindow.save_flag = False

        if mainWindow.exit_flag:
            print("Exiting...")
            return

        G.iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--new', action = 'store_true', dest = 'train_new', default = False)
    parser.add_argument('-i', '--interactive-mode', action = 'store_true', dest = 'interactive_mode', default = True)
    parser.add_argument('-c', '--checkpoint-iteration', dest = 'cp_iter', type = int, default = 100)
    parser.add_argument('-cd', '--checkpoint-dir', dest = 'cp_src', type = str, default = 'pretrained/anime')
    parser.add_argument('-d', '--data-dir', dest = 'data_src', type = str, default = '/media/khoa/LHC/anime_dataset/d1k_256x256.h5')
    parser.add_argument('-l', '--log', dest = 'log_iter', type = int, default = 10)
    args = parser.parse_args()

    INTERACTIVE_MODE = not('google.colab' in str(get_ipython()))
    #if not interactive, save preview images
    if args.interactive_mode and INTERACTIVE_MODE:
        app = QApplication(sys.argv)
        mainWindow = QMainWindow()
        t = threading.Thread(target=train, args=[mainWindow, args])
        t.start()
        return_code = app.exec_()
        t.join()
        sys.exit(return_code)
    else:
        mainWindow = MainWindow()
        train(mainWindow, args)