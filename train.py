import argparse
import torch
import sys
from utils.QMainWindow import QMainWindow
from utils.dataset_torch_utils import *
from utils.plotlib import *
from model.models import Generator, Discriminator
from model.configs import *
from model.losses import *
from model.utils import save_models, load_models
from PyQt5.QtWidgets import QApplication
import threading


def train(qMainWindow, args):
    
    mini_batch_size = int(BATCH_SIZE / GRAD_ACCUMULATE_FACTOR)
    dataset = Torch_Dataset(args.data_src)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = mini_batch_size, shuffle = True)
    H, W = dataset.resolution
    batch_iter = iter(dataloader)

    if not(args.train_new) and os.path.exists(args.cp_src):
         G, optimizer_G, D, optimizer_D, visual_z = load_models(args.cp_src)
    else:
        G = Generator(LATENT_SIZE, NUM_MAPPING_LAYER, H)
        D = Discriminator(H)
        optimizer_G = torch.optim.Adam(G.parameters(), lr = LEARNING_RATE, betas = [0, 0.99], eps=1e-8)
        optimizer_D = torch.optim.Adam(D.parameters(), lr = LEARNING_RATE, betas = [0, 0.99], eps=1e-8)
        visual_z = torch.randn(size = (mini_batch_size, LATENT_SIZE))

    G.to(DEVICE)
    D.to(DEVICE)

    visual_z = visual_z.to(DEVICE)

    iteration = 0
    epoch = 0
    while (True):
        real_samples = None
        for _ in range(N_CRITICS):
            D.zero_grad()
            for _ in range(GRAD_ACCUMULATE_FACTOR):
                real_samples = next(batch_iter).to(DEVICE)
                z = torch.randn(size = (mini_batch_size, LATENT_SIZE)).to(DEVICE)
                d_loss = D_loss_r1(G, D, z, real_samples) / GRAD_ACCUMULATE_FACTOR
                d_loss.backward(inputs=list(D.parameters()))

            optimizer_D.step()
       
        G.zero_grad()
        for _ in range(GRAD_ACCUMULATE_FACTOR):
            z = torch.randn(size = (mini_batch_size, LATENT_SIZE)).to(DEVICE)
            g_Loss = G_loss(G, D, z) / GRAD_ACCUMULATE_FACTOR
            g_Loss.backward(inputs=list(G.parameters()))

        optimizer_G.step()

        qMainWindow.image_lock.acquire()
        if (qMainWindow.fakes_list is None or qMainWindow.reals_list is None or qMainWindow.static_fakes_list is None):
            with torch.no_grad():
                qMainWindow.fakes_list = list((G(z).permute(0,2,3,1).cpu().numpy() + 1) * 127.5)
                qMainWindow.reals_list = list((real_samples.permute(0,2,3,1).cpu().numpy() + 1) * 127.5)
                qMainWindow.static_fakes_list = list((G(visual_z).permute(0,2,3,1).cpu().numpy() + 1) * 127.5)
        qMainWindow.image_lock.release()

        if (iteration % args.log_iter == 0):
            print("Epoch: ", epoch, "Iteration: ", iteration, "Loss G", g_Loss, "Loss D", d_loss)
        
        if (iteration % args.cp_iter == 0):
            save_models(args.cp_src, G, D, optimizer_G, optimizer_D, visual_z)

        if qMainWindow.save_flag:
            save_models(args.cp_src, G, D, optimizer_G, optimizer_D, visual_z)
            qMainWindow.save_flag = False

        if qMainWindow.exit_flag:
            save_models(args.cp_src, G, D, optimizer_G, optimizer_D, visual_z)
            print("Exiting...")
            return

        iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--new', action = 'store_true', dest = 'train_new', default = False)
    parser.add_argument('-c', '--checkpoint-iteration', dest = 'cp_iter', type = int, default = 100)
    parser.add_argument('-cd', '--checkpoint-dir', dest = 'cp_src', type = str, default = 'pretrained/anime')
    parser.add_argument('-d', '--data-dir', dest = 'data_src', type = str, default = 'E:/anime_dataset/d1k_256x256.h5')
    parser.add_argument('-fp16', action = 'store_true', dest = 'fp16', default = False)
    parser.add_argument('-l', '--log', dest = 'log_iter', type = int, default = 1)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = QMainWindow()
    t = threading.Thread(target=train, args=[window, args])
    t.start()
    return_code = app.exec_()
    t.join()
    sys.exit(return_code)