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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mini_batch_size = int(BATCH_SIZE / GRAD_ACCUMULATE_FACTOR)
    dataset = Torch_Dataset(args.data_src)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = mini_batch_size, shuffle = True)
    H, W = dataset.resolution
    batch_iter = iter(dataloader)

    if not(args.train_new) and os.path.exists(args.cp_src):
         G, optimizer_G, D, optimizer_D, visual_z, visual_noise = load_models(args.cp_src)
    else:
        G = Generator(LATENT_SIZE, NUM_MAPPING_LAYER, H)
        D = Discriminator(H)
        optimizer_M = torch.optim.Adam(G.mapping_network.parameters(), lr = MAPPING_NETWORK_LEARNING_RATE *0.1, betas = [0, 0.9], eps=1e-8)
        optimizer_S = torch.optim.Adam(G.synthesis.parameters(), lr = SYNTHESIS_LEARNING_RATE, betas = [0, 0.9], eps=1e-8)
        optimizer_D = torch.optim.Adam(D.parameters(), lr = SYNTHESIS_LEARNING_RATE, betas = [0, 0.9], eps=1e-8)
        visual_z = torch.randn(size = (mini_batch_size, LATENT_SIZE))
        visual_noise = torch.randn(size = (mini_batch_size, 1, H, W))

    G.to(device)
    D.to(device)

    scaler_G = None 
    scaler_D = None
    if args.fp16:
        scaler_D = torch.cuda.amp.GradScaler()
        scaler_G = torch.cuda.amp.GradScaler()
    visual_z = visual_z.to(device)
    visual_noise = visual_noise.to(device)

    iteration = 0
    lazy_factor = 0
    epoch = 0
    while (True):
        real_samples = None
        for _ in range(N_CRITICS):
            D.zero_grad()
            for _ in range(GRAD_ACCUMULATE_FACTOR):
                
                if (scaler_D is not None):
                    with torch.cuda.amp.autocast():
                        real_samples = next(batch_iter).to(device)
                        z = torch.randn(size = (mini_batch_size, LATENT_SIZE)).to(device)
                        noise = torch.randn(size = (mini_batch_size, 1, H, W)).to(device)
                        d_loss = D_loss_r1(G, D, z, noise, real_samples) / GRAD_ACCUMULATE_FACTOR
                    scaler_D.scale(d_loss).backward(inputs=list(D.parameters()))
                else:
                    real_samples = next(batch_iter).to(device)
                    z = torch.randn(size = (mini_batch_size, LATENT_SIZE)).to(device)
                    noise = torch.randn(size = (mini_batch_size, 1, H, W)).to(device)
                    d_loss = D_loss_r1(G, D, z, noise, real_samples) / GRAD_ACCUMULATE_FACTOR
                    d_loss.backward(inputs=list(D.parameters()))

            if (scaler_D is not None):
                scaler_D.step(optimizer_D)
                scaler_D.update()
            else:
                optimizer_D.step()
            D.zero_grad()
       
        G.zero_grad()
        for _ in range(GRAD_ACCUMULATE_FACTOR):

            if (scaler_G is not None):
                with torch.cuda.amp.autocast():
                    z = torch.randn(size = (mini_batch_size, LATENT_SIZE)).to(device)
                    noise = torch.randn(size = (mini_batch_size, 1, H, W)).to(device)
                    g_Loss = G_loss(G, D, z, noise) / GRAD_ACCUMULATE_FACTOR
                scaler_G.scale(g_Loss).backward(inputs=list(G.parameters()))
            else:
                z = torch.randn(size = (mini_batch_size, LATENT_SIZE)).to(device)
                noise = torch.randn(size = (mini_batch_size, 1, H, W)).to(device)
                g_Loss = G_loss(G, D, z, noise) / GRAD_ACCUMULATE_FACTOR
                g_Loss.backward(inputs=list(G.parameters()))
            
        if (scaler_G is not None):
            scaler_G.step(optimizer_G)
            scaler_G.update()
        else:
            optimizer_M.step()
            optimizer_S.step()
        G.zero_grad()
        
        qMainWindow.image_lock.acquire()
        if (qMainWindow.fakes_list is None or qMainWindow.reals_list is None or qMainWindow.static_fakes_list is None):
            with torch.no_grad():
                qMainWindow.fakes_list = list((G(z, noise).permute(0,2,3,1).cpu().numpy() + 1) * 127.5)
                qMainWindow.reals_list = list((real_samples.permute(0,2,3,1).cpu().numpy() + 1) * 127.5)
                qMainWindow.static_fakes_list = list((G(visual_z, visual_noise).permute(0,2,3,1).cpu().numpy() + 1) * 127.5)
        qMainWindow.image_lock.release()

        if (iteration % args.log_iter == 0):
            print("Epoch: ", epoch, "Iteration: ", iteration, "Loss G", g_Loss, "Loss D", d_loss)
            qMainWindow.updatePreviewImage()
            qMainWindow.updateDisplay()
        
        if (iteration % args.cp_iter == 0):
            pass
            #save_models(args.cp_src, G, D, optimizer_G, optimizer_D, visual_z, visual_noise)

        if qMainWindow.save_flag:
            #save_models(args.cp_src, G, D, optimizer_G, optimizer_D, visual_z, visual_noise)
            qMainWindow.save_flag = False

        if qMainWindow.exit_flag:
            save_models(args.cp_src, G, D, optimizer_G, optimizer_D, visual_z, visual_noise)
            print("Exiting...")
            return

        iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--new', action = 'store_true', dest = 'train_new', default = True)
    parser.add_argument('-c', '--checkpoint-iteration', dest = 'cp_iter', type = int, default = 100)
    parser.add_argument('-cd', '--checkpoint-dir', dest = 'cp_src', type = str, default = 'pretrained/anime')
    parser.add_argument('-d', '--data-dir', dest = 'data_src', type = str, default = 'E:/anime_dataset/d1k_256x256.h5')
    parser.add_argument('-fp16', action = 'store_true', dest = 'fp16', default = False)
    parser.add_argument('-l', '--log', dest = 'log_iter', type = int, default = 10)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = QMainWindow()
    t = threading.Thread(target=train, args=[window, args])
    t.start()
    return_code = app.exec_()
    t.join()
    sys.exit(return_code)