import pickle
import torch
import sys
from utils.QMainWindow import QMainWindow
from utils.dataset_torch_utils import *
from utils.plotlib import *
from model.models import Generator, Discriminator
from model.configs import *
from model.losses import *
from PyQt5.QtWidgets import QApplication
import threading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_models(save_path, G, D, optimizer_G, optimizer_D, visual_z, visual_noise):
    print("Saving...",end='')
    if not(os.path.exists(save_path)):
        os.makedirs(save_path)

    with open(save_path + "/Generator.pkl", "wb") as f:
        G_pkl = {
            "model": G,
            "optimizer": optimizer_G
        }
        pickle.dump(G_pkl, f)

    with open(save_path + "/Discriminator.pkl", "wb") as f:
        D_pkl = {
            "model": D,
            "optimizer": optimizer_D
        }
        pickle.dump(D_pkl, f)

    with open(save_path + "/static_noise_seed.pkl", "wb") as f:
        static_noise_seed_pkl = {
            "visual_z": visual_z,
            "visual_noise": visual_noise
        }
        pickle.dump(static_noise_seed_pkl, f)
        
    print("Done!")


def load_models(model_file_path):
    print("Loading pretrained models...")
    G = optimizer_G = D = optimizer_D = visual_z = visual_noise = None
    with open(model_file_path + "/Generator.pkl", "rb") as f:
        G_pkl = pickle.load(f)
        G = G_pkl["model"]
        optimizer_G = G_pkl["optimizer"]

    with open(model_file_path + "/Discriminator.pkl", "rb") as f:
        D_pkl = pickle.load(f)
        D = D_pkl["model"]
        optimizer_D = D_pkl["optimizer"]

    with open(model_file_path + "/static_noise_seed.pkl", "rb") as f:
        static_noise_seed_pkl = pickle.load(f)
        visual_z = static_noise_seed_pkl["visual_z"]
        visual_noise = static_noise_seed_pkl["visual_noise"]
    print("Done!")
    return G, optimizer_G, D, optimizer_D, visual_z, visual_noise

def train(qMainWindow, args):
    H, W = 256, 256

    mini_batch_size = int(BATCH_SIZE / GRAD_ACCUMULATE_FACTOR)
    dataset = Torch_Dataset("E:/anime_dataset/d1k_256x256.h5")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = mini_batch_size, shuffle = True)
    batch_iter = iter(dataloader)
    model_path = "pretrained/anime"
    if (os.path.exists(model_path)):
        G, optimizer_G, D, optimizer_D, visual_z, visual_noise = load_models(model_path)
    else:

        G = Generator(LATENT_SIZE, NUM_MAPPING_LAYER, H)
        D = Discriminator(H)
        optimizer_G = torch.optim.Adam(G.parameters(), lr = 2e-5, betas = [0, 0.9])
        optimizer_D = torch.optim.Adam(D.parameters(), lr = 2e-5, betas = [0, 0.9])
        visual_z = torch.randn(size = (mini_batch_size, LATENT_SIZE))
        visual_noise = torch.randn(size = (mini_batch_size, 1, H, W))


    G.to(device)
    D.to(device)
    scaler_G = torch.cuda.amp.GradScaler()
    scaler_D = torch.cuda.amp.GradScaler()
    visual_z = visual_z.to(device)
    visual_noise = visual_noise.to(device)

    iteration = 0
    accumulate_num = 0
    
    G.zero_grad()
    D.zero_grad()

    epoch = 0
    while (True):
        for _ in range(N_CRITICS):
            with torch.cuda.amp.autocast():
                real_samples = (next(batch_iter) / 127.5 - 1).permute(0,3,1,2).to(device)
                z = torch.randn(size = (mini_batch_size, LATENT_SIZE)).to(device)
                noise = torch.randn(size = (mini_batch_size, 1, H, W)).to(device)
                d_loss = D_loss_r1(G, D, z, noise, real_samples) / GRAD_ACCUMULATE_FACTOR
            scaler_D.scale(d_loss).backward(inputs=list(D.parameters()))

        z = torch.randn(size = (mini_batch_size, LATENT_SIZE)).to(device)
        noise = torch.randn(size = (mini_batch_size, 1, H, W)).to(device)
        with torch.cuda.amp.autocast():
            g_Loss = G_loss(G, D, z, noise) / GRAD_ACCUMULATE_FACTOR
        scaler_G.scale(g_Loss).backward(inputs=list(G.parameters()))

        accumulate_num += 1
        if (accumulate_num == GRAD_ACCUMULATE_FACTOR):
            scaler_D.step(optimizer_D)
            scaler_G.step(optimizer_G)
            D.zero_grad()
            G.zero_grad()
            scaler_D.update()
            scaler_G.update()
            accumulate_num = 0

        qMainWindow.image_lock.acquire()
        if (qMainWindow.fakes_list is None or qMainWindow.reals_list is None or qMainWindow.static_fakes_list is None):
            with torch.no_grad():
                qMainWindow.fakes_list = list((G(z, noise).permute(0,2,3,1).cpu().numpy() + 1) * 127.5)
                qMainWindow.reals_list = (real_samples.permute(0,2,3,1).cpu().numpy() + 1) * 127.5
                qMainWindow.static_fakes_list = list((G(visual_z, visual_noise).permute(0,2,3,1).cpu().numpy() + 1) * 127.5)
        qMainWindow.image_lock.release()

        if (iteration % 100 == 0):
            print("Epoch: ", epoch, "Iteration: ", iteration, "Loss G", g_Loss, "Loss D", d_loss)
            qMainWindow.updatePreviewImage()
            qMainWindow.updateDisplay()

        if qMainWindow.save_flag:
            save_models(model_path, G, D, optimizer_G, optimizer_D, visual_z, visual_noise)
            qMainWindow.save_flag = False

        if qMainWindow.exit_flag:
            save_models(model_path, G, D, optimizer_G, optimizer_D, visual_z, visual_noise)
            print("Exiting...")
            return

        iteration += 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMainWindow()
    args = 0
    t = threading.Thread(target=train, args=[window, args])
    t.start()
    return_code = app.exec_()
    t.join()
    sys.exit(return_code)