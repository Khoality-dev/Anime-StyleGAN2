import pickle
import os
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
    print("Loading pretrained models...", end = '')
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
