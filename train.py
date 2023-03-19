import torch
if __name__ == "__main__":
    x = torch.tensor([ [3,2,1],[1,2,3]])
    y = torch.zeros(2,3)
    z = x * y
    print(z)