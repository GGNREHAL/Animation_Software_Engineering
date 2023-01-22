import torchvision.transforms as transforms


def get_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

def add_noise(data, noise_factor):
    data += noise_factor * torch.randn(*data.shape)
    data = torch.clamp(data, 0., 1.)
    return data

def denoising_loss(data, denoised_data):
    return torch.nn.MSELoss()(data, denoised_data)