import torchvision.transforms as transforms


def get_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
