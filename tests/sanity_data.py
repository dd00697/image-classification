from src.data import get_dataloaders, get_cifar10_classes
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid 

def main():
    train, val, test = get_dataloaders("data/", num_workers=0)

    print(f"Training samples: {len(train.dataset)}")
    print(f"Validation samples: {len(val.dataset)}")
    print(f"Test samples: {len(test.dataset)}")

    image, labels = next(iter(train))
    print(image.shape)
    print(labels.shape)

    print(image.min())
    print(image.max())

    first_16 = image[:16]
    mean: tuple[float, float, float] = (0.4919, 0.4822, 0.4465)
    std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

    mean_tensor = torch.tensor(mean).view(3,1,1)
    std_tensor = torch.tensor(std).view(3,1,1)

    first_16 = first_16 * std_tensor + mean_tensor 

    first_16 = first_16.clamp(0,1)

    grid = make_grid(first_16,nrow=4, padding=2).permute(1,2,0).numpy()

    plt.figure(figsize=(8,8))
    plt.imshow(grid)
    plt.title("CIFAR-10 Training Samples")
    plt.axis("off")
    plt.savefig("outputs/sample_grid.png")
    plt.close()

    class_names = get_cifar10_classes()
    labels_16 = labels[:16]
    label_names = [class_names[i] for i in labels_16]
    print(label_names)
if __name__ == '__main__':
    main()