from src.models.simple_cnn import SimpleCNN
from src.data import get_dataloaders
import torch

def main():
    model = SimpleCNN()
    model.eval()
    train, val, test = get_dataloaders("data/")
    images, labels = next(iter(train))

    with torch.no_grad():
        outputs = model(images)
    print(outputs.shape)
    
    count=0
    for i in model.parameters():
        count += i.numel()

    print(outputs[0])
    print(outputs[1])
    print(outputs[2])
    print(count)
if __name__ == "__main__":
    main()