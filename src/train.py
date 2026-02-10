import time 

import torch 
import torch.nn as nn 
import torch.optim as optim 
from src.data import get_dataloaders
from src.models.simple_cnn import SimpleCNN


def train_one_epoch(model: nn.Module, loader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device | str) -> tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0.0, 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size 

        #Accuracy
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size
    average_loss = running_loss / total 
    
    accuracy = correct / total 

    return average_loss, accuracy
        
@torch.no_grad()
def validate(model: nn.Module, loader, criterion: nn.Module, device: torch.device | str) -> tuple[float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0.0, 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        #Accuracy
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size 
    average_loss = running_loss / total 
    accuracy = correct / total 

    return average_loss, accuracy

def main():
    
    epochs: int = 10
    learning_rate: float = 0.01
    momentum: float = 0.9

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    train, val, test = get_dataloaders()
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    start_time = time.perf_counter()

    for epoch in range(0, epochs):
        train_loss, train_accuracy = train_one_epoch(model, train, criterion, optimizer, device)
        validation_loss, validaiton_accuracy = validate(model, val, criterion, device)
        print(f"Epoch: {epoch+1}/10 | Train Loss: {train_loss} | Train Accuracy: {train_accuracy} | Validation Loss: {validation_loss} | Validation Accuracy: {validaiton_accuracy}")
    end_time = time.perf_counter()
    total_time = end_time - start_time
    mins = int(total_time // 60)
    secs = total_time % 60
    print(f"Total training time: {mins}m {secs:.2f}s")

if __name__ == "__main__":
    main()