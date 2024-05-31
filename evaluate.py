import torch
import os
from data_preparation import get_data_loader
from model import ResNet50


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
    acc = correct.double() / len(test_loader.dataset)
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50()
    model.load_state_dict(torch.load("model_20240529223518.pth",map_location="cpu"))
    model.to(device)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_loader = get_data_loader(base_dir, "test")
    evaluate_model(model, test_loader, device)
