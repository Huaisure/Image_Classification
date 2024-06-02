import torch
import os
from data_preparation import get_data_loader
from model import ResNet50


def evaluate_model(model, test_loader, device):
    model.eval()

    # 分类别计算准确率，一共有6个类别
    class_correct = list(0.0 for i in range(6))
    len_class = list(0 for i in range(6))

    correct = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        c = (preds == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            len_class[label] += 1
    acc = correct.double() / len(test_loader.dataset)
    print(f"Class Accuracies:")
    for i in range(6):
        print(f"Class {i}: {class_correct[i] / len_class[i]:.4f}")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50()
    model.load_state_dict(torch.load("model_20240529223518.pth", map_location="cpu"))
    model.to(device)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_loader = get_data_loader(base_dir, "test")
    evaluate_model(model, test_loader, device)
