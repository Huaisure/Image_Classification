import torch
import torch.nn.functional as F
from torchvision import transforms
from model import ResNet50
from PIL import Image
import numpy as np
import cv2

# 预处理输入图像
preprocess = transforms.Compose(
    [
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 加载预训练模型
model = ResNet50()
model.load_state_dict(torch.load("model_20240529223518.pth", map_location="cpu"))
model.eval()

# 注册钩子来获取特征图和梯度
features = []
gradients = []


def save_gradient(grad):
    gradients.append(grad)


def get_features_hook(module, input, output):
    features.append(output)
    output.register_hook(save_gradient)


target_layer = model.layer4[2].conv3
target_layer.register_forward_hook(get_features_hook)

# 加载和预处理图像
n = input('输入图像序号: ')
img = Image.open(f"data/imgs/{n}.jpg")
img_tensor = preprocess(img).unsqueeze(0)

# 前向传播
output = model(img_tensor)
pred_class = output.argmax(dim=1)

# 反向传播
model.zero_grad()
output[0, pred_class].backward()

# 获取梯度和特征图
gradients = gradients[0].cpu().data.numpy()[0]
features = features[0].cpu().data.numpy()[0]

# 计算权重并生成 Grad-CAM 图
weights = np.mean(gradients, axis=(1, 2))
cam = np.zeros(features.shape[1:], dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * features[i]

cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (150, 150))
cam = cam - np.min(cam)
cam = cam / np.max(cam)

# 可视化 Grad-CAM 图
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
img = np.array(img) / 255
cam_img = heatmap + np.float32(img)
cam_img = cam_img / np.max(cam_img)

cv2.imshow("Grad-CAM", cam_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
