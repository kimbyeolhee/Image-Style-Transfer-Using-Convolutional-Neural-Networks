import numpy as np
from PIL import Image

import torch
from torchvision import transforms

def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

def postprocess_image(tensor: torch.Tensor):
    image = tensor.to("cpu").detach().numpy() # (1, c, h, w)
    image = image.squeeze() # (c,h,w)
    image = image.transpose(1,2,0) # (h, w, c)
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406] # denormalization
    image = image.clip(0,1)*255 # 픽셀 값을 0에서 1 사이로 제한한 후 255를 곱하여 0-255 범위로 변환
    image = image.astype(np.uint8)

    return Image.fromarray(image)


