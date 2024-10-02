from PIL import Image

from .data_utils import postprocess_image, preprocess_image


class DataModule:
    def __init__(self, content_image_path, style_image_path):
        self.content_image_path = content_image_path
        self.style_image_path = style_image_path

    def _load_preprocessed_image(self, image_path):
        image = Image.open(image_path)
        image_tensor = preprocess_image(image)

        return image_tensor

    def get_image_tensors(self):
        content_image_tensor = self._load_preprocessed_image(self.content_image_path)
        style_image_tensor = self._load_preprocessed_image(self.style_image_path)

        return content_image_tensor, style_image_tensor

    def get_noise_image_tensor_to_image(self, noise_image_tensor):
        return postprocess_image(noise_image_tensor)
