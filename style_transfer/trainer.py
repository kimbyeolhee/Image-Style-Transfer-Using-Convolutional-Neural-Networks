import os

import torch
import torch.optim as optim
from tqdm import tqdm

from .data import DataModule
from .loss import ContentLoss, StyleLoss
from .model import StyleTransfer


def compute_loss(
    model,
    content_image_tensor,
    style_image_tensor,
    noise_image_tensor,
    content_loss_fn,
    style_loss_fn,
    cfg,
):
    x_content_list = model(noise_image_tensor, mode="content")
    y_content_list = model(content_image_tensor, mode="content")

    x_style_list = model(noise_image_tensor, mode="style")
    y_style_list = model(style_image_tensor, mode="style")

    content_loss, style_loss, total_loss = 0.0, 0.0, 0.0

    for x_content, y_content in zip(x_content_list, y_content_list):
        content_loss += content_loss_fn(x_content, y_content)

    for x_style, y_style in zip(x_style_list, y_style_list):
        style_loss += style_loss_fn(x_style, y_style)

    total_loss = cfg.trainer.alpha * content_loss + cfg.trainer.beta * style_loss

    return content_loss, style_loss, total_loss


def train(
    model,
    datamodule,
    content_image_tensor,
    style_image_tensor,
    noise_image_tensor,
    optimizer,
    content_loss_fn,
    style_loss_fn,
    current_epoch,
    cfg,
):
    model.eval()

    def closure():
        optimizer.zero_grad()
        content_loss, style_loss, total_loss = compute_loss(
            model,
            content_image_tensor,
            style_image_tensor,
            noise_image_tensor,
            content_loss_fn,
            style_loss_fn,
            cfg,
        )
        total_loss.backward()
        return total_loss

    if isinstance(optimizer, optim.LBFGS):
        optimizer.step(closure)
    else:
        content_loss, style_loss, total_loss = compute_loss(
            model,
            content_image_tensor,
            style_image_tensor,
            noise_image_tensor,
            content_loss_fn,
            style_loss_fn,
            cfg,
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    if current_epoch % cfg.trainer.step_size == 0:
        if isinstance(optimizer, optim.LBFGS):
            with torch.no_grad():
                content_loss, style_loss, total_loss = compute_loss(
                    model,
                    content_image_tensor,
                    style_image_tensor,
                    noise_image_tensor,
                    content_loss_fn,
                    style_loss_fn,
                    cfg,
                )
                print(
                    f"Content Loss: {content_loss.cpu().item()}, Style Loss: {style_loss.cpu().item()}, Total Loss: {total_loss.cpu().item()}"
                )
        else:
            print(
                f"Content Loss: {content_loss.cpu().item()}, Style Loss: {style_loss.cpu().item()}, Total Loss: {total_loss.cpu().item()}"
            )

        gen_image = datamodule.get_noise_image_tensor_to_image(noise_image_tensor)
        if not os.path.exists(r"C:\Users\User\neural_style_transfer\output"):
            os.makedirs(r"C:\Users\User\neural_style_transfer\output")
        gen_image.save(
            rf"C:\Users\User\neural_style_transfer\output\{cfg.trainer.optimizer}_epoch_{current_epoch}.jpg"
        )


def run(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    datamodule = DataModule(cfg.data.content_image_path, cfg.data.style_image_path)
    content_image_tensor, style_image_tensor = datamodule.get_image_tensors()

    content_image_tensor = content_image_tensor.to(device)
    style_image_tensor = style_image_tensor.to(device)

    # Load Model
    model = StyleTransfer().to(device)

    # Load Loss
    content_loss_fn = ContentLoss()
    style_loss_fn = StyleLoss()

    # Initialize Noise Image
    noise_image_tensor = content_image_tensor.clone().requires_grad_(True)
    # noise_image_tensor = (
    #     torch.randn(content_image_tensor.shape).to(device).requires_grad_(True)
    # )

    if cfg.trainer.optimizer == "Adam":
        optimizer = optim.Adam([noise_image_tensor], lr=cfg.trainer.lr)
    elif cfg.trainer.optimizer == "LBGFS":
        optimizer = optim.LBFGS([noise_image_tensor], lr=cfg.trainer.lr)
    else:
        raise ValueError(f"Select optimizer from Adam and LBGFS")

    # Train Model
    for epoch in tqdm(range(cfg.trainer.epochs)):
        train(
            model,
            datamodule,
            content_image_tensor,
            style_image_tensor,
            noise_image_tensor,
            optimizer,
            content_loss_fn,
            style_loss_fn,
            current_epoch=epoch,
            cfg=cfg,
        )
