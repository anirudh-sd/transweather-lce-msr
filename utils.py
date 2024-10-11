import torch
import math
import time
import numpy as np
import sys
from tqdm import tqdm
import os
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transweather_model import Transweather  
import os
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch
from torch import nn
import os
from collections import OrderedDict
from tqdm import tqdm 



def to_psnr(pred_image, gt):
    """
    Computes the PSNR (Peak Signal-to-Noise Ratio) between the predicted image and the ground truth.

    Args:
        pred_image (torch.Tensor): The predicted image tensor with values in [0, 1].
        gt (torch.Tensor): The ground truth image tensor with values in [0, 1].

    Returns:
        list: A list of PSNR values for each image in the batch.
    """
    mse = torch.mean((pred_image - gt) ** 2, dim=[1, 2, 3])  # Mean Squared Error per image in batch
    psnr = 10 * torch.log10(1 / mse)
    return psnr.detach().cpu().numpy().tolist()

def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, exp_name):
    """
    Prints and logs the training and validation metrics.

    Args:
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
        one_epoch_time (float): Time taken for one epoch.
        train_psnr (float): Average training PSNR.
        val_psnr (float): Average validation PSNR.
        val_ssim (float): Average validation SSIM.
        exp_name (str): Experiment name for logging.
    """
    message = 'Epoch [{}/{}], Time: {:.2f}s, Train_PSNR: {:.2f}, Val_PSNR: {:.2f}, Val_SSIM: {:.4f}'.format(
        epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim)
    print(message)
    # Write to a log file
    log_file = './{}/training_log.txt'.format(exp_name)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def validation(net, val_data_loader, device, exp_name, save_tag=False):
    """
    Evaluates the model on the validation dataset.

    Args:
        net (torch.nn.Module): The neural network model.
        val_data_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device to run the computations on.
        exp_name (str): Experiment name for saving results.
        save_tag (bool): If True, saves the predicted images.

    Returns:
        tuple: Average PSNR and SSIM over the validation dataset.
    """
    net.eval()
    psnr_list = []
    ssim_list = []
    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_data_loader), 
                            total=len(val_data_loader), 
                            desc=f'Validation - {exp_name}', 
                            leave=False)
        
        for batch_id, val_data in progress_bar:    
        # for batch_id, val_data in tenumerate(val_data_loader):
            input_image, gt, image_name = val_data
            input_image = input_image.to(device)
            gt = gt.to(device)

            pred_image = net(input_image)

            # Clamp the output to [0,1]
            pred_image = torch.clamp(pred_image, 0, 1)

            # Compute PSNR
            psnr = to_psnr(pred_image, gt)
            psnr_list.extend(psnr)

            # Compute SSIM
            pred_image_np = pred_image.cpu().numpy()
            gt_np = gt.cpu().numpy()
            for i in range(pred_image_np.shape[0]):
                pred_i = pred_image_np[i].transpose(1, 2, 0)
                gt_i = gt_np[i].transpose(1, 2, 0)
                # Convert to uint8
                pred_i = (pred_i * 255.0).astype(np.uint8)
                gt_i = (gt_i * 255.0).astype(np.uint8)
                ssim = compare_ssim(pred_i, gt_i, multichannel=True, data_range=255)
                ssim_list.append(ssim)

            # Update the progress bar with current metrics
            progress_bar.set_postfix({'PSNR': psnr, 'SSIM': ssim})                

            if save_tag:
                # Save the predicted images
                save_dir = './{}/val_results/'.format(exp_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                for i in range(pred_image_np.shape[0]):
                    img_name = os.path.basename(image_name[i])
                    pred_i = pred_image_np[i].transpose(1, 2, 0)
                    pred_i = (pred_i * 255.0).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_dir, img_name), cv2.cvtColor(pred_i, cv2.COLOR_RGB2BGR))

    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    return avg_psnr, avg_ssim

def adjust_learning_rate(optimizer, epoch, initial_lr=2e-4, decay_rate=0.5, decay_epoch=50):
    """
    Adjusts the learning rate of the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate needs adjustment.
        epoch (int): Current epoch number.
        initial_lr (float): Initial learning rate.
        decay_rate (float): The rate at which to decay the learning rate.
        decay_epoch (int): The number of epochs after which to decay the learning rate.
    """
    lr = initial_lr * (decay_rate ** (epoch // decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_model(exp_name):
    net = Transweather()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    device_count = torch.cuda.device_count()
    checkpoint_path = f'./{exp_name}/best'
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please check the path.")
    else:
        # Load the state_dict from the checkpoint
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        if device_count == 1 or device_count == 0:
            # Single GPU or CPU: Remove 'module.' prefix from state_dict keys
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    name = k[7:]  # remove 'module.' prefix
                else:
                    name = k
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)
            print('Model weights loaded successfully on a single GPU or CPU.')
        else:
            # Multiple GPUs: Wrap the model with DataParallel and load state_dict as is
            net = nn.DataParallel(net)
            net.load_state_dict(state_dict)
            print('Model weights loaded successfully using DataParallel.')
    
    net.eval()
    return net, device

def preprocess_image(image_path, device):
    # Load the image using PIL
    input_img = Image.open(image_path).convert('RGB')
    
    # Define the transformations (same as during training)
    transform_input = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Apply the transformations
    input_tensor = transform_input(input_img)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)
    return input_tensor, input_img.size


def postprocess_output(output_tensor):
    # Remove batch dimension and move to CPU
    output_tensor = output_tensor.squeeze(0).cpu()
    
    # Ensure the output is in [0, 1] range
    output_tensor = torch.clamp(output_tensor, 0, 1)
    
    # Convert tensor to PIL image
    to_pil = transforms.ToPILImage()
    output_image = to_pil(output_tensor)
    return output_image

def run_inference(image_path, net,device):
    
    # Preprocess the image
    input_tensor, original_size = preprocess_image(image_path,device)
    
    # Run inference
    with torch.no_grad():
        output_tensor = net(input_tensor)
    
    # Postprocess the output
    output_image = postprocess_output(output_tensor)
    
    # Optionally, resize the output image back to original size
    output_image = output_image.resize(original_size, Image.ANTIALIAS)
    
    return output_image
