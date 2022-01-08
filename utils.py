import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from datetime import datetime
from base64 import b64encode


def print_head():
    """Print head of logging
    """
    head = f"{'Timestamp':<22}{'Epoch':<10}{'Count':<12}\t\t{'Content loss':<20}{'Style loss':<20}{'Total loss':<20}"
    print(head)
    print("="*len(head))



def print_log(epoch, content_loss, style_loss, total_loss, count=None, len_dataset=None):
    """Print a row of logging
    """
    if (count is None and len_dataset is None):
        msg = (
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<22}"
            f"{epoch+1 :<10}"
            f"{' ':<12}\t\t"
            f"{content_loss:<20.1f}"
            f"{style_loss:<20.1f}"
            f"{total_loss:<20.1f}"
        )
    else:
        msg = (
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<22}"
            f"{epoch+1 :<10}"
            f"[{count:<4}/{len_dataset:>5}]\t\t"
            f"{content_loss:<20.1f}"
            f"{style_loss:<20.1f}"
            f"{total_loss:<20.1f}"
        )
    print(msg)



def make_gif_one_canvas(list_images):
    """Make an animation from a list of images
    """
    fig, ax = plt.subplots(1, 1)
    def frame(n):
        ax.clear()
        plot = ax.imshow(list_images[n])
        return plot

    anim = animation.FuncAnimation(fig, frame, frames=len(list_images), blit=False, repeat=True, interval=50)
    plt.close()
    return anim



def make_gif_both(list_images_original, list_images_style):
    """Make a subplot animation from two list of images
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plt.tight_layout()
    
    def frame(n):
        ax1.clear()
        ax1.imshow(list_images_original[n])
        ax2.clear()
        ax2.imshow(list_images_style[n])
        ax1.set_axis_off()
        ax2.set_axis_off()
           
    anim = animation.FuncAnimation(fig, frame, frames=len(list_images_style), blit=False, repeat=True, interval=50)
    plt.close()
    return anim



# =================== GATYS ===========================
def get_img_gatys(img):
    """Post process and convert a tensor to image, in the first section
    """
    post = transforms.Compose([
        transforms.Lambda(lambda x: x.mul_(1./255)),
        transforms.Normalize(mean=[-0.406, -0.456, -0.485], # add imagenet mean
                            std=[1,1,1]),
        transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), # turn to RGB
        transforms.Lambda(lambda x: torch.clip(x,0,1)),
        transforms.ToPILImage()
    ])

    return post(img.clone().data[0].cpu().squeeze())



def show_img_gatys(img, ax=None, title="", **kwargs):
    """Display an image from its tensor representation
    """
    disp_img = get_img_gatys(img)
    if ax:
        ax.imshow(disp_img)
        ax.set_title(title)
    else:
        plt.figure(*kwargs)
        plt.imshow(disp_img)
        plt.title(title)



def save_img_gatys(img, filename):
    """Save an image from its tensor reprensentation
    """
    get_img_gatys(img).save(filename)


# =================== JOHNSON ==========================

class ImageTransformNet(torch.nn.Module):
    """Stylizing network
    """
    def __init__(self):
        super(ImageTransformNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y



class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out



class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out



class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out



class VGG16Features(nn.Module):
    def __init__(self):
        super(VGG16Features, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x, out_keys):
        out = dict()
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['p3'] = self.pool3(out['r33'])
        
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['p4'] = self.pool4(out['r43'])
        
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['p5'] = self.pool5(out['r53'])
        
        return [out[key] for key in out_keys]



def initialize_pretrained_vgg16(return_full=False):
    """Initialize a pre-trained VGG-16 
    """
    full_vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    model = VGG16Features()
    
    # Retrieve parameters from the full VGG
    list_params_original = []
    for layer in full_vgg.features:
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            params = layer.parameters()
            list_params_original.append(torch.nn.utils.parameters_to_vector(params))


    # Set params to the new model
    i = 0
    for layer in model.children():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            torch.nn.utils.vector_to_parameters(list_params_original[i], layer.parameters())
            i += 1

    for param in model.parameters():
        param.requires_grad = False
            
    if return_full:
        return model, full_vgg
    else:
        return model



def load_model(model):
    """Load a stylizing network from a file
    """
    style_model = ImageTransformNet()
    state_dict = torch.load(model)
    style_model.load_state_dict(state_dict)
    return style_model



def show_image_johnson(data, ax=None):
    """Display an image from its tensor representation
    """
    img = get_image_johnson(data)
    if ax:
        ax.imshow(img)
    else:
        plt.figure()
        plt.imshow(img)



def get_image_johnson(data):
    """Convert an image from torch.tensor to np.array
    """
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    return img



def get_frame_gif(frame):
    """Get the frame from a gif
    """
    content_image = torch.tensor(frame, dtype=torch.float32)
    content_image = content_image.transpose(0, 2)
    content_image = content_image.transpose(1, 2)
    content_image = content_image[:3,:,:]
    content_image = content_image.unsqueeze(0)
    return content_image



def colab_video(video_path):
    mp4 = open(video_path,'rb').read()
    decoded_vid = "data:video/mp4;base64," + b64encode(mp4).decode()
    return f"""<video width=400 controls loop autoplay><source src={decoded_vid} type="video/mp4"></video>"""



def download_coco_dataset():
    """Download a part of the COCO Dataset and moves it in a coco/ folder
    """
    import fiftyone as fo
    import fiftyone.zoo as foz
    import shutil

    dataset = foz.load_zoo_dataset(
        "coco-2014",
        splits=["train"],
        max_samples=1000,
        dataset_dir="./coco_temp"
    )

    shutil.move("./coco_temp/train/data", "./")
    shutil.rmtree('coco_temp')
    os.rename('data', 'coco')



if __name__ == '__main__':
    download_coco_dataset()