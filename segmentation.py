import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torchvision import transforms, models
import cv2
import random


model = models.segmentation.fcn_resnet50(pretrained=True).eval()
labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
# img = Image.open('test_img.jpg')
# background = Image.open('editing_img/0.jpg')
def segment(net, img, colors):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    output = model(input_batch)['out'][0] # (21, height, width)

    output_predictions = output.argmax(0).byte().cpu().numpy() # (height, width) 

    r = Image.fromarray(output_predictions).resize((img.shape[1], img.shape[0]))
    r.putpalette(colors)

    return r, output_predictions

def make_palette_map():
    cmap = plt.cm.get_cmap('tab20c')
    colors = (cmap(np.arange(cmap.N)) * 255).astype(np.int)[:, :3].tolist()
    np.random.seed(2020)
    np.random.shuffle(colors)
    colors.insert(0, [0, 0, 0]) # background color must be black
    colors = np.array(colors, dtype=np.uint8)

    palette_map = np.empty((10, 0, 3), dtype=np.uint8)
    legend = []

    for i in range(21):
        legend.append(mpatches.Patch(color=np.array(colors[i]) / 255., label='%d: %s' % (i, labels[i])))
        c = np.full((10, 10, 3), colors[i], dtype=np.uint8)
        palette_map = np.concatenate([palette_map, c], axis=1)  
    
    return colors, palette_map

def get_resized_background(img, background):
    fg_h, fg_w, _ = img.shape
    bg_h, bg_w, _ = background.shape
    background = cv2.resize(background, dsize=(fg_w, int(fg_w * bg_h / bg_w)))
    bg_h, bg_w, _ = background.shape
    margin = (bg_h - fg_h) // 2
    if margin > 0:
        background = background[margin:-margin, :, :]
    else:
        background = cv2.copyMakeBorder(background, top=abs(margin), bottom=abs(margin), left=0, right=0, borderType=cv2.BORDER_REPLICATE)
    background = cv2.resize(background, dsize=(fg_w, fg_h))
    return background


def get_alpha(pred):
    mask = (pred == 15).astype(float) * 255 # 15: person
    _, alpha = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    alpha = cv2.GaussianBlur(alpha, (7, 7), 0).astype(float)

    alpha = alpha / 255.
    alpha = np.repeat(np.expand_dims(alpha, axis=2), 3, axis=2)
    return alpha


def people_seg_mask(img):
    colors, _ = make_palette_map()
    img = np.array(img)
    _, pred = segment(model, img, colors)
    alpha = get_alpha(pred)

    return alpha
