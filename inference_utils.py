import math
import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage
import skimage.measure
import PIL

def resize_image(image):
    ratio = (775 / 512)
    new_size = (int(round(image.size[0] / ratio)), 
                int(round(image.size[1] / ratio)))

    image = image.resize(new_size)
    return image

def pad_image(image):
    pad_h = max((352 - image.shape[0]) / 2, 0)
    pad_w = max((512 - image.shape[1]) / 2, 0)
    pad_h = (math.floor(pad_h), math.ceil(pad_h))
    pad_w = (math.floor(pad_w), math.ceil(pad_w))

    # pad to image size
    padded_image = np.pad(image, ((pad_h[0], pad_h[1]), (pad_w[0], pad_w[1]), (0, 0)), mode='reflect')
    return padded_image
    
def prepare_image(image):
    resized = resize_image(image)
    resized = np.array(resized)
    padded = pad_image(resized)
    return padded, resized
    
def crop_result(result, image):
    pad_h = max((352 - image.shape[0]) / 2, 0)
    pad_w = max((512 - image.shape[1]) / 2, 0)
    pad_h = (math.floor(pad_h), math.ceil(pad_h))
    pad_w = (math.floor(pad_w), math.ceil(pad_w))
    
    result = result[:,
                    pad_h[0]:result.shape[1] - pad_h[1],
                    pad_w[0]:result.shape[2] - pad_w[1]]
    return result

def inference_image(net, image, shouldpad=False):
    input_image, resized = prepare_image(image)
    result = eval_image(net, input_image, shouldpad=shouldpad)
    result = crop_result(result, resized)
    return result, resized

def eval_image(net, tile, TTA=True, resize=1, patch_size=512, shouldpad=False):
    net.eval();
    with torch.no_grad():
        if shouldpad:
            pad = 192
            padded_np_image = np.pad(tile, ((pad//2, pad//2), (pad//2, pad//2), (0, 0)), mode='reflect')
            tile = padded_np_image[0:704,0:704]

        # TTA
        transposed_image = tile.transpose(2, 0, 1) / 255
        transposed_image_ud = np.flipud(tile).transpose(2, 0, 1) / 255
        transposed_image_lr = np.fliplr(tile).transpose(2, 0, 1) / 255
        
        torch_image = torch.from_numpy(transposed_image).float()
        torch_image_ud = torch.from_numpy(transposed_image_ud).float()
        torch_image_lr = torch.from_numpy(transposed_image_lr).float()

        result = net(torch_image[None].cuda())
        result_ud = net(torch_image_ud[None].cuda())
        result_lr = net(torch_image_lr[None].cuda())

        soft_result = torch.sigmoid(result)[0].cpu()
        soft_result_ud = torch.sigmoid(result_ud)[0].cpu()
        soft_result_lr = torch.sigmoid(result_lr)[0].cpu()

        soft_result_np = soft_result.detach().numpy().transpose(1, 2, 0)
        soft_result_np_ud = soft_result_ud.detach().numpy().transpose(1, 2, 0)
        soft_result_np_lr = soft_result_lr.detach().numpy().transpose(1, 2, 0)
        
        if shouldpad: soft_result_np = soft_result_np
        if shouldpad: soft_result_np_ud = soft_result_np_ud
        if shouldpad: soft_result_np_lr = soft_result_np_lr

        soft_result_np_ud = np.flipud(soft_result_np_ud)
        soft_result_np_lr = np.fliplr(soft_result_np_lr)
        soft_result_np_lr_ud = np.fliplr(np.flipud(soft_result_np_lr)) # incorrect

        if TTA: soft_result_np = (soft_result_np + soft_result_np_ud + soft_result_np_lr) / 3

        if shouldpad: soft_result_np = soft_result_np[2:-2,2:-2]
        return soft_result_np.transpose(2, 0, 1)

def split_objects(image):
    return (image[0] > 0.7)

def remove_small_object(labeled_image, threshold=500):
    regionprops = skimage.measure.regionprops(labeled_image)
    new_results = np.array(labeled_image).copy()
    for prop in regionprops:
        if prop.area < threshold:
            new_results[new_results == prop.label] = 0
    return new_results

def grow_to_fill_borders(image, result):
    grow_labeled = image

    for i in range(10):
        new_labeled = scipy.ndimage.maximum_filter(grow_labeled, 3)
        grow_labeled[result==1] = new_labeled[result==1]
    grow_labeled[result==0] = 0
    return grow_labeled

def hole_filling_per_object(image):
    grow_labeled = image
    for i in np.unique(grow_labeled):
        if i == 0: continue
        filled = scipy.ndimage.morphology.binary_fill_holes(grow_labeled == i)
        grow_labeled[grow_labeled == i] = 0
        grow_labeled[filled == 1] = i
    return grow_labeled

def resize_to_size(image, gt):
    new_results_img = PIL.Image.fromarray(image.squeeze().astype(np.uint8))
    new_results_img = new_results_img.resize(gt.size)
    new_results_img = np.array(new_results_img)
    return new_results_img

def postprocess(result, image):
    splitted = split_objects(result)
    labeled = skimage.measure.label(np.array(splitted))
    temp = remove_small_object(labeled, threshold=500)
    growed = grow_to_fill_borders(temp, result[1] > 0.5)
    hole_filled = hole_filling_per_object(growed)
    temp = remove_small_object(hole_filled, threshold=500)
    final = resize_to_size(temp, image)
    return final
