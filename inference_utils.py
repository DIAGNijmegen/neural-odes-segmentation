import math
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
    
    result = result[pad_h[0]:result.shape[0] - pad_h[1],
                    pad_w[0]:result.shape[1] - pad_w[1]]
    return result

def inference_image(image, unet=False):
    input_image, resized = prepare_image(image)
    result = eval_image(input_image, shouldpad=unet)
    result = np.argmax(result, axis=2)
    result = crop_result(result, resized)
    return result, resized

def eval_image(tile, TTA=False, resize=1, patch_size=512, shouldpad=False):
    net.eval();
    with torch.no_grad():
        
        # we should pad if the network is UNet
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
    
        soft_result = F.softmax(result[0], dim=1).cpu()
        soft_result_ud = F.softmax(result_ud[0], dim=1).cpu()
        soft_result_lr = F.softmax(result_lr[0], dim=1).cpu()

        soft_result_np = soft_result.detach().numpy().transpose(1, 2, 0)
        soft_result_np_ud = soft_result_ud.detach().numpy().transpose(1, 2, 0)
        soft_result_np_lr = soft_result_lr.detach().numpy().transpose(1, 2, 0)

        soft_result_np_ud = np.flipud(soft_result_np_ud)
        soft_result_np_lr = np.fliplr(soft_result_np_lr)

        if TTA: soft_result_np = (soft_result_np + soft_result_np_ud + soft_result_np_lr) / 3

        if shouldpad: 
            soft_result_np = soft_result_np[2:-2,2:-2]
            
        return soft_result_np