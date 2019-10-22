import PIL.Image
import numpy as np
import cv2
import torchvision

def ElasticTransformations(alpha, sigma, random_state=np.random.RandomState(42)):
    """Returns a function to elastically transform multiple images."""
    
    # Good values for:
    #   alpha: 2000
    #   sigma: between 40 and 60
    
    grid_scale = 4
    alpha //= grid_scale
    sigma //= grid_scale

    def distort_elastic_cv2(image):
        # Originally from https://github.com/rwightman/tensorflow-litterbox
        
        image = np.array(image)
        """Elastic deformation of images as per [Simard2003].  """

        shape_size = image.shape[:2]

        # Downscaling the random grid and then upsizing post filter
        # improves performance. Approx 3x for scale of 4, diminishing returns after.
        grid_shape = (shape_size[0] // grid_scale, 
                      shape_size[1] // grid_scale)

        blur_size = int(4 * sigma) | 1
        
        rand_x = cv2.GaussianBlur(
            (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
            ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
        
        rand_y = cv2.GaussianBlur(
            (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
            ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
        
        if grid_scale > 1:
            rand_x = cv2.resize(rand_x, shape_size[::-1])
            rand_y = cv2.resize(rand_y, shape_size[::-1])

        grid_x, grid_y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
        
        grid_x = (grid_x + rand_x).astype(np.float32)
        grid_y = (grid_y + rand_y).astype(np.float32)

        distorted_img = cv2.remap(image, grid_x, grid_y,
                                  borderMode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_LINEAR)

        return distorted_img

    return distort_elastic_cv2

def RandomRotationWithMask(degrees, resample=False, expand=True, center=None):
    def _rotate(img):
        numpy_img = np.array(img)
        img, mask = numpy_img[:, :, 0:3], numpy_img[:, :, 3]
        
        pad = img.shape[0] // 3
        img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        mask = np.pad(mask, ((pad, pad), (pad, pad)), mode='reflect')

        angle = torchvision.transforms.RandomRotation.get_params([-degrees, degrees])
        img = PIL.Image.fromarray(img).rotate(angle, resample, expand, center)
        mask = PIL.Image.fromarray(mask).rotate(angle, resample, expand, center)
        
        img = np.array(img)[pad:-pad, pad:-pad]
        mask = np.array(mask)[pad:-pad, pad:-pad, None]
        img = np.concatenate((img, mask), axis=-1)
        return PIL.Image.fromarray(img)
    
    return _rotate