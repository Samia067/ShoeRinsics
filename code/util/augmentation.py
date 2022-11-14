import torchvision.transforms.functional as TF

'''Get 3 flip augmentations.'''
def get_image_flip_modifications(image):
    transforms = ["flip horizontal", "flip vertical", "flip both"]
    images = [TF.hflip(image), TF.vflip(image), TF.hflip(TF.vflip(image))]
    return images, transforms

'''Get rotate augmentations.'''
def get_image_rotate_modifications(image):
    transforms = ["rotate 5", "rotate 10", "rotate -5", "rotate -10"]

    fill = image[0, 0, 0, 0].item()
    images = [
        # rotate - 5, 10, -5, -10
        TF.rotate(image, 5, fill=fill), TF.rotate(image, 10, fill=fill),
        TF.rotate(image, -5, fill=fill), TF.rotate(image, -10, fill=fill)]
    return images, transforms

'''Get scale augmentations.'''
def get_image_scale_modifications(image):
    transforms = ["scale 0.5", "scale 0.8", "scale 1.5", "scale 1.8"]
    images = [
        # scale - 0.5, 0.8, 1.5, 2
        TF.resize(image, size=int(image.shape[2] * 0.5), antialias=True),
        TF.resize(image, size=int(image.shape[2] * 0.8), antialias=True),
        TF.resize(image, size=int(image.shape[2] * 1.5), antialias=True),
        TF.resize(image, size=int(image.shape[2] * 1.8), antialias=True),
    ]
    return images, transforms

'''Get all test-time image augmentations.'''
def get_image_modifications(image):
    transforms = ['original']
    images = [image]

    flip_images, flip_transforms = get_image_flip_modifications(image)
    transforms.extend(flip_transforms)
    images.extend(flip_images)

    rotate_images, rotate_transforms = get_image_rotate_modifications(image)
    transforms.extend(rotate_transforms)
    images.extend(rotate_images)

    scale_images, scale_transforms = get_image_scale_modifications(image)
    transforms.extend(scale_transforms)
    images.extend(scale_images)

    for rotate_image, rotate_transform in zip(rotate_images, rotate_transforms):
        rotate_flip_images, rotate_flip_transforms = get_image_flip_modifications(rotate_image)
        transforms.extend([rotate_transform + ' ' + x for x in rotate_flip_transforms])
        images.extend(rotate_flip_images)

    return images, transforms


'''Reverse effect of test-time image augmentation.'''
def reverse_modification(image, transform, label, original_shape=(384, 768)):
    assert (label in ['albedo', 'depth'])
    fill = image[0, 0, 0, 0].item()
    transform = transform.split()
    if len(transform) > 2:
        image = reverse_modification(image, ' '.join(transform[-2:]), label, original_shape=original_shape)
        transform = ' '.join(transform[:-2]).split()
    if transform[0] == 'flip':
        if transform[1] == 'horizontal':
            image = TF.hflip(image)
        elif transform[1] == 'vertical':
            image = TF.vflip(image)
        elif transform[1] == 'both':
            image = TF.hflip(TF.vflip(image))
    elif transform[0] == 'rotate':
        angle = int(transform[1])
        image = TF.rotate(image, -angle, fill=fill)
    elif transform[0] == 'scale':
        scale = float(transform[1])
        image = TF.resize(image, size=original_shape, antialias=True)
        if label == 'depth':
            image = image / scale
    return image

