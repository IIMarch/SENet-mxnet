import cv2
import random
import numpy as np

def scale_down(src_size, size):
    """Scale down crop size if it's bigger than image size."""
    w, h = size
    sw, sh = src_size
    if sh < h:
        w, h = float(w*sh)/h, sh
    if sw < w:
        w, h = sw, float(h*sw)/w
    return int(w), int(h)

def resize_short(src, size, interp=2):
    """Resize shorter edge to size."""
    h, w, _ = src.shape
    if h > w:
        new_h, new_w = size*h/w, size
    else:
        new_h, new_w = size, size*w/h
    return cv2.resize(src, (new_w, new_h), interpolation=interp)

def resize_fixed(src, size, interp=2):
    return cv2.resize(src, (size, size), interpolation=interp)

def fixed_crop(src, x0, y0, w, h, size=None, interp=2):
    """Crop src at fixed location, and (optionally) resize it to size."""
    out = src[y0:y0+h, x0:x0+w, :]
    if size is not None and (w, h) != size:
        out = cv2.resize(out, size, interpolation=interp)
    return out

def random_crop(src, size, interp=2):
    """Randomly crop src with size. Upsample result if src is smaller than size."""
    h, w, _ = src.shape
    new_w, new_h = scale_down((w, h), size)

    x0 = random.randint(0, w - new_w)
    y0 = random.randint(0, h - new_h)

    out = fixed_crop(src, x0, y0, new_w, new_h, size, interp)
    return out, (x0, y0, new_w, new_h)

def center_crop(src, size, interp=2):
    """Randomly crop src with size. Upsample result if src is smaller than size."""
    h, w, _ = src.shape
    new_w, new_h = scale_down((w, h), size)

    x0 = (w - new_w)/2
    y0 = (h - new_h)/2

    out = fixed_crop(src, x0, y0, new_w, new_h, size, interp)
    return out, (x0, y0, new_w, new_h)

def color_normalize(src, mean, std=None):
    """Normalize src with mean and std."""
    if mean is not None:
        src -= mean
    if std is not None:
        src /= std
    return src

def random_size_crop(src, size, min_area, ratio, interp=2):
    """Randomly crop src with size. Randomize area and aspect ratio."""
    h, w, _ = src.shape
    new_ratio = random.uniform(*ratio)
    if new_ratio * h > w:
        max_area = w*int(w/new_ratio)
    else:
        max_area = h*int(h*new_ratio)

    min_area *= h*w
    if max_area < min_area:
        return random_crop(src, size, interp)
    new_area = random.uniform(min_area, max_area)
    new_w = int(np.sqrt(new_area*new_ratio))
    new_h = int(np.sqrt(new_area/new_ratio))

    assert new_w <= w and new_h <= h
    x0 = random.randint(0, w - new_w)
    y0 = random.randint(0, h - new_h)

    out = fixed_crop(src, x0, y0, new_w, new_h, size, interp)
    return out, (x0, y0, new_w, new_h)

def ResizeAug(size, interp=2):
    """Make resize shorter edge to size augumenter."""
    def aug(src):
        """Augumenter body"""
        return resize_short(src, size, interp)
    return aug

def ResizeFix(size, interp=2):
    def aug(src):
        return resize_fixed(src, size, interp)
    return aug

def RandomCropAug(size, interp=2):
    """Make random crop augumenter"""
    def aug(src):
        """Augumenter body"""
        return random_crop(src, size, interp)[0]
    return aug

def RandomSizedCropAug(size, min_area, ratio, interp=2):
    """Make random crop with random resizing and random aspect ratio jitter augumenter."""
    def aug(src):
        """Augumenter body"""
        return random_size_crop(src, size, min_area, ratio, interp)[0]
    return aug

def CenterCropAug(size, interp=2):
    """Make center crop augmenter."""
    def aug(src):
        """Augumenter body"""
        return center_crop(src, size, interp)[0]
    return aug

def RandomOrderAug(ts):
    """Apply list of augmenters in random order"""
    def aug(src):
        """Augumenter body"""
        src = [src]
        random.shuffle(ts)
        for t in ts:
            src = [j for i in src for j in t(i)]
        return src
    return aug

def ColorJitterAug(brightness, contrast, saturation):
    """Apply random brightness, contrast and saturation jitter in random order."""
    ts = []
    coef = np.array([[[0.299, 0.587, 0.114]]])
    if brightness > 0:
        def baug(src):
            """Augumenter body"""
            alpha = 1.0 + random.uniform(-brightness, brightness)
            src *= alpha
            return src
        ts.append(baug)

    if contrast > 0:
        def caug(src):
            """Augumenter body"""
            alpha = 1.0 + random.uniform(-contrast, contrast)
            gray = src*coef
            gray = (3.0*(1.0-alpha)/gray.size)*np.sum(gray)
            src *= alpha
            src += gray
            return src
        ts.append(caug)

    if saturation > 0:
        def saug(src):
            """Augumenter body"""
            alpha = 1.0 + random.uniform(-saturation, saturation)
            gray = src*coef
            gray = np.sum(gray, axis=2, keepdims=True)
            gray *= (1.0-alpha)
            src *= alpha
            src += gray
            return src
        ts.append(saug)
    return RandomOrderAug(ts)

def LightingAug(alphastd, eigval, eigvec):
    """Add PCA based noise."""
    def aug(src):
        """Augumenter body"""
        alpha = np.random.normal(0, alphastd, size=(3,))
        rgb = np.dot(eigvec*alpha, eigval)
        src += np.array(rgb)
        return src
    return aug

def ColorNormalizeAug(mean, std):
    """Mean and std normalization."""
    def aug(src):
        """Augumenter body"""
        return color_normalize(src, mean, std)
    return aug

def HorizontalFlipAug(p):
    """Random horizontal flipping."""
    def aug(src):
        """Augumenter body"""
        if random.random() < p:
            src = np.flip(src, axis=1)
        return src
    return aug

def CastAug():
    """Cast to float32"""
    def aug(src):
        """Augumenter body"""
        src = src.astype(np.float32)
        return src
    return aug

def CreateAugmenter(data_shape, resize=0, rand_crop=False, rand_resize=False, rand_mirror=False,
                    mean=None, std=None, brightness=0, contrast=0, saturation=0,
                    pca_noise=0, inter_method=2):
    """Create augumenter list."""
    auglist = []
    if resize > 0:
        auglist.append(ResizeAug(resize, inter_method))

    crop_size = (data_shape[2], data_shape[1])
    if rand_resize:
        assert rand_crop
        auglist.append(RandomSizedCropAug(crop_size, 0.3, (3.0/4.0, 4.0/3.0), inter_method))
    elif rand_crop:
        auglist.append(RandomCropAug(crop_size, inter_method))
    else:
        if data_shape[2] != resize or data_shape[1] != resize:
            auglist.append(CenterCropAug(crop_size, inter_method))
        else:
            pass

    if rand_mirror:
        auglist.append(HorizontalFlipAug(0.5))

    auglist.append(CastAug())

    if brightness or contrast or saturation:
        auglist.append(ColorJitterAug(brightness, contrast, saturation))

    if pca_noise > 0:
        eigval = np.array([55.46, 4.794, 1.148])
        eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948, 0.4203]])
        auglist.append(LightingAug(pca_noise, eigval, eigvec))

    if mean is None:
        mean = np.array([123.68, 116.28, 103.53])
    elif mean is not None:
        assert isinstance(mean, np.ndarray) and mean.shape[0] in [1, 3]

    if std is None:
        std = np.array([58.395, 57.12, 57.375])
    elif std is not None:
        assert isinstance(std, np.ndarray) and std.shape[0] in [1, 3]

    if mean is not None or std is not None:
        auglist.append(ColorNormalizeAug(mean, std))

    return auglist

class Augmenter(object):
    def __init__(self, data_shape, resize, rand_crop, rand_resize,
                 rand_mirror, mean, std):
        self.data_shape = data_shape
        self.resize = resize
        self.rand_crop = rand_crop
        self.rand_resize = rand_resize
        self.rand_mirror = rand_mirror
        self.mean = mean
        self.std = std
        self.augmenters = CreateAugmenter(
            data_shape=data_shape, resize=resize, rand_crop=rand_crop,
            rand_resize=rand_resize, rand_mirror=rand_mirror, mean=mean, std=std)

    def __call__(self, im):
        for aug in self.augmenters:
            im = aug(im)
        return im
