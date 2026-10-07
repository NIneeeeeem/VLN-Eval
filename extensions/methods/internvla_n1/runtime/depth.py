"""Simulator-independent depth resize, extracted from InternNav.

Source revision and MIT notice are recorded alongside this module.
"""
from PIL import Image
from transformers.image_utils import to_numpy_array

def preprocess_depth_image_v2(depth_image, do_depth_scale=True, depth_scale=1000, target_height=384, target_width=384):
    resized_depth_image = depth_image.resize((target_width, target_height), Image.NEAREST)

    img = to_numpy_array(resized_depth_image)
    if do_depth_scale:
        img = img / depth_scale

    return img, (target_width, target_height)

