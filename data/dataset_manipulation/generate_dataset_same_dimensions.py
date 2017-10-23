# Path to the original datasets
dataset_path = "../4000unlabeledLP/"
dataset_img_ext = "jpg"
img_type = "JPEG"

# Process the images and save to new path
import glob, os
from PIL import Image

def get_max_dims(dataset_path, dataset_img_ext):
    max_width = 0
    max_height = 0

    for infile in glob.glob(dataset_path + "*." + dataset_img_ext):
        im = Image.open(infile)
        width, height = im.size
        max_width = max(max_width, width)
        max_height = max(max_height, height)

    return (max_width, max_height)

def process_dataset(new_dataset_path, dataset_img_ext, process):
    for infile in glob.glob(dataset_path + "*." + dataset_img_ext):
        old_im = Image.open(infile)
        file, ext = os.path.splitext(infile)
        new_im = process(old_im)
        new_im.save(new_dataset_path + os.path.basename(file) + "." + dataset_img_ext, img_type)

# max_size = get_max_dims(dataset_path, dataset_img_ext)
max_size = 270, 90

def pad_with_black(old_im):
    old_size = old_im.size
    new_im = Image.new("RGB", max_size)
    new_im.paste(old_im, (int((max_size[0]-old_size[0])/2),
                      int((max_size[1]-old_size[1])/2)))
    return new_im

def pad_with_black_and_scale(im):
    im_size = im.size
    scale = min(max_size[1]/im_size[1], max_size[0]/im_size[0])
    new_im = im.resize((int(scale * im_size[0]), int(scale * im_size[1])), Image.ANTIALIAS)
    new_im = pad_with_black(new_im)
    return new_im

# process_dataset("../4000unlabeledLP_same_dims/", dataset_img_ext, pad_with_black)
process_dataset("../4000unlabeledLP_same_dims_scaled/", dataset_img_ext, pad_with_black_and_scale)
