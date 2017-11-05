import glob, os
from PIL import Image

# Path to the original datasets
dataset_directory_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
dataset_path = os.path.abspath(dataset_directory_path) + "/4000unlabeledLP"

dataset_img_ext = "jpg"
img_type = "JPEG"
max_size = 270, 90

def process_dataset(new_dataset_path, dataset_img_ext, process):
    for file_path in os.listdir(dataset_path):
        file_path = dataset_path + "/" + file_path
        old_im = Image.open(file_path)
        file, ext = os.path.splitext(file_path)
        new_im = process(old_im)
        new_im.save(new_dataset_path + os.path.basename(file) + "." + dataset_img_ext, img_type)

def pad_with_black(old_im):
    old_size = old_im.size

    new_size_w = int((max_size[0] - old_size[0]) / 2)
    new_size_h = int((max_size[1] - old_size[1]) / 2)

    new_im = Image.new("RGB", max_size)
    new_im.paste(old_im, (new_size_w, new_size_h))

    return new_im

def pad_with_black_and_scale(im):
    im_size = im.size
    scale = min(max_size[1] / im_size[1], max_size[0] / im_size[0])
    new_im = im.resize((int(scale * im_size[0]), int(scale * im_size[1])), Image.ANTIALIAS)
    new_im = pad_with_black(new_im)
    return new_im

if __name__ == "__main__":
    new_path = os.path.abspath(dataset_directory_path) + "/4000unlabeledLP_same_dims_scaled/"
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        process_dataset(new_path, dataset_img_ext, pad_with_black_and_scale)
    else:
        print('Dataset directory already exists, no new dataset generated.')
