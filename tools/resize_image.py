import os
from PIL import Image

seg_data = '~/data/ugrp_nexys'
seg_data = os.path.expanduser(seg_data)
target_size = (256, 256)

for (path, dir, files) in os.walk(seg_data):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.png' or ext == '.jpg':
            image_path = os.path.join(path, filename)
            with Image.open(image_path) as image:
                image = image.resize(target_size)  # Image.ANTIALIAS
                image.save(image_path)