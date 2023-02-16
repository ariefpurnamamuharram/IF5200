import os
from PIL import Image


def get_square_resize(im: Image, dim: int):
    
    # Resize the image
    im = im.resize((dim, dim))
    
    # Return the resized image
    return im

def get_segment(im: Image, segment: int):
    
    if segment == 0 or segment > 3:
        raise ValueError('Segment number is out of index!')
    
    # Crop the image
    if segment == 1:
        im = im.crop((0, 0, im.width, round((im.height * 0.5), 0)))
    elif segment == 2:
        im = im.crop((0, round((im.height * 0.5), 0), im.width, im.height))
    elif segment == 3:
        im = im.crop((0, round((im.height * 0.15), 0), im.width, (im.height - round((im.height * 0.15), 0))))
    
    # Return the image segment
    return im
    