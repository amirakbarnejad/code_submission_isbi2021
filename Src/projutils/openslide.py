


import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import openslide





def magnfree_get_region(ops_image, w_at_5x, h_at_5x, w_resize, h_resize, x_center, y_center):
    '''
    Reads a region from a WSI. The input size is independent of the magnification level at which the slide is scanned.
    Inputs.
        - ops_image: an instance of openslide object. 
        - w_at_5x: the width of the extracted patch in 5x.
        - h_at_5x: the height of the extracted patch in 5x.
        - [w_resize, h_resize]: the final resizing after cropping [w_at_5x x h_at_5x] in 5x magnification level.
        - [x_center, y_center] : x,y coordinates of the center of the extracted patch.
    '''
    #compute w_at_mag and h_at_mag ======
    int_maglevel = int(ops_image.properties["aperio.AppMag"]) #this number will be, for instnace, 20 for 20x.
    assert(int_maglevel >= 5)
    list_downsamples = ops_image.level_downsamples
    w_at_mag, h_at_mag = int(w_at_5x*int_maglevel/5.0), int(h_at_5x*int_maglevel/5.0) #to get the equivalent information in the magnification level of the current slide.
    #extract the big patch =====
    x_topleft = int(x_center - w_at_mag*0.5)
    y_topleft = int(y_center - h_at_mag*0.5)
    pil_toret = ops_image.read_region([x_topleft, y_topleft], 0, [w_at_mag, h_at_mag]) 
    #resize to [w_resize, h_resize] =====
    return pil_toret.resize([w_resize, h_resize])
