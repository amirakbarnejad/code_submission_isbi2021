


import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import openslide



def get_listpolygons(str_fnamexml):
    '''
    Given an xml file produced by ASAP, this function returns the list of the annotated
    polygons.
    '''
    tree = ET.parse(str_fnamexml)
    root = tree.getroot() #ASAP annotations
    for child in list(root.getchildren()):
        if(child.tag == "Annotations"):
            break
    #now child is annotations
    list_polygons = []
    for idx_annotation, annotation in enumerate(list(child.getchildren())):
        print("annotation {}".format(idx_annotation))
        coordinates = list(annotation.getchildren())[0]
        list_coordinate = list(coordinates.getchildren())
        list_xy = []
        for coordinate in list_coordinate:
            list_xy.append([coordinate.get("X"), coordinate.get("Y")])
        print(np.array(list_xy))
        list_polygons.append(np.array(list_xy))
        print("\n\n\n")
    return list_polygons





def get_foreground_from_polyg(fname_wsi, fname_xml, scale):
    '''
    Given a wsi and the asap's annotated polygons, it returns the foreground mask
    of the polygons.
    '''
    #find the size of the thumbnail
    ops_image = openslide.OpenSlide(fname_wsi)
    W, H = ops_image.dimensions
    w, h = int(W*scale), int(H*scale)
    #get the polygons
    list_polygons = get_listpolygons(fname_xml)
    #draw the polyongs
    im = Image.new('RGB', (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    for polygon in list_polygons:
        polygon = polygon.astype(np.float32)
        draw.polygon(polygon*scale,\
                     fill=(255, 255, 255), outline=(0,0,0))
    toret = np.array(im)
    return toret
