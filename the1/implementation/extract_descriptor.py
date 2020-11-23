from cyvlfeat.sift import sift, dsift
import os
from scipy.misc import imread
import numpy as np

def main(image_list="images.dat"):
    cwd = os.getcwd()
    data_path = cwd + "/dataset/"
    sift_path = cwd + "/sift_descriptor/"
    dsift_path = cwd + "/dsift_descriptor/"
    with open("images.dat", "r") as im:
        images = im.readlines() 
        i = 1
        total = len(images)
        for image in images:
            print(i, "/", total)
            image_matrice = imread(data_path + image.strip(), mode='F')
            sift_frame, sift_desc = sift(image_matrice, compute_descriptor=True)
            dsift_frame, dsift_desc = dsift(image_matrice, step=10)
            
            sift_image_path = sift_path + image.strip("\n")
            dsift_image_path = dsift_path + image.strip("\n")
            
            os.makedirs( os.path.dirname(sift_image_path), exist_ok=True)
            os.makedirs( os.path.dirname(dsift_image_path), exist_ok=True)
            np.save(sift_image_path, sift_desc)
            np.save(dsift_image_path, dsift_desc)
            i+=1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: 'python extract_descriptors.py [imageFiles.txt]")
        sys.exit()
    main()
