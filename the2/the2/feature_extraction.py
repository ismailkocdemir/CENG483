#####################################################################################
#																					#
#	 This is an example script for extracting features of an image. For your own    #
#	 self portraits, you should get inspiration from this script. After extracting  #
#	 the feature vector, you can use it with your trained network.                  #
#																					#
#	 Note that, this requires torchvision, Pillow and NumPy packages.				#
#	 You are not forced to totally understand how the feature extractor works.      #
#	 You can just ignore the warnings given by the script.							#
#																					#
#####################################################################################
import sys

from PIL import Image
import numpy as np
from img_to_vec import Img2Vec

if __name__ == '__main__':
    if len(sys.argv) != 2:
        pritn("Give the path of the image as an argument.")
        sys.exit(0)
    image_path = sys.argv[1]
    fe = Img2Vec(cuda=False) # change this if you use Cuda version of the PyTorch.
    img = Image.open(image_path)
    img = img.resize((224, 500))
    feats = fe.get_vec(img).reshape(1, -1)
    np.save(image_path + "_features.npy", feats)
    print(type(feats), feats.shape)
