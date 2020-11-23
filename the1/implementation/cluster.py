import os
import numpy as np
from sklearn.cluster import KMeans
import sys


def main(image_list="images.dat" ,mode="sift", number_of_clusters=32):
    cwd = os.getcwd()

    if mode=="sift":
        print("Mode: Sift")
        _path = cwd + "/sift_descriptor/"
        hist_path = cwd + "/sift_histogram/"
    elif mode=="dense_sift":
        print("Mode: Dense Sift.")
        _path = cwd + "/dsift_descriptor/"
        hist_path = cwd + "/dsift_histogram/"
    else:
        print("Unkown mode:", mode)
        return

    images = []
    try:    
        with open(image_list, "r") as im:
            imgs = im.readlines()
            for image in imgs:
                images.append(image.strip("\n"))
        
        stacking_list = []
        for image in images:
            desc_path = _path+image + ".npy"
            stacking_list.append( np.load(desc_path) )
    
    except IOError:
        print("Could not read file.")
        return

    _img_combined =  np.concatenate(stacking_list, axis=0) 
    np.random.shuffle(_img_combined)
    print("Loaded image descriptors with total shape of:", _img_combined.shape)
    
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0, verbose=False)
    print("Number of clusters:", number_of_clusters)
    print("Clustering...")

    size = int(_img_combined.shape[0]/4)
    print("Number of data points to fit:", size)
    kmeans.fit(_img_combined[:size] )

    del _img_combined
    _cluster_index = range(number_of_clusters)

    print("Extracting bag of features...")
    i = 0
    for image in images:
        _labels = kmeans.predict(stacking_list[i])        
        histogram = np.histogram(_labels, _cluster_index)
        _bof = histogram[0] / histogram[0].sum()
        os.makedirs(os.path.dirname(hist_path + image), exist_ok=True)
        np.save(hist_path+image, _bof)
        i += 1

    

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: 'python cluster.py [imageFiles.txt] [mode] [num_of_clusters]'. Modes={'sift', 'dense_sift'}")
        sys.exit()
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
