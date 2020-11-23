import numpy as np
import os
import sys


def main( query_list="validation_quueries.dat", image_list="images.dat", mode="sift"):
    cwd = os.getcwd()   
    
    if mode == "sift":
        _bof_path = cwd + "/sift_histogram/" 
    elif mode == "dense_sift":
        _bof_path = cwd + "/dsift_histogram/"
    else:
        print("Unknown mode.")
        return
    
    try:
        with open(query_list, "r") as qs, open("result.out", "w") as o, open(image_list, "r") as im:
            queries = qs.readlines()
            images = im.readlines()
            i = 1
            total_q = len(queries)
            for query in queries:
                query = query.strip().strip("\n")
                if i%50==0 or i==0:
                    print('query: ' + str(i) + "/" + str(total_q))
                i += 1

                query_image = np.load(_bof_path + query + ".npy")
                result_line = "{}:".format(query, end='')
                for image in images:
                    image = image.strip().strip("\n")
                    
                    test_image = np.load(_bof_path + image + ".npy")
                    if query_image.shape == test_image.shape:
                        euc_distance = np.sqrt(np.sum(np.square(query_image - test_image)))
                        result_line += " {} {}".format(float(euc_distance), image.strip("\n"), end='')
                    else:
                        print("Histogram shapes do not match.")
                        return

                if len(result_line.strip().split(" ")) > 1:
                    o.write(result_line + "\n")
    except IOError:
        print("Could not read file.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: 'python make_query.py [queryFile.txt] [imageFile.txt] [mode]'. Modes={'sift', 'dense_sift'}")
        sys.exit()
    main(sys.argv[1], sys.argv[2], sys.argv[3])

