import os
import h5py
import sys
import numpy as np
from skimage.io import imread, imsave

def main(EVdir_s):
    EVdir = "./CVPPP_"+EVdir_s
    RSdir = EVdir+"_rf"
    save_name = RSdir +'.h5'
    if not os.path.exists(EVdir):
        os.makedirs(EVdir)
    if not os.path.exists(RSdir):
        os.makedirs(RSdir)
    

    GTdir = "/work/scratch/wu/CVPPP2017_CodaLab/sub_example"
    for group in os.listdir(EVdir):

        EV_whole_dir = os.path.join(EVdir, group)
        GT_whole_dir = os.path.join(GTdir, group)
        for EV_img_file in os.listdir(EV_whole_dir):
            EV_img = imread(os.path.join(EV_whole_dir, EV_img_file))
            GT_img = imread(os.path.join(GT_whole_dir, EV_img_file))

            RS_img = np.where( GT_img == 0, 0, EV_img)
            if not os.path.exists(os.path.join(RSdir, group)):
                os.makedirs(os.path.join(RSdir, group))
            imsave( os.path.join(RSdir, group, EV_img_file), RS_img)
            print(os.path.join(RSdir, group, EV_img_file))


    # Write data to HDF5
    with h5py.File(save_name, 'w') as data_file:
        for group in os.listdir(RSdir):
            if not os.path.isdir(os.path.join(RSdir, group)):
                continue
            h5_group = data_file.create_group(group)
            for plant in os.listdir(os.path.join(RSdir, group)):
                label = imread(os.path.join(RSdir, group, plant))
                if np.int(np.max(label))== 0:
                    print(plant)
                    continue
                b, e = os.path.splitext(plant)
                h5_plant = h5_group.create_group(b)
                h5_plant.create_dataset('label', dtype='uint8', data=label)
                fname = b + '_label' + e
                h5_plant.create_dataset('label_filename', data=fname)

    print(save_name)

if __name__ == "__main__":
    main(sys.argv[1])
