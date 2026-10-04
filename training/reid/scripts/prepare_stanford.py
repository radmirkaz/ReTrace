import scipy.io
import os
import tqdm

DATASET_ROOT_DIR = "../datasets/stanford"


def prepare_datasets(root_dir, split):

    res_path = os.path.join(root_dir, f"cars_{split}_annos.csv")

    data = scipy.io.loadmat(os.path.join(root_dir, f"cars_{split}_annos.mat"))["annotations"][0]

    f = open(res_path, "w")
    f.write("img_path,x_1,y_1,x_2,y_2,class\n")

    for d in tqdm.tqdm(data):
        coords = [c[0][0] for c in [d[0], d[1], d[2], d[3]]]
        classid = d[4][0][0]
        filename = "val/" + d[5][0]
        f.write("%s,%s,%s,%s,%s,%s\n" % (filename, *coords, classid))
    f.close()


prepare_datasets(os.path.join(DATASET_ROOT_DIR, "car_devkit"), split="train")
