import numpy as np
import nibabel as nib
import os


def shuffle(*parameters):
    first_arr = parameters[0]
    indexs = range(len(first_arr))
    np.random.shuffle(indexs)
    res = []
    for i in range(len(parameters)):
        res.append(np.array(parameters[i])[indexs])
    return res


def read_nii_file(path, show=False):
    img = nib.load(path)
    if show:
        print np.shape(img)
    return img.get_data()


def save_nii_file(path, image):
    img = nib.Nifti1Image(image, np.eye(4))
    nib.save(img, path)


def convert2batchfirst(arr):
    shape = list(np.shape(arr))
    result = np.zeros([shape[-1], shape[0], shape[1], shape[2]], dtype=np.float32)
    for i in range(shape[-1]):
        result[i] = arr[:, :, :, i]
    return result

if __name__ == '__main__':
    # print shuffle(np.array([0, 1, 2]), np.array([0, 1, 2]))
    # image = read_nii_file('/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 2/volume-28.nii')
    # save_nii_file('./test.nii', image)
    imgs = []
    path = '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 2'
    for name in os.listdir(path):
        imgs.append(read_nii_file(os.path.join(path, name), show=True))
    path = '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 1'
    for name in os.listdir(path):
        imgs.append(read_nii_file(os.path.join(path, name), show=True))