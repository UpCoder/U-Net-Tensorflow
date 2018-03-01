import os
from glob import glob
from util import shuffle, read_nii_file, save_nii_file
from Generator import GenerateBatch
import numpy as np


class Reader:
    def __init__(self, batch1_path, batch2_path, batch_size):
        self.batch1_path = batch1_path
        self.batch2_path = batch2_path
        self.img_paths_test, self.gt_paths_test = Reader.load_all_files(self.batch1_path)
        self.img_paths_batch2, self.gt_paths_batch2 = Reader.load_all_files(self.batch2_path)
        self.img_paths_train, self.img_paths_val, self.gt_paths_train, self.gt_paths_val = Reader.split2_train_val(
            self.img_paths_batch2, self.gt_paths_batch2)
        print len(self.img_paths_train), len(self.img_paths_val), len(self.img_paths_test)
        self.train_generator = GenerateBatch(self.img_paths_train, self.gt_paths_train, batch_size=batch_size,
                                             dataset_op=read_nii_file, label_op=read_nii_file).generate_next_batch()
        self.val_generator = GenerateBatch(self.img_paths_val, self.gt_paths_val, batch_size=batch_size,
                                           dataset_op=read_nii_file, label_op=read_nii_file).generate_next_batch()
        self.test_generator = GenerateBatch(self.img_paths_test, self.gt_paths_test, batch_size=batch_size,
                                            dataset_op=read_nii_file, label_op=read_nii_file).generate_next_batch()

    @staticmethod
    def load_all_files(path):
        image_paths = glob(os.path.join(path, 'volume-*.nii'))
        gt_paths = glob(os.path.join(path, 'segmentation-*.nii'))
        image_paths.sort()
        gt_paths.sort()
        return image_paths, gt_paths

    @staticmethod
    def split2_train_val(image_paths, gt_paths):
        [image_paths, gt_paths] = shuffle(image_paths, gt_paths)
        rate = 0.8
        boundary = int(len(image_paths) * rate)
        img_paths_train = image_paths[:boundary]
        img_paths_val = image_paths[boundary:]
        gt_paths_train = gt_paths[:boundary]
        gt_paths_val = gt_paths[boundary:]
        return img_paths_train, img_paths_val, gt_paths_train, gt_paths_val

    @staticmethod
    def extract_liver_slice(img_path, gt_path):
        img = read_nii_file(img_path)
        gt = read_nii_file(gt_path)
        img_slices = []
        gt_slices = []
        for z in range(np.shape(img)[2]):
            cur_slice_gt = gt[:, :, z]
            if np.sum(cur_slice_gt) != 0:
                img_slices.append(img[:, :, z])
                gt_slices.append(gt[:, :, z])
        return np.array(img_slices), np.array(gt_slices)

    @staticmethod
    def extract_save_liver_slice(img_path, gt_path, save_dir):
        img_name = os.path.basename(img_path).split('.')[0]
        gt_name = os.path.basename(gt_path).split('.')[0]
        img_slices, gt_slices = Reader.extract_liver_slice(img_path, gt_path)
        for index in range(len(img_slices)):
            print os.path.join(save_dir, img_name + '_' + str(index) + '.nii')
            print os.path.join(save_dir, gt_name + '_' + str(index) + '.nii')
            save_nii_file(os.path.join(save_dir, img_name + '_' + str(index) + '.nii'), img_slices[index, :, :])
            save_nii_file(os.path.join(save_dir, gt_name + '_' + str(index) + '.nii'), gt_slices[index, :, :])

    def extract_all_liver_slices(self, save_dir):
        for index in range(len(self.img_paths_test)):
            Reader.extract_save_liver_slice(self.img_paths_test[index], self.gt_paths_test[index],
                                            save_dir=os.path.join(save_dir, 'Train Batch1'))
        for index in range(len(self.img_paths_batch2)):
            Reader.extract_save_liver_slice(self.img_paths_batch2[index], self.gt_paths_batch2[index],
                                            save_dir=os.path.join(save_dir, 'Train Batch2'))
if __name__ == '__main__':
    reader = Reader('/home/give/Documents/dataset/ISBI2017/Slice/Train Batch1',
                    '/home/give/Documents/dataset/ISBI2017/Slice/Train Batch2',
                    batch_size=5)
    train_generator = reader.train_generator
    while True:
        img, gt = train_generator.next()
        print np.shape(img)
