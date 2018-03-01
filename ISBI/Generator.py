import numpy as np
from util import convert2batchfirst


class GenerateBatch:
    def __init__(self, dataset, label, batch_size, epoch_num=None, dataset_op = None, label_op = None):
        self.dataset = dataset
        self.label = label
        self.batch_size = batch_size
        self.start = 0
        self.epoch_num = epoch_num
        self.dataset_op = dataset_op
        self.label_op = label_op

    def generate_next_batch(self):
        if self.epoch_num is not None:
            for i in range(self.epoch_num):
                while self.start < len(self.dataset):
                    cur_image_batch = self.dataset[self.start: self.start + self.batch_size]
                    cur_label_batch = self.label[self.start: self.start + self.batch_size]
                    self.start += self.batch_size
                    if self.dataset_op is not None:
                        cur_image_batch = [np.expand_dims(np.expand_dims(self.dataset_op(cur_image), axis=2), axis=3) for cur_image in
                                           cur_image_batch]
                        cur_image_batch = np.concatenate(cur_image_batch, axis=3)
                    if self.label_op is not None:
                        cur_label_batch = [np.expand_dims(np.expand_dims(self.label_op(cur_label), axis=2), axis=3) for cur_label in
                                           cur_label_batch]
                        cur_label_batch = np.concatenate(cur_label_batch, axis=3)
                    cur_image_batch = np.asarray(cur_image_batch, np.float32) / 255.0
                    yield convert2batchfirst(cur_image_batch), convert2batchfirst(cur_label_batch)
        else:
            while True:
                cur_image_batch = self.dataset[self.start: self.start + self.batch_size]
                cur_label_batch = self.label[self.start: self.start + self.batch_size]
                self.start = (self.start + self.batch_size) % len(self.dataset)
                if self.dataset_op is not None:
                    cur_image_batch = [np.expand_dims(np.expand_dims(self.dataset_op(cur_image), axis=2), axis=3) for
                                       cur_image in
                                       cur_image_batch]
                    cur_image_batch = np.concatenate(cur_image_batch, axis=3)
                if self.label_op is not None:
                    cur_label_batch = [np.expand_dims(np.expand_dims(self.label_op(cur_label), axis=2), axis=3) for
                                       cur_label in
                                       cur_label_batch]
                    cur_label_batch = np.concatenate(cur_label_batch, axis=3)
                cur_image_batch = np.asarray(cur_image_batch, np.float32) / 255.0
                yield convert2batchfirst(cur_image_batch), convert2batchfirst(cur_label_batch)