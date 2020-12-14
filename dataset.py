from skimage import io
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
import h5py
import random
import torch
import os

class FaceData(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.face_info = self.get_dict()
        self.transform = transform
        self.data_dir1 = '/content/text2facegan'

    def __getitem__(self, index):
        img_file = self.face_info['image_list'][index]
        caption_vector = torch.FloatTensor(self.face_info['captions'][img_file][0])

        img = Image.open(os.path.join(self.data_dir1, 'img_align_celeba', img_file))
        if self.transform is not None:
            img = self.transform(img)

        return img, caption_vector

    def __len__(self):
        return self.face_info['data_length']


    def get_dict(self):
        print ("Reading .hdf5 file ...")
        data_path = os.path.join(self.data_dir, 'faces_2k.hdf5')
        data_path = '/content/faces_5k.hdf5'
        h = h5py.File(data_path, 'r')
        face_captions = {}
        for key in h.keys():
            if h[key].shape[0] == 0:
                continue
            face_captions[key] = h[key]

        training_image_list = [key for key in face_captions]
        random.shuffle(training_image_list)
        return {
            'image_list' : training_image_list,
            'captions' : face_captions,
            'data_length' : len(training_image_list)
        }