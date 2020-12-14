import argparse
import nltk
nltk.download('punkt')
import torch
from torchvision.utils import save_image
import torchvision.utils as vutils
import os
import cv2
import h5py
import numpy as np

from dcgan import Generator
import skipthoughts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default='The man has oval face.The man looks young.',)
    parser.add_argument('--epoch_load', type=str, default='latest',
                        help='Model path of DcGAN Generator.')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='/content/drive/MyDrive/Deep_Learning/projects/dcgan/continue_train/checkpoints/',
                        help='Checkpoint directory.')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--t_in', type=int, default=4800)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--save_dir', type=str, default='/content/drive/MyDrive/Deep_Learning/results/')

    args = parser.parse_args()
    args.text = 'The man has high cheekbones.He has straight hair which is brown in colour.The young man is smiling.'
    image_captions = {}
    h = h5py.File('/content/drive/MyDrive/Deep_Learning/projects/test_faces1000.hdf5', 'r')
    face_captions = {}
    for key in h.keys():
        if h[key].shape[0] == 0:
            continue
        face_captions[key] = h[key]
    print (len(face_captions))
    # 1. Encode text using skipthought model
    # model = skipthoughts.load_model()
    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    netG = Generator(1, args.nz + args.t_in, args.ngf, args.nc).to(device)
    model_name = 'netG_' + args.epoch_load + '.pth'
    print (model_name)
    netG.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, model_name)))
    print (f"Done loading {model_name}.")

    if torch.cuda.is_available():
      netG = netG.cuda()
    for key in face_captions.keys():
        print (key)
        text_embedding = torch.FloatTensor(face_captions[key][0])
        # text_embedding = torch.FloatTensor(skipthoughts.encode(model, text)[0]) # 1x4800 embedding vector

        # 2. Gen faces using DcGAN
        '''
        input: concat[random gaussian noise (1x100), text_embedding (1x4800)]
        output: face image of size 64x64x3
        '''
        # 2.1 Define Generator
        
        # 2.2 Create noise (gaussian)
        noise = torch.cat([torch.randn(1, args.nz, 1, 1), 
                           torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(text_embedding, 0), 2), 3)], 1)
        if torch.cuda.is_available():
          noise = noise.cuda()
        # 2.3 Generate from noise
        fake_img = netG(noise)

        # 2.4 Save generated image
        if not os.path.exists(args.save_dir):
          os.makedirs(args.save_dir)
        name_save = f'synthesized_face_{key}'
        path_save = os.path.join(args.save_dir, name_save)
        fake_img = vutils.make_grid(fake_img.detach().cpu(), padding=2, normalize=True)
        img = (np.transpose(fake_img.numpy(),(1,2,0))*255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path_save, img)
    print("Done!")










