import argparse
import nltk
nltk.download('punkt')
import torch
from torchvision.utils import save_image
import os
import torchvision.utils as vutils
import numpy as np
import cv2

from dcgan import Generator
import skipthoughts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default='The woman has oval face and high cheekbones',
                       help='description of the people you want to synthesize.')
    parser.add_argument('--epoch_load', type=str, default='latest',
                        help='Model path of DcGAN Generator.')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='/content/drive/MyDrive/Deep_Learning/projects/dcgan/continue_train/checkpoints/',
                        help='Checkpoint directory.')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--t_in', type=int, default=4800)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--save_dir', type=str, default='results')

    args = parser.parse_args()
    args.text = "The woman has high cheekbones.She has straight hair which is brown in colour."
    print (args.text)
    ngpu = 1
    # 1. Encode text using skipthought model
    model = skipthoughts.load_model()
    text = [i for i in args.text.split('.') if i != '']
    text_embedding = torch.FloatTensor(skipthoughts.encode(model, text)[0]) # 1x4800 embedding vector
    print ("Done encoding text.")

    # 2. Gen faces using DcGAN
    '''
    input: concat[random gaussian noise (1x100), text_embedding (1x4800)]
    output: face image of size 64x64x3
    '''
    # 2.1 Define Generator
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    netG = Generator(1, args.nz + args.t_in, args.ngf, args.nc).to(device)
    model_name = 'netG_' + args.epoch_load + '.pth'
    netG.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, model_name)))
    print (f"Done loading {model_name}.")

    if torch.cuda.is_available():
      netG = netG.cuda()
    # 2.2 Create noise (gaussian)
    noise = torch.cat([torch.randn(1, args.nz, 1, 1), 
                       torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(text_embedding, 0), 2), 3)], 1)
    if torch.cuda.is_available():
      noise = noise.cuda()
    # 2.3 Generate from noise
    fake_img = netG(noise)
    print("Image generated.")

    # 2.4 Save generated image
    if not os.path.exists(args.save_dir):
      os.makedirs(args.save_dir)
    name_save = 'synthesized_face.png'
    path_save = os.path.join(args.save_dir, name_save)
    if torch.cuda.is_available():
      fake_img = vutils.make_grid(fake_img.detach().cpu(), padding=2, normalize=True)
    else:
      fake_img = fake_img.detach()
    img = (np.transpose(fake_img.numpy(),(1,2,0))*255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path_save, img)
    print("Done!")










