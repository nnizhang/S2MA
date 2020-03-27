# -*-coding:utf-8-*-
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from dataset import get_loader
import transforms as trans
from torchvision import transforms
import math
import time
from parameter import *
from ImageDepthNet import ImageDepthNet

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
cudnn.benchmark = True


def test_net(net):

    for test_dir_img in test_lists:

        test_loader = get_loader(test_dir_img, img_size, 1, mode='test', num_thread=1)

        print('''
                   Starting testing:
                       dataset: {}
                       Testing size: {}
                   '''.format(test_dir_img.split('/')[-1], len(test_loader.dataset)))

        for i, data_batch in enumerate(test_loader):
            print('{}/{}'.format(i, len(test_loader.dataset)))
            images, depths, image_w, image_h, image_path = data_batch
            images, depths = Variable(images.cuda()), Variable(depths.cuda())

            outputs_image, outputs_depth = net(images, depths)
            _, _, _, _, _, imageBran_output = outputs_image
            _, _, _, _, _, depthBran_output = outputs_depth

            image_w, image_h = int(image_w[0]), int(image_h[0])

            output_imageBran = F.sigmoid(imageBran_output)
            output_depthBran = F.sigmoid(depthBran_output)

            output_imageBran = output_imageBran.data.cpu().squeeze(0)
            output_depthBran = output_depthBran.data.cpu().squeeze(0)

            transform = trans.Compose([
                transforms.ToPILImage(),
                trans.Scale((image_w, image_h))
            ])
            outputImageBranch = transform(output_imageBran)
            outputDepthBranch = transform(output_depthBran)

            dataset = image_path[0].split('RGBdDataset_processed')[1].split('/')[1]

            filename = image_path[0].split('/')[-1].split('.')[0]

            # save image branch output
            save_test_path = save_test_path_root + dataset + '/' + test_model + '/'
            if not os.path.exists(save_test_path):
                os.makedirs(save_test_path)
            outputImageBranch.save(os.path.join(save_test_path, filename + '.png'))


if __name__ == '__main__':

    start = time.time()

    net = ImageDepthNet(3)
    net.cuda()
    net.eval()
    # load model
    net.load_state_dict(torch.load(test_model_dir))
    print('Model loaded from {}'.format(test_model_dir))

    test_net(net)
    print('total time {}'.format(time.time()-start))