import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from dataset import get_loader
import math
from parameter_finetune_DUT_RGBD import *

from ImageDepthNet import ImageDepthNet
import os

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
cudnn.benchmark = True


def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    fh = open(save_dir, 'a')
    epoch_total_loss = str(epoch_total_loss)
    epoch_loss = str(epoch_loss)
    fh.write('until_' + str(epoch) + '_run_iter_num' + str(whole_iter_num) + '\n')
    fh.write(str(epoch) + '_epoch_total_loss' + epoch_total_loss + '\n')
    fh.write(str(epoch) + '_epoch_loss' + epoch_loss + '\n')
    fh.write('\n')
    fh.close()


def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer


def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('encode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('decode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('\n')
    fh.close()


def train_net(net):

    train_loader = get_loader(train_dir_img, img_size, batch_size, mode='train',
                              num_thread=4)

    print('''
        Starting training:
            Train steps: {}
            Batch size: {}
            Learning rate: {}
            Training size: {}
        '''.format(train_steps, batch_size, lr, len(train_loader.dataset)))

    N_train = len(train_loader) * batch_size

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    criterion = nn.BCEWithLogitsLoss()
    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / batch_size)
    for epoch in range(epochs):

        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        print('epoch:{0}-------lr:{1}'.format(epoch + 1, lr))

        epoch_total_loss = 0
        epoch_loss = 0

        for i, data_batch in enumerate(train_loader):
            if (i + 1) > iter_num: break
            images, depths, label_256, label_32, label_64, label_128, filename = data_batch
            images, depths, label_256 = Variable(images.cuda()), Variable(depths.cuda()), Variable(label_256.cuda())
            label_32, label_64, label_128 = Variable(label_32.cuda()), Variable(label_64.cuda()), \
                                            Variable(label_128.cuda())

            outputs_image, outputs_depth = net(images, depths)
            for_loss6, for_loss5, for_loss4, for_loss3, for_loss2, for_loss1 = outputs_image
            depth_for_loss6, depth_for_loss5, depth_for_loss4, depth_for_loss3, depth_for_loss2, depth_for_loss1 = outputs_depth

            # loss
            loss6 = criterion(for_loss6, label_32)
            loss5 = criterion(for_loss5, label_32)
            loss4 = criterion(for_loss4, label_32)
            loss3 = criterion(for_loss3, label_64)
            loss2 = criterion(for_loss2, label_128)
            loss1 = criterion(for_loss1, label_256)

            img_total_loss = loss_weights[0] * loss1 + loss_weights[1] * loss2 + loss_weights[2] * loss3\
                         + loss_weights[3] * loss4 + loss_weights[4] * loss5 + loss_weights[5] * loss6

            # depth loss

            depth_loss6 = criterion(depth_for_loss6, label_32)
            depth_loss5 = criterion(depth_for_loss5, label_32)
            depth_loss4 = criterion(depth_for_loss4, label_32)
            depth_loss3 = criterion(depth_for_loss3, label_64)
            depth_loss2 = criterion(depth_for_loss2, label_128)
            depth_loss1 = criterion(depth_for_loss1, label_256)

            depth_total_loss = loss_weights[0] * depth_loss1 + loss_weights[1] * depth_loss2 + loss_weights[2] * depth_loss3\
                         + loss_weights[3] * depth_loss4 + loss_weights[4] * depth_loss5 + loss_weights[5] * depth_loss6

            total_loss = img_total_loss + depth_total_loss

            epoch_total_loss += total_loss.cpu().data.item()
            epoch_loss += loss1.cpu().data.item()

            print('whole_iter_num: {0} --- {1:.4f} --- total_loss: {2:.6f} --- loss: {3:.6f}'.format((whole_iter_num + 1),
                                                     (i + 1) * batch_size / N_train, total_loss.item(), loss1.item()))

            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()
            whole_iter_num += 1

            if whole_iter_num == train_steps:
                torch.save(net.state_dict(),
                           save_model_dir + 'iterations{}.pth'.format(train_steps))
                return

            if whole_iter_num == stepvalue1 or whole_iter_num == stepvalue2:
                optimizer = adjust_learning_rate(optimizer, decay_rate=lr_decay_gamma)
                save_lr(save_lossdir, optimizer)
                print('have updated lr!!')

        print('Epoch finished ! Loss: {}'.format(epoch_total_loss / iter_num))

        save_loss(save_lossdir, whole_iter_num, epoch_total_loss / iter_num, epoch_loss/iter_num, epoch+1)
        torch.save(net.state_dict(),
                   save_model_dir + 'MODEL_EPOCH{}.pth'.format(epoch + 1))
        print('Saved')


if __name__ == '__main__':

    net = ImageDepthNet(3)

    net.load_state_dict(torch.load(load_model))
    print('load model:', load_model)

    net.train()
    net.cuda()

    train_net(net)

