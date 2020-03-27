import os


# dataset_dir = 'your rgb-d dataset path'
dataset_dir = '/data/zhangni/Data/RGB-D_Saliency/RGBdDataset_processed'


# train
NJU2K_train = True
NLPR_train = True
DUT_RGBD_train = True

# test
NJU2K_test = True
NLPR_test = True
DUT_RGBD_test = True
RGBD135 = True
LFSD = True
SSD = True
STERE = True


if NJU2K_train:
    root = dataset_dir + '/NJU2K/trainset'
    imgs = os.listdir(os.path.join(root, 'RGB'))

    for img in imgs:
        f = open('list/train/NJU2K_NLRP_train_list.txt', 'a')
        f.write(root + '/RGB/' + img + ' ' + root + '/depth/' + img.replace('.jpg', '.bmp') + ' ' + root
                + '/GT/' + img.replace('.jpg', '.png') + '\n')

if NLPR_train:
    root = dataset_dir + '/NLPR/trainset'
    imgs = os.listdir(os.path.join(root, 'RGB'))

    for img in imgs:
        f = open('list/train/NJU2K_NLRP_train_list.txt', 'a')
        f.write(root + '/RGB/' + img + ' ' + root + '/depth/' + img.replace('.jpg', '.bmp') + ' ' + root
                + '/GT/' + img.replace('.jpg', '.png') + '\n')


if DUT_RGBD_train:
    root = dataset_dir + '/DUT-RGBD/trainset'
    imgs = os.listdir(os.path.join(root, 'RGB'))

    for img in imgs:
        f = open('list/train/DUT_train_list.txt', 'a')
        f.write(root + '/RGB/' + img + ' ' + root + '/depth/' + img.replace('.jpg', '.png') + ' ' + root
                + '/GT/' + img.replace('.jpg', '.png') + '\n')


if NJU2K_test:
    root = dataset_dir + '/NJU2K/testset'
    imgs = os.listdir(os.path.join(root, 'RGB'))

    for img in imgs:
        f = open('list/test/NJU2K_test_list.txt', 'a')
        f.write(root + '/RGB/' + img + ' ' + root + '/depth/' + img.replace('.jpg', '.bmp') + ' ' + root
                + '/GT/' + img.replace('.jpg', '.png') + '\n')

if NLPR_test:
    root = dataset_dir + '/NLPR/testset'
    imgs = os.listdir(os.path.join(root, 'RGB'))

    for img in imgs:
        f = open('list/test/NLRP_test_list.txt', 'a')
        f.write(root + '/RGB/' + img + ' ' + root + '/depth/' + img.replace('.jpg', '.bmp') + ' ' + root
                + '/GT/' + img.replace('.jpg', '.png') + '\n')


if DUT_RGBD_test:
    root = dataset_dir + '/DUT-RGBD/testset'
    imgs = os.listdir(os.path.join(root, 'RGB'))

    for img in imgs:
        f = open('list/test/DUT_RGBD_test_list.txt', 'a')
        f.write(root + '/RGB/' + img + ' ' + root + '/depth/' + img.replace('.jpg', '.png') + ' ' + root
                + '/GT/' + img.replace('.jpg', '.png') + '\n')


if RGBD135:
    root = dataset_dir + '/RGBD135'
    imgs = os.listdir(os.path.join(root, 'RGB'))

    for img in imgs:
        f = open('list/test/RGBD135_test_list.txt', 'a')
        f.write(root + '/RGB/' + img + ' ' + root + '/depth/' + img.replace('.jpg', '.bmp') + ' ' + root
                + '/GT/' + img.replace('.jpg', '.png') + '\n')

if LFSD:
    root = dataset_dir + '/LFSD'
    imgs = os.listdir(os.path.join(root, 'RGB'))

    for img in imgs:
        f = open('list/test/LFSD_test_list.txt', 'a')
        f.write(root + '/RGB/' + img + ' ' + root + '/depth/' + img.replace('.jpg', '.bmp') + ' ' + root
                + '/GT/' + img.replace('.jpg', '.png') + '\n')

if SSD:
    root = dataset_dir + '/SSD100'
    imgs = os.listdir(os.path.join(root, 'RGB'))

    for img in imgs:
        f = open('list/test/SSD_test_list.txt', 'a')
        f.write(root + '/RGB/' + img + ' ' + root + '/depth/' + img.replace('.jpg', '.bmp') + ' ' + root
                + '/GT/' + img.replace('.jpg', '.png') + '\n')

if STERE:
    root = dataset_dir + '/STERE'
    imgs = os.listdir(os.path.join(root, 'GT'))

    for img in imgs:
        f = open('list/test/STERE_test_list.txt', 'a')
        f.write(root + '/RGB/' + img.replace('.png', '.jpg') + ' ' + root + '/depth/' + img + ' ' + root
                + '/GT/' + img + '\n')