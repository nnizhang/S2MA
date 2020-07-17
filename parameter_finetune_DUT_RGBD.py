
# train.py
gpu_id = "8"
img_size = 256
scale_size = 288
batch_size = 8
lr = 0.001
epochs = 200
train_steps = 40000
lr_decay_gamma = 0.1
stepvalue1 = 20000
stepvalue2 = 30000
loss_weights = [1, 0.8, 0.8, 0.5, 0.5, 0.5]
bn_momentum = 0.001

load_model = 'models/S2MA.pth'

train_dir_img = 'list/train/DUT_train_list.txt'
save_lossdir = './model_epoch_loss/loss.txt'
save_model_dir = './models/'


# test.py

# test_lists = ['list/test/NLRP_test_list.txt', 'list/test/NJU2K_test_list.txt',
#               'list/test/STERE_test_list.txt', 'list/test/SSD_test_list.txt',
#               'list/test/RGBD135_test_list.txt', 'list/test/LFSD_test_list.txt']
# test_model = 'S2MA.pth'


# test on DUT-RGBD dataset
test_lists = ['list/test/DUT_RGBD_test_list.txt']
test_model = 'S2MA_DUT.pth'

test_model_dir = save_model_dir + test_model
save_test_path_root = './output/'


