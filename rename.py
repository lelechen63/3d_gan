import os

root = '/mnt/disk1/dat/lchen63/lrw/test_result/model_gan_r2/'
output = '/mnt/disk1/dat/lchen63/lrw/test_result/model_gan_r2/rename'
for i in range(0, 6):
    for j in range(0, 64):
        real_frm_name = 'real_{}_{}.png'.format(i, j)
        fake_frm_name = 'fake_{}_{}.png'.format(i, j)
        frm_idx_name = '{}.png'.format(i*64 + j)
        real_path = os.path.join(root, real_frm_name)
        fake_path = os.path.join(root, fake_frm_name)
        real_sym = os.path.join(output, real_frm_name)
        fake_sym = os.path.join(output, fake_frm_name)
        os.symlink(real_path, real_sym)
        os.symlink(fake_path, fake_sym)
