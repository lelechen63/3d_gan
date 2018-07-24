import os
import pickle
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import glob
import torch
import torchvision
from torch.autograd import Variable
from dataset import VaganDataset
import scipy.misc
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from collections import OrderedDict
import multiprocessing
import math
from scipy.ndimage import gaussian_filter

from numpy.lib.stride_tricks import as_strided as ast
from skimage.measure import compare_ssim as ssim_f
from PIL import Image
import numpy as np
from skimage import feature
import math
from scipy.ndimage import correlate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/mnt/disk1/dat/lchen63/grid/data/pickle/")
    parser.add_argument("--model_dir",
                        type=str,
                        default="/mnt/disk1/dat/lchen63/grid/model/model2/model1_stage1_generator_7.pth")
    parser.add_argument("--sample_dir",
                        type=str,
                        default="/mnt/disk1/dat/lchen63/grid/test_result/model2/")
    parser.add_argument("--batch_size",
                        type=int,
                        default=8)
    parser.add_argument("--num_thread",
                        type=int,
                        default=40)
    return parser.parse_args()

##################CPDB##############

def block_process(A, block):
    block_contrast = np.zeros((A.shape[0]/block[0], A.shape[1]/block[1]), dtype=np.int32)
    flatten_contrast = list()
    for i in range(0, A.shape[0], block[0]):
        for j in range(0, A.shape[1], block[1]):
            block_view = A[i:i+block[0], j:j+block[1]]
            block_view = np.max(block_view) - np.min(block_view)
            flatten_contrast.append(block_view)
    block_contrast = np.array(flatten_contrast).reshape(block_contrast.shape)
    return block_contrast


def cpbd_compute(image):
    if isinstance(image, str):
        image = Image.open(image)
        image = image.convert('L')
    img = np.array(image, dtype=np.float32)
    m, n = img.shape

    threshold = 0.002
    beta = 3.6
    rb = 64
    rc = 64
    max_blk_row_idx = int(m/rb)
    max_blk_col_idx = int(n/rc)
    widthjnb = np.array([np.append(5 * np.ones((1, 51)), 3*np.ones((1, 205)))])
    total_num_edges = 0
    hist_pblur = np.zeros(101, dtype=np.float64)
    input_image_canny_edge = feature.canny(img)

    input_image_sobel_edge = matlab_sobel_edge(img)
    width = marziliano_method(input_image_sobel_edge, img)
    # print width
    for i in range(1, max_blk_row_idx+1):
        for j in range(1, max_blk_col_idx+1):
            rows = slice(rb*(i-1), rb*i)
            cols = slice(rc*(j-1), rc*j)
            decision = get_edge_blk_decision(input_image_canny_edge[rows, cols], threshold)
            if decision == 1:
                local_width = width[rows, cols]
                local_width = local_width[np.nonzero(local_width)]
                blk_contrast = block_process(img[rows, cols], [rb, rc]) + 1
                blk_jnb = widthjnb[0, int(blk_contrast)-1]
                prob_blur_detection = 1 - math.e ** (-np.power(np.abs(np.true_divide(local_width, blk_jnb)), beta))
                for k in range(1, local_width.size+1):
                    temp_index = int(round(prob_blur_detection[k-1] * 100)) + 1
                    hist_pblur[temp_index-1] = hist_pblur[temp_index-1] + 1
                    total_num_edges = total_num_edges + 1
    if total_num_edges != 0:
        hist_pblur = hist_pblur / total_num_edges
    else:
        hist_pblur = np.zeros(hist_pblur.shape)
    sharpness_metric = np.sum(hist_pblur[0:63])
    return sharpness_metric


def marziliano_method(E, A):
    # print E
    edge_with_map = np.zeros(A.shape)
    gy, gx = np.gradient(A)
    M, N = A.shape
    angle_A = np.zeros(A.shape)
    for m in range(1, M+1):
        for n in range(1, N+1):
            if gx[m-1, n-1] != 0:
                angle_A[m-1, n-1] = math.atan2(gy[m-1,n-1], gx[m-1,n-1]) * (180/np.pi)
            if gx[m-1, n-1] == 0 and gy[m-1, n-1] == 0:
                angle_A[m-1, n-1] = 0
            if gx[m-1, n-1] == 0 and gy[m-1, n-1] == np.pi/2:
                angle_A[m-1, n-1] = 90
    if angle_A.size != 0:
        angle_Arnd = 45 * np.round(angle_A/45.0)
        # print angle_Arnd
        count = 0
        for m in range(2, M):
            for n in range(2, N):
                if E[m-1, n-1] == 1:
                    if angle_Arnd[m-1, n-1] == 180 or angle_Arnd[m-1, n-1] == -180:
                        count += 1
                        for k in range(0, 101):
                            posy1 = n-1-k
                            posy2 = n - 2 - k
                            if posy2 <= 0:
                                break
                            if A[m-1, posy2-1] - A[m-1, posy1-1] <= 0:
                                break
                        width_count_side1 = k + 1
                        for k in range(0, 101):
                            negy1 = n + 1 + k
                            negy2 = n + 2 + k
                            if negy2 > N:
                                break
                            if A[m-1, negy2-1] > A[m-1, negy1-1]:
                                break
                        width_count_side2 = k + 1
                        edge_with_map[m-1, n-1] = width_count_side1 + width_count_side2
                    elif angle_Arnd[m-1, n-1] == 0:
                        count += 1
                        for k in range(0, 101):
                            posy1 = n+1+k
                            posy2 = n + 2 + k
                            if posy2 > N:
                                break
                            # print m, posy2
                            if A[m-1, posy2-1] <= A[m-1, posy1-1]:
                                break
                        width_count_side1 = k + 1
                        for k in range(0, 101):
                            negy1 = n -1-k
                            negy2 = n -2 -k
                            if negy2 <=0:
                                break
                            if A[m-1, negy2-1] >= A[m-1, negy1-1]:
                                break
                        width_count_side2 = k + 1
                        edge_with_map[m-1, n-1] = width_count_side1 + width_count_side2
    return edge_with_map


def get_edge_blk_decision(im_in, T):
    m, n = im_in.shape
    L = m * n
    im_edge_pixels = np.sum(im_in)
    im_out = im_edge_pixels > (L * T)
    return im_out


def matlab_sobel_edge(img):
    mask = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 8.0
    bx = correlate(img, mask)
    b = bx*bx
    # print b
    b = b > 4.0
    return np.array(b, dtype=np.int)
################################################################################################


def _load(generator, directory):
    # paths = glob.glob(os.path.join(directory, "*.pth"))
    path = directory
    print generator
    print path

    state_dict = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    # load params
    generator.load_state_dict(new_state_dict)
    # print torch.load(path).keys()
    # gen_path = [path for path in paths if "generator" in path][0]
    # generator.load_state_dict(torch.load(path))
    
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in generator.parameters()])))
    print("Load pretrained [{}]".format(path))





# def _load(generator, directory):
#     # paths = glob.glob(os.path.join(directory, "*.pth"))
#     path = directory
#     # print torch.load(path).keys()
#     weight = torch.load(path)
#     for key in torch.load(path).keys():
#         if 'face' in key:
#             print 'ggg'
#             weight.pop(key, None)


    # generator.load_state_dict(weight)
    # print generator
    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in generator.parameters()])))
    # print("Load pretrained [{}]".format(path))



def _sample( config):
    
    dataset = VaganDataset(config.dataset_dir, output_shape=[64, 64], train=False)
    num_test = len(dataset)
    # num_test=100


    real_path = os.path.join(config.sample_dir,'image/real64')
    fake_path = os.path.join(config.sample_dir,'image/fake64')

    if not os.path.exists(config.sample_dir):
        os.mkdir(config.sample_dir)
    if not os.path.exists(os.path.join(config.sample_dir,'image')):
        os.mkdir(os.path.join(config.sample_dir,'image'))
    if not os.path.exists(os.path.join(config.sample_dir,'image/real64')):
        os.mkdir(os.path.join(config.sample_dir,'image/real64'))
    if not os.path.exists(os.path.join(config.sample_dir,'image/fake64')):
        os.mkdir(os.path.join(config.sample_dir,'image/fake64'))



    paths = []
   
    stage1_generator = Generator()
    _load(stage1_generator,config.model_dir)
    examples, ims, landmarks, embeds, captions = [], [], [],[],[]
    for idx in range(num_test):
        example,im, landmark, embed, caption = dataset[idx]
        examples.append(example)
        ims.append(im)
        embeds.append(embed)
        captions.append(caption)
        landmarks.append(landmark)
    examples = torch.stack(examples,0)
    ims    = torch.stack(ims, 0)
    embeds = torch.stack(embeds, 0)
    noise  = Variable(torch.randn(num_test,  100))

    if config.cuda:
        examples= Variable( examples).cuda()
        noise = noise.cuda()
        # landmarks = Variable( landmarks).cuda()
        embeds = Variable(embeds).cuda()
        stage1_generator = stage1_generator.cuda()
    else:
        examples =Variable(examples)
        embeds = Variable(embeds)
        landmarks = Variable(landmarks)
   
    batch_size = config.batch_size
    for i in range(num_test/batch_size):
        example = examples[i*batch_size:  i*batch_size + batch_size]
        _noise = noise[i * batch_size: i*batch_size + batch_size]
        embed = embeds[i * batch_size : i * batch_size +batch_size]

        print '---------------------' + str(i) + '/' + str(num_test/batch_size)
        fake_ims_stage1 = stage1_generator(example,_noise, embed)

        
        for inx in range(batch_size):
            real_ims = ims[inx + i * batch_size]
            fake_ims = fake_ims_stage1[inx]
            real_ims = real_ims.cpu().permute(1,2,3,0).numpy()
            fake_ims =fake_ims.data.cpu().permute(1,2,3,0).numpy()
           
            fff={}
            rp = []
            fp =[]
            for j in range(real_ims.shape[0]):
              
                real = real_ims[j]
                fake= fake_ims[j]
              
                temp = captions[ inx+ i  * batch_size][j].split('/')
                if not os.path.exists(os.path.join(fake_path,temp[-2])):
                    os.mkdir(os.path.join(fake_path,temp[-2]))
                if not os.path.exists(os.path.join(real_path,temp[-2])):
                    os.mkdir(os.path.join(real_path,temp[-2]))
                real_name = os.path.join(real_path,temp[-2]) + '/' + temp[-1][:-4] + '.jpg'
                fake_name = os.path.join(fake_path,temp[-2]) + '/' + temp[-1][:-4] + '.jpg'
                scipy.misc.imsave(real_name,real)
                scipy.misc.imsave(fake_name,fake)
                rp.append(real_name)
                fp.append(fake_name)
            fff["real_path"] = rp
            fff["fake_path"] = fp
            paths.append(fff)
        # print os.path.join(fake_path,temp[-2])
        real_im  = ims[ i * batch_size : i * batch_size + batch_size]
        fake_store = fake_ims_stage1.data.permute(0,2,1,3,4).contiguous().view(config.batch_size*16,3,64,64)
        torchvision.utils.save_image(fake_store, 
            "{}/fake_{}.png".format(os.path.join(fake_path,temp[-2]),i), nrow=16,normalize=True)
        real_store = real_im.permute(0,2,1,3,4).contiguous().view(config.batch_size*16,3,64,64)
        torchvision.utils.save_image(real_store,
            "{}/real_{}.png".format(os.path.join(real_path,temp[-2]),i), nrow=16,normalize=True)

        with open(os.path.join(config.sample_dir,'image/test_result.pkl'), 'wb') as handle:
            pickle.dump(paths, handle, protocol=pickle.HIGHEST_PROTOCOL)

   

def generating_landmark_lips(test_inf):
    image = cv2.imread(os.path.join(config.sample_dir,'bg.jpg'))
    image_real = image.copy()
    image_fake = image.copy()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('/mnt/disk1/dat/lchen63/grid/data/shape_predictor_68_face_landmarks.dat')

    for inx in range(len(test_inf)):
        real_paths = test_inf[inx]["real_path"]
        fake_paths = test_inf[inx]["fake_path"]
        # print len(real_paths)
        for i in range(len(real_paths)):
            rp = real_paths[i]
            # print rp
            fp = fake_paths[i]
            # print fp
            temp_r = rp.split('/')
            # temp_f = fp.split('/')
            if not os.path.exists( os.path.join(config.sample_dir,'landmark/real64/' + temp_r[-2])):
                os.mkdir( os.path.join(config.sample_dir,'landmark/real64/' + temp_r[-2]))
            if not os.path.exists(os.path.join(config.sample_dir,'landmark/fake64/' + temp_r[-2])):
                os.mkdir(os.path.join(config.sample_dir,'landmark/fake64/' + temp_r[-2]))
            lm_r = os.path.join(config.sample_dir,'landmark/real64/' + temp_r[-2] + '/' + temp_r[-1][:-4] + '.npy' )
            lm_f = os.path.join(config.sample_dir,'landmark/fake64/' + temp_r[-2] + '/' + temp_r[-1][:-4] + '.npy' )
            i_lm_r = os.path.join(config.sample_dir,'landmark/real64/' + temp_r[-2] + '/' + temp_r[-1][:-4] + '.jpg' )
            i_lm_f = os.path.join(config.sample_dir,'landmark/fake64/' + temp_r[-2] + '/' + temp_r[-1][:-4] + '.jpg' )
            real_mask = cv2.imread(rp)
            fake_mask = cv2.imread(fp)
            image_real[237:301,181:245,:] = real_mask

            image_fake[237:301,181:245,:] = fake_mask

            real_gray = cv2.cvtColor(image_real, cv2.COLOR_BGR2GRAY)
            real_rects = detector(real_gray, 1)
            fake_gray = cv2.cvtColor(image_fake,cv2.COLOR_BGR2GRAY)
            fake_rects = detector(fake_gray, 1)
            if real_rects is None or fake_rects is None:
                print '--------------------------------'
            for (i,rect) in enumerate(fake_rects):
                shape = predictor(fake_gray, rect)
                shape = face_utils.shape_to_np(shape)
                for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                    # print name
                    if name != 'mouth':
                        continue
                    clone = image_fake.copy()
                    for (x, y) in shape[i:j]:
                        cv2.circle(clone, (x, y), 1, (0, 255, 0), -1)
                    cv2.imwrite(i_lm_f, clone)

                    mouth_land = shape[i:j].copy()
                    original = np.sum(mouth_land,axis=0) / 20.0
                    # print (mouth_land)
                    mouth_land = mouth_land - original
                    np.save(lm_f,mouth_land)
            for (i,rect) in enumerate(real_rects):
                shape = predictor(real_gray, rect)
                shape = face_utils.shape_to_np(shape)
                for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                    # print name
                    if name != 'mouth':
                        continue
                    clone = image_real.copy()
                    for (x, y) in shape[i:j]:
                        cv2.circle(clone, (x, y), 1, (0, 255, 0), -1)
                    cv2.imwrite(i_lm_r, clone)
                    mouth_land = shape[i:j].copy()
                    original = np.sum(mouth_land,axis=0) / 20.0

                    # print (mouth_land)
                    mouth_land = mouth_land - original
                    np.save(lm_r,mouth_land) 
def generate_landmarks(pickle_path):
    num_thread = config.num_thread
    test_inf = pickle.load(open(pickle_path, "rb"))
    print test_inf[0]
    datas = []
    batch_size = len(test_inf)/num_thread
    temp = []

    if not os.path.exists( os.path.join(config.sample_dir,'landmark')):
        os.mkdir( os.path.join(config.sample_dir,'landmark'))
    if not os.path.exists( os.path.join(config.sample_dir,'landmark/real64')):
        os.mkdir( os.path.join(config.sample_dir,'landmark/real64'))
    if not os.path.exists( os.path.join(config.sample_dir,'landmark/fake64')):
        os.mkdir( os.path.join(config.sample_dir,'landmark/fake64'))
    for i,d in enumerate(test_inf):
        temp.append(d)
        if (i+1) % batch_size ==0:
            datas.append(temp)
            temp = []

    for i in range(num_thread):
        process = multiprocessing.Process(target = generating_landmark_lips,args = (datas[i],))
        process.start()

def compare_landmarks(path):
    fake_path = os.path.join(path + 'fake64')

    real_path = os.path.join(path + 'real64')
    # fakes = os.walk(fake_path)
    rps = []
    fps = []
    print fake_path
    for root, dirs, files in os.walk(fake_path):
        for name in files:
            # print name
            if name[-3:] == 'npy':
                rps.append(real_path + '/' + name.split('_')[0] + '/' + name)
                fps.append(fake_path + '/' + name.split('_')[0] + '/' + name)
    dis_txt = open(path + 'distance.txt','w')
    distances = []
    print len(rps)
    for inx in range(len(rps)):
        rp = np.load(rps[inx])
        fp = np.load(fps[inx])
        # print rp.shape
        # print fp.shape
        dis = (rp-fp)**2
        dis = np.sum(dis,axis=1)
        dis = np.sqrt(dis)
        # print dis
        dis = np.sum(dis,axis=0)
        distances.append(dis)
        dis_txt.write(rps[inx] + '\t' + str(dis) + '\n') 
    average_distance = sum(distances) / len(rps)
    print average_distance

def psnr_f(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def compare_ssim(pickle_path):
    test_inf = pickle.load(open(pickle_path, "rb"))
    print test_inf[0]
    dis_txt = open(config.sample_dir + 'ssim.txt','w')
    ssims = []
    psnrs =[]
    for i,d in enumerate(test_inf):
        fake_paths = d['fake_path']
        real_paths = d['real_path']
        for inx in range(len(fake_paths)):
            f_i = cv2.imread(fake_paths[inx])
            r_i = cv2.imread(real_paths[inx])
            f_i = cv2.cvtColor(f_i, cv2.COLOR_BGR2GRAY)
            r_i = cv2.cvtColor(r_i, cv2.COLOR_BGR2GRAY)

            ssim = ssim_f(f_i,r_i)
            psnr = psnr_f(f_i,r_i)


            psnrs.append(psnr)
            ssims.append(ssim)
            print "ssim: {:.4f},\t psnr: {:.4f}".format( ssim, psnr)


            dis_txt.write(fake_paths[inx] + '\t {:.4f} \t {:.4f}'.format( ssim, psnr) + '\n') 
    average_ssim = sum(ssims) / len(ssims)
    average_psnr = sum(psnrs) / len(psnrs)
    print "------ssim: {:.4f},\t psnr: {:.4f}".format( average_ssim, average_psnr)
    return average_psnr,average_ssim
def compare_cpdb(pickle_path):
    test_inf = pickle.load(open(pickle_path, "rb"))
    print test_inf[0]
    dis_txt = open(config.sample_dir + 'cpdb.txt','w')
    r_cpdb = []
    f_cpdb = []
    for i,d in enumerate(test_inf):
        fake_paths = d['fake_path']
        real_paths = d['real_path']
        for inx in range(len(fake_paths)):
            real_cpdb = cpbd_compute(real_paths[inx])
            fake_cpdb = cpbd_compute(fake_paths[inx])
            r_cpdb.append(real_cpdb)
            f_cpdb.append(fake_cpdb)
            print "real: {:.4f},\t fake: {:.4f}".format( real_cpdb, fake_cpdb)

            dis_txt.write(fake_paths[inx] + '\t real: {:.4f} \t fake: {:.4f}'.format( real_cpdb, fake_cpdb) + '\n') 
    average_r = sum(r_cpdb) / len(r_cpdb)
    average_f = sum(f_cpdb) / len(f_cpdb)
    print "Aeverage: \t real: {:.4f},\t fake: {:.4f}".format( average_r, average_f)

def main(config):
    # _sample( config)
    p = os.path.join( config.sample_dir , 'image/test_result.pkl')
    # a,b = compare_ssim(p)
    # generate_landmarks(p)
    # compare_landmarks(os.path.join(config.sample_dir ,'landmark/'))
    # print "-------ssim: {:.4f},\t psnr: {:.4f}".format( b, a)
    compare_cpdb(p)
if __name__ == "__main__":
    config = parse_args()
    from model_vgan import Generator  
    main(config)
