from __future__ import print_function
import cPickle as pickle
from multiprocessing import Pool
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt


def worker(data):
    vname, audio_deri, flow = data
    if len(audio_deri) != len(flow):
        return vname, None
    corr, _ = pearsonr(audio_deri, flow)
    print(vname)
    return vname, corr


if __name__ == '__main__':
    audio_deri_f_path = '/mnt/disk1/dat/lchen63/lrw/data/pickle/trend_lms.pkl'
    flow_f_path = '/home/zhiheng/lipmotion/3d_gan/of_result.pkl'
    output_file = '/home/zhiheng/lipmotion/3d_gan/corr_result.pkl'
    figure_output_file = '/home/zhiheng/lipmotion/3d_gan/corr_histo.png'

    audio_deri_dict = pickle.load(open(audio_deri_f_path))
    flow_dict = dict(pickle.load(open(flow_f_path)))
    
    print('audio derivative dict length: {}'.format(len(audio_deri_dict)))
    print('flow dict length {}'.format(len(flow_dict)))

    pool = Pool(40)
    # input_lst = [(video_name, audio_deri_dict[video_name], flow_dict[video_name])  
    #                 for video_name in audio_deri_dict.keys()
    #                     if not np.any(np.isinf(audio_deri_dict[video_name]))]
    input_lst = []
    for video_name, audio_deri in audio_deri_dict.iteritems():
        if np.any(np.isinf(audio_deri)):
            continue
        if not not flow_dict.has_key(video_name):
            continue
        
        flows = flow_dict[video_name]
        input_lst.append((video_name, audio_deri, flows))
    
    print('input dict length: {}'.format(len(input_lst)))

    vname_corr = pool.map(worker, input_lst)
    vname_corr = [(vname, corr) for (vname, corr) in vname_corr if corr is not None]
    pickle.dump(vname_corr, open(output_file, 'wb+'))

    n, bins, patches = plt.hist(vname_corr.values(), 50, normed=1, facecolor='green', alpha=0.75)
    plt.xlabel('correlation')
    plt.ylabel('count')
    plt.title('Correlation Analysis')
    # plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    plt.savefig(figure_output_file)
    plt.close()
