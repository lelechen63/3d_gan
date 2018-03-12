from __future__ import print_function
import cPickle as pickle
from multiprocessing import Pool
from scipy.stats import pearsonr
import numpy as np
from chunk_histo import make_delay


def worker(data):
    vname, audio_deri, flow = data
    if len(audio_deri) != len(flow):
        return vname, None

    max_corr = 0
    max_corr_delay = None
    for delay in range(-15, 16):
        au, fl = make_delay(audio_deri, flow, delay)
        chunked_aud_of = [(au[i: i+16], fl[i: i+16])
                          for i in range(0, len(au), 16)
                            if i + 16 <= len(au)]
        sum_corr = 0
        for a, f in chunked_aud_of:
            corr, _ = pearsonr(a, f)
            sum_corr += corr
        avg_corr = sum_corr / len(chunked_aud_of)
        if avg_corr > max_corr:
            max_corr = avg_corr
            max_corr_delay = delay

    print(vname)
    return vname, max_corr, max_corr_delay


if __name__ == '__main__':
    audio_deri_f_path = '/mnt/disk0/dat/zhiheng/lip_movements/grid_trend_lms.pkl'
    flow_f_path = '/home/zhiheng/lipmotion/3d_gan/of_result.pkl'
    output_file = '/home/zhiheng/lipmotion/3d_gan/best_delay_result.pkl'

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
        if not flow_dict.has_key(video_name):
            continue

        flows = flow_dict[video_name]
        input_lst.append((video_name, audio_deri, flows))

    print('input dict length: {}'.format(len(input_lst)))

    vname_corr = pool.map(worker, input_lst)
    vname_corr = [(vname, corr, delay) for (vname, corr, delay) in vname_corr if delay is not None]
    pickle.dump(vname_corr, open(output_file, 'wb+'))
