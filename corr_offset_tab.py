from chunk_histo import make_delay
import cPickle as pickle
from scipy.stats import pearsonr
from random import shuffle
import numpy as np

corr_result = '/home/zhiheng/lipmotion/3d_gan/corr_result.pkl'
p = pickle.load(open(corr_result))
lst = []
for i in p:
    lst.append((i, p[i]))

lst.sort(key=lambda x: x[1])

selected_vnames = [vname for vname, corr in lst]

vname_lms = pickle.load(open('/mnt/disk0/dat/zhiheng/lip_movements/grid_trend_lms.pkl'))
vname_flow = pickle.load(open('/home/zhiheng/lipmotion/3d_gan/of_result.pkl'))

result = {}
for vname in selected_vnames:
    result[vname] = []

for vname in selected_vnames:
    for delay in range(-15, 16):
        lms = vname_lms[vname]
        flow = vname_flow[vname]
        lms, flow = make_delay(lms, flow, delay)
        chunked_aud_of = [(lms[i: i+16], flow[i: i+16])
                          for i in range(0, len(lms), 16)
                            if i + 16 <= len(lms)]
        sum_corr = 0
        for a, f in chunked_aud_of:
            corr, _ = pearsonr(a, f)
            sum_corr += corr
        avg_corr = sum_corr / len(chunked_aud_of)
        # lms = [(l - min(lms)) / (max(lms) - min(lms)) for l in lms]
        # flow = [(f - min(flow)) / (max(flow) - min(flow)) for f in flow]
        result[vname].append((delay, avg_corr))
    result[vname].sort(key=lambda x: np.abs(x[1]), reverse=True)

pickle.dump(result, open('corr_offset_tab.pkl', 'wb+'))

selected_result = [(vname, result[vname]) for vname in result if result[vname][0] > 0 and result[vname][1] > 0]
