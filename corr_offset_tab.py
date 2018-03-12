from chunk_histo import make_delay
import cPickle as pickle
from scipy.stats import pearsonr

lst = [('bbwr6p', 0.3954638720888051), ('bgie8n', 0.34279523293199615), ('bbifza', 0.32421508975559526), ('bbbk9p', 0.30602610295239646), ('bbaq7s', 0.30533454202306354)]

vname_lms = pickle.load(open('/mnt/disk0/dat/zhiheng/lip_movements/grid_trend_lms.pkl'))
vname_flow = pickle.load(open('/home/zhiheng/lipmotion/3d_gan/of_result.pkl'))

selected_vnames = [e[0] for e in lst]
result = {}
for vname in selected_vnames:
    result[vname] = []

for delay in range(-16, 0):
    for vname in selected_vnames:
        lms = vname_lms[vname]
        flow = vname_flow[vname]
        lms, flow = make_delay(lms, flow, delay)
        lms = [(l - min(lms)) / (max(lms) - min(lms)) for l in lms]
        flow = [(f - min(flow)) / (max(flow) - min(flow)) for f in flow]
        corr, _ = pearsonr(lms, flow)
        result[vname].append((delay, corr))

pickle.dump(open('corr_offset_tab.pkl', 'wb+'))
