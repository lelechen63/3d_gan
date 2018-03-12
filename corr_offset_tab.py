from chunk_histo import make_delay
import cPickle as pickle
from scipy.stats import pearsonr
from random import shuffle

vnames = ['bbwr6p', 'bgie8n', 'bbifza', 'bbbk9p', 'bbaq7s', 'bgbs5s', 'bbaq1a', 'bbwd2a', 'bgbz3p', 'bgbg7n', 'bgak2n', 'bbwj4p', 'bbbd8s', 'bgay4n', 'bbiu8n', 'bgaf4s', 'bbik4n', 'bbwv9s', 'bbaj3s', 'bbiv3n', 'bbij9a', 'bgaf1a', 'bgbz1n', 'bgam1p', 'bbax2s', 'bbic1a', 'bbwgzp', 'bgig6a', 'bgbm7s', 'bbap8a', 'bbij5s', 'bgbfzp', 'bgar6p', 'bbwp8s', 'bbac9a', 'bbbp9s', 'bgam4n', 'bbad8p', 'bbwy7a', 'bbac7s', 'bgid7n', 'bbad7p', 'bgbz9n', 'bgbe9s', 'bgbn6s', 'bbbk7s', 'bbwf4p', 'bgba7s', 'bgaa7s', 'bgar3n', 'bgaa3p', 'bbaezp', 'bbbz8s', 'bbbe4s', 'bbie6s', 'bbae6n', 'bgah2p', 'bbwm1n', 'bbik1p', 'bgan4s', 'bbbx1n', 'bbac3n', 'bbap3p', 'bgbn4a', 'bbie8a', 'bbwz8s', 'bbbd4p', 'bbiq4s', 'bbbp2p', 'bbii2s', 'bgbs5p', 'bgba4p', 'bbaz5p', 'bgaf4n', 'bbbk1p', 'bgbtza', 'bbwc8p', 'bgbs6s', 'bgak4s', 'bbayzs', 'bbwk1s', 'bbwr2s', 'bbap2s', 'bbayzp', 'bbix3p', 'bgbf9p', 'bgbe9n', 'bgbs6n', 'bgal7n', 'bgbg2s', 'bgbe6p', 'bbwj3s', 'bbby8p', 'bbaz1a', 'bgbh3a', 'bgbf1a', 'bbil2a', 'bgbg1s', 'bbbyzn', 'bbix5p']
shuffle(vnames)
selected_vnames = vnames[:4]


vname_lms = pickle.load(open('/mnt/disk0/dat/zhiheng/lip_movements/grid_trend_lms.pkl'))
vname_flow = pickle.load(open('/home/zhiheng/lipmotion/3d_gan/of_result.pkl'))

result = {}
for vname in selected_vnames:
    result[vname] = []

for delay in range(-15, 1):
    for vname in selected_vnames:
        lms = vname_lms[vname]
        flow = vname_flow[vname]
        lms, flow = make_delay(lms, flow, delay)
        lms = [(l - min(lms)) / (max(lms) - min(lms)) for l in lms]
        flow = [(f - min(flow)) / (max(flow) - min(flow)) for f in flow]
        corr, _ = pearsonr(lms, flow)
        result[vname].append((delay, corr))

pickle.dump(result, open('corr_offset_tab.pkl', 'wb+'))
