from scipy.stats import pearsonr
import numpy as np
import multiprocessing
import json
import cPickle as pickle

audio = None

def make_delay(audio_deris, optical_flows, delay):
    assert len(audio_deris) == len(optical_flows)
    '''
    when delay is positive, move optical_flows forward
    when delay is negtive, move audio_deris forward
    '''

    if delay > 0:
        return audio_deris[:-delay], optical_flows[delay:]
    elif delay < 0:
        return audio_deris[-delay:], optical_flows[:delay]
    else:
        return audio_deris, optical_flows


def worker(info):
    global audio
    video_name, optical_flows = info
    if not audio.has_key(video_name):
        return None
    audio_deris = audio[video_name]
    x, y = make_delay(audio_deris, optical_flows, delay)
    corr = pearsonr(x, y)[0]
    print('video name: {}, delay: {}, corr: {}'.format(video_name, delay, corr))
    return video_name, corr

def main():
    global audio
    audio_path = '/mnt/disk1/dat/lchen63/grid/data/trend_lms.pkl'
    visual_path = '/home/zhiheng/lipmotion/gen_flow_distr/of_result.pkl'

    pool = multiprocessing.Pool(24)

    with open(audio_path) as audio_f:
        with open(visual_path) as visual_f:
            audio = pickle.load(audio_f)
            visual = pickle.load(visual_f)
            delay_to_videos = {}
            for delay in range(-10, 11):
                video_name_corr_lst = pool.map(worker, visual)
                video_to_corr = {video_name: corr for video_name,
                                 corr in video_name_corr_lst if corr is not None}
                delay_to_videos[delay] = video_to_corr

            with open('/home/zhiheng/lipmotion/corr_delay.pkl', 'wb') as corr_f:
                pickle.dump(delay_to_videos, corr_f)
                print('result saved to /home/zhiheng/lipmotion/corr_delay.pkl')

            stats = []
            for delay in delay_to_videos:
                stat = {}
                print('delay: {}'.format(delay))
                video_to_corr = delay_to_videos[delay]
                mean = np.mean(video_to_corr.values())
                std = np.std(video_to_corr.values())
                max_corr = np.max(video_to_corr.values())
                min_corr = np.min(video_to_corr.values())
                stat['delay'] = delay
                stat['mean'] = mean
                stat['std'] = std
                stat['max'] = max_corr
                stat['min'] = min_corr
                stats.append(stat)

            stats.sort(key=lambda stat: stat['mean'])
            with open('delay_stats.json', 'wb+') as f:
                json.dump(stats, f, indent=4)

if __name__ == '__main__':
    main()
