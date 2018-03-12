from scipy.stats import pearsonr
import math
import multiprocessing
import cPickle as pickle
import matplotlib.pyplot as plt
import os


figure_output_root = '/home/zhiheng/lipmotion/3d_gan/histo_figure_minmax/'


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


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


def in_worker(q_in, q_out):
    while True:
        info = q_in.get()
        if info is None:
            break

        video_name, optical_flows, audio_deris, delay = info
        assert len(optical_flows) == len(audio_deris)
        if len(audio_deris) - int(math.fabs(delay)) < 16:
            continue
        audio_deris, optical_flows = make_delay(audio_deris, optical_flows, delay)
        chunked_aud_of = [(audio_deris[i: i+16], optical_flows[i: i+16])
                          for i in range(0, len(audio_deris), 16)
                            if i + 16 <= len(audio_deris)]
        for aud, of in chunked_aud_of:
            aud = [(a - min(aud)) / (max(aud) - min(aud)) for a in aud]
            of = [(o - min(of)) / (max(of) - min(aud)) for o in of]
            corr, p_value = pearsonr(aud, of)
            print('video name: {}, delay: {}, corr: {}, p_value: {}'
                    .format(video_name, delay, corr, p_value))
            q_out.put((corr, delay))


def out_worker(q_out):
    delay_to_videos = {}
    while True:
        info = q_out.get()
        if info is None:
            for delay, values in delay_to_videos.iteritems():
                plt.hist(values, 50, normed=1, facecolor='green', alpha=0.75)
                plt.xlabel('correlation')
                plt.ylabel('count')
                plt.title('Delay: {}'.format(delay))
                plt.grid(True)
                figure_path = os.path.join(figure_output_root, 'delay_{}'.format(delay))
                plt.savefig(figure_path)
                print('figure saved to: {}'.format(figure_path))
                plt.gcf().clf()
            break

        corr, delay = info
        if delay not in delay_to_videos:
            delay_to_videos[delay] = []
        delay_to_videos[delay].append(corr)


def main():
    audio_path = '/mnt/disk0/dat/zhiheng/lip_movements/grid_trend_lms.pkl'
    visual_path = '/home/zhiheng/lipmotion/3d_gan/of_result.pkl'

    with open(audio_path) as audio_f, open(visual_path) as visual_f:
        audio = pickle.load(audio_f)
        visual = pickle.load(visual_f)

        q_in = [multiprocessing.Queue(1024) for _ in range(40)]
        q_out = multiprocessing.Queue(1024)
        read_process = [multiprocessing.Process(target=in_worker, args=(q_in[i], q_out))
                        for i in range(40)]
        for p in read_process:
            p.start()
        write_process = multiprocessing.Process(target=out_worker, args=(q_out,))
        write_process.start()

        for delay in range(-59, 60):
            visual_audio_pairs = [(video_name, ofs, audio[video_name], delay)
                                  for video_name, ofs in visual.iteritems() if audio.has_key(video_name)]
            for i, item in enumerate(visual_audio_pairs):
                q_in[i % len(q_in)].put(item)

        for q in q_in:
            q.put(None)
        for p in read_process:
            p.join()

        q_out.put(None)
        write_process.join()


if __name__ == '__main__':
    main()
