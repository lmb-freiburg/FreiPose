""" Reads thumbnails of a video on startup and gives them to you right away. After some time it shows you the full res frame. """
import cv2
from tqdm import tqdm
import time

from utils.dataset_util import read_vid_frame
from utils.VideoReaderFast import VideoReaderFast


class VideoThumbnailReader(object):
    def __init__(self, path, thumbnail_size=256, queue_size=128):
        self.path = path
        self.thumbnail_size = thumbnail_size
        self.queue_size = queue_size
        self.transform = lambda x: cv2.resize(x, (thumbnail_size, thumbnail_size))

        self.thumbs = self.read_thumbnails()

    def read_thumbnails(self):
        reader = VideoReaderFast(self.path, self.transform, self.queue_size)
        reader.start()
        time.sleep(1)

        self.vid_size = reader.get_size()

        pbar = tqdm(total=reader.get_size(), ncols=100)
        thumbs = list()
        while reader.more():
            thumbs.append(reader.read())
            pbar.update()
        return thumbs

    def get_thumb(self, tid):
        assert 0 <= tid < self.vid_size, 'Out of video range.'
        return self.thumbs[tid]

    def get_fs(self, tid):
        assert 0 <= tid < self.vid_size, 'Out of video range.'
        return read_vid_frame(self.path, tid)


if __name__ == '__main__':
    # reader = VideoThumbnailReader('/misc/lmbraid18/zimmermc/datasets/ExampleData/run00_cam1.mp4')
    # 17684 frames read in ~ 1min
    reader = VideoThumbnailReader('/misc/lmbraid18/zimmermc/datasets/RatTrack_set4/Peller_dispencer/run00_cam1.mp4')

    import time
    from utils.mpl_setup import plt_figure
    import numpy as np

    while True:
        i = np.random.randint(17684)

        plt, fig, axes = plt_figure(2)
        s = time.time()
        axes[0].imshow(reader.get_thumb(i))
        print('Time for thumb %.3f' % (time.time() - s))
        s = time.time()
        axes[1].imshow(reader.get_fs(i))
        print('Time for thumb %.3f' % (time.time() - s))
        plt.show()