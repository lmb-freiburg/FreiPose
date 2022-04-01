"""
    Class that represents many images stitched together.
"""
import numpy as np
import cv2


class StitchedImage(object):
    def __init__(self, image_list, target_size=None, inpaint_id=False):
        assert len(image_list) > 0, 'There should be at least one image.'
        # turn into a list of numpy images
        self.images = list()
        for img in image_list:
            if type(img) == str:
                img = cv2.imread(img)[:, :, ::-1]
            else:
                img = np.copy(img)

            # normalize shape
            if len(img.shape) == 2:
                img = np.expand_dims(img, 2)
            self.images.append(img)

        self._subframe_img_id_map = dict()
        self._subframe_scales = dict()
        self._common_shape = None

        # create stitched image
        self.inpaint_id = inpaint_id
        self.image = self._stitch()

        if target_size is None:
            target_size = (800, 1000)

        if target_size == 'max':
            # show image at maximal resolution (no final downsampling)
            target_size = self.image.shape[:2]

        self._global_scale = np.array(target_size[:2], dtype=np.float32) / np.array(self.image.shape[:2], dtype=np.float32)
        self.image = cv2.resize(self.image, target_size[::-1])

    @staticmethod
    def _calc_stack_shape(num_samples):
        """ Calculates the size of a N, M grid so that num_samples can be accomodated within.
            When N, M correspond to rows, columns then it favors a landscapeish output.
        """
        M = int(np.ceil(np.sqrt(num_samples)))
        for i in range(1, M + 1):
            if i * M >= num_samples:
                N = i
                break

        return N, M

    @staticmethod
    def _find_common_size(img_list):
        sizes = list()
        for img in img_list:
            sizes.append(img.shape[:2])
        sizes = np.stack(sizes, 0)
        return np.min(sizes, 0)

    def _stitch(self):
        # work out final size
        stack_shape = self._calc_stack_shape(len(self.images))

        # work out image size (use the smallest shape)
        self._common_shape = self._find_common_size(self.images)

        # stack images
        i = 0
        img_list = list()
        for v_shape in range(stack_shape[0]):
            img_list_v = list()
            for h_shape in range(stack_shape[1]):
                if i == len(self.images):
                    # we are running out of images --> fill with dummy data
                    img_list_v.append(np.zeros((self._common_shape[0], self._common_shape[1], 3), dtype=np.uint8))
                    self._subframe_img_id_map[v_shape, h_shape] = -1
                    self._subframe_scales[v_shape, h_shape] = None
                    continue

                if self.inpaint_id:
                    cv2.putText(self.images[i], '%d' % i, org=(10, 75), fontFace=3, fontScale=2.5,
                                color=(0, 0, 255), thickness=3)

                self._subframe_img_id_map[v_shape, h_shape] = i
                self._subframe_scales[i] = np.array(self._common_shape[:2], dtype=np.float32) / np.array(self.images[i].shape[:2], dtype=np.float32)
                img = cv2.resize(self.images[i], tuple(self._common_shape[::-1]))
                img_list_v.append(img)
                i += 1

            img_list.append(np.concatenate(img_list_v, 1))
        image = np.concatenate(img_list, 0)
        return image

    def _get_subframe_id(self, u, v):
        """ Returns the subframe id for a given u, v location."""
        # coords in the unscaled image
        u, v = u / self._global_scale[1], v / self._global_scale[0]

        h_shape, v_shape = int(u // self._common_shape[1]), int(v // self._common_shape[0])

        return self._subframe_img_id_map[v_shape, h_shape]

    def _get_subframe_coords(self, u, v):
        """ Returns the coordinate within the subframe."""
        # coords in the unscaled image
        u, v = u / self._global_scale[1], v / self._global_scale[0]

        # offset due to other stacked images
        u_offset, v_offset = int(u // self._common_shape[1]) * self._common_shape[1], int(v // self._common_shape[0]) * self._common_shape[0]
        return u - u_offset, v - v_offset

    def map_stitch2orig(self, u, v):
        """ Given a coordinate in the stichted image returns the image id and subframe u, v. """
        # map into subframe
        u_sub, v_sub = self._get_subframe_coords(u, v)

        # get subframe id
        id = self._get_subframe_id(u, v)

        return (u_sub / self._subframe_scales[id][1], v_sub / self._subframe_scales[id][0]), id

    def map_orig2stitch(self, u, v, img_id, return_bounds=False):
        """ Given a coordinate in the original image space and its subframe id it returns the coordinate in the stitched image. """
        # make a range check
        u1 = min(max(0, u), self.images[img_id].shape[1]-1)
        v1 = min(max(0, v), self.images[img_id].shape[0]-1)

        in_bounds = abs(u1 - u) < 1.0 and abs(v1 - v) < 1.0
        u, v = u1, v1

        # scale down
        u_v = u * self._subframe_scales[img_id][1] * self._global_scale[1]
        v_v = v * self._subframe_scales[img_id][0] * self._global_scale[0]

        # find out offset wrt to the images location
        v_shape, h_shape = None, None
        for k, v in self._subframe_img_id_map.items():
            if v == img_id:
                v_shape, h_shape = k
                break

        # calculate view coords
        u_v = u_v + h_shape * self._common_shape[1] * self._global_scale[1]
        v_v = v_v + v_shape * self._common_shape[0] * self._global_scale[0]

        if return_bounds:
            return u_v, v_v, in_bounds

        else:
            return u_v, v_v


if __name__ == '__main__':
    img_list = ['/home/egg/czimmerm/datasets/FreiHAND/subject3/take01/run01/cam0/00000020.png',
                '/home/egg/czimmerm/datasets/FreiHAND/subject3/take01/run01/cam1/00000020.png',
                '/home/egg/czimmerm/datasets/FreiHAND/subject3/take01/run01/cam2/00000020.png',
                # '/home/egg/czimmerm/datasets/FreiHAND/subject3/take01/run01/cam3/00000020.png',
                '/home/egg/czimmerm/datasets/FreiHAND/subject3/take01/run01/cam4/00000020.png',
                '/home/egg/czimmerm/datasets/FreiHAND/subject3/take01/run01/cam5/00000020.png']

    stitched = StitchedImage(img_list)

    pt_cam1 = (430, 248)
    pt_cam2 = (428, 283)

    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.imshow(scipy.misc.imread(img_list[1]))
    ax.plot(pt_cam1[0], pt_cam1[1], "ro")
    plt.show(block=False)

    import matplotlib.pyplot as plt
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    ax.imshow(scipy.misc.imread(img_list[3]))
    ax.plot(pt_cam2[0], pt_cam2[1], "bo")
    plt.show(block=False)

    pt_cam1_st = stitched.map_orig2stitch(pt_cam1[0], pt_cam1[1], 1)
    pt_cam2_st = stitched.map_orig2stitch(pt_cam2[0], pt_cam2[1], 3)

    # vice versa
    pt_cam_st, id = stitched.map_stitch2orig(pt_cam1_st[0], pt_cam1_st[1])
    print(pt_cam_st, ' should be == to', pt_cam1)
    print(id, ' should be == to', 1)

    pt_cam_st, id = stitched.map_stitch2orig(pt_cam2_st[0], pt_cam2_st[1])
    print(pt_cam_st, ' should be == to', pt_cam2)
    print(id, ' should be == to', 3)

    fig = plt.figure(3)
    ax = fig.add_subplot(111)
    ax.imshow(stitched.image)
    ax.plot(pt_cam1_st[0], pt_cam1_st[1], "ro")
    ax.plot(pt_cam2_st[0], pt_cam2_st[1], "bo")
    plt.show()






