from __future__ import print_function, unicode_literals
import os
import cv2
import tensorflow as tf
import numpy as np

import utils.CamLib as cl
from utils.triangulate.TriangTool import TriangTool, t_triangulation


def load_inference_graph(path_to_graph):
    """ Load a frozen inference graph into memory. """
    # load frozen tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')
    input_tensors = {'image_tensor': image_tensor}
    pred_tensors = {'detection_scores': detection_scores,
                    'detection_boxes': detection_boxes,
                    'detection_classes': detection_classes,
                    'num_detections': num_detections}
    return sess, input_tensors, pred_tensors


def draw_box_on_image(score_thresh, scores, boxes, image_np, max_num=2):
    c = (77, 255, 9)
    im_width, im_height = image_np.shape[1], image_np.shape[0]
    # box_list = list()
    boxes_drawn = 0
    image_np = image_np.copy()
    for i, s in enumerate(scores):
        if boxes_drawn >= max_num:
            break

        if s > score_thresh:
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            # box_list.append((p1, p2))
            cv2.rectangle(image_np, p1, p2, c, 3, 1)
            cv2.putText(image_np, '%.2f' % s, p1,
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.0, color=c, thickness=2)
            boxes_drawn += 1
    return image_np
    # return image_np, box_list


triang_tool = None
def post_process_detections(boxes, scores, K_list, M_list, img_shape,
                            min_score_cand=0.2, min_score_pick=0.1, max_reproj_error=10.0,
                            verbose=True, img=None, logger=None):
    """ Some post processing to increase the quality of bounding box detections.
        We consider all bounding boxes as candidate above some min_score_cand and calculate their 2D center position.
        Using all centers we aim to find a 3D hypothesis explaining as many centers as possible. Subsequently, we pick
        the boxes with minimal distance to the
    """
    global triang_tool
    if triang_tool is None:
        triang_tool = TriangTool()

    output = {
        'boxes': None,
        'xyz': None
    }

    # calculate bounding box centers
    box_centers = np.stack([0.5 * (boxes[:, :, 1] + boxes[:, :, 3]) * img_shape[1],
                            0.5 * (boxes[:, :, 0] + boxes[:, :, 2]) * img_shape[0]], -1)

    if img is not None:
        # If an image was give we assume that it should be showed
        img_list = list()
        # show centers
        for ind, I in enumerate(img):
            I = I.copy()
            for s, b in zip(scores[ind], box_centers[ind]):
                if s < min_score_cand:
                    continue
                c = (int(b[0]),
                     int(b[1]))
                I = cv2.circle(I, c, radius=5, color=(0, 0, 255), thickness=-1)
            img_list.append(I)

        from utils.StitchedImage import StitchedImage
        merge = StitchedImage(img_list)
        cv2.imshow('centers all', merge.image)
        cv2.waitKey(10)

    # resort data
    points2d = list()
    cams = list()
    for cid in range(boxes.shape[0]):
        for i in range(boxes.shape[1]):
            if scores[cid, i] > min_score_cand:
                points2d.append(box_centers[cid, i])
                cams.append(cid)

        if verbose and logger is not None:
            logger.log('Cam %d contributes %d points for triangulation' % (cid, np.sum(scores[cid] > min_score_cand)))
    points2d = np.array(points2d)

    if np.unique(cams).shape[0] >= 3:
        # find consistent 3D hypothesis for the center of bounding boxed
        point3d, inlier = triang_tool.triangulate([K_list[i] for i in cams],
                                                  [M_list[i] for i in cams],
                                                  np.expand_dims(points2d, 1),
                                                  mode=t_triangulation.RANSAC,
                                                  threshold=max_reproj_error)
        if verbose and logger is not None:
            logger.log('Found 3D point with %d inliers' % np.sum(inlier))

            if img is not None:
                img_list = list()
                for ind, (I, K, M) in enumerate(zip(img, K_list, M_list)):
                    p2d = cl.project(cl.trafo_coords(point3d, M), K)
                    c = (int(p2d[0, 0]),
                         int(p2d[0, 1]))
                    print(ind, c)
                    I = cv2.circle(I.copy(), c, radius=5, color=(0, 0, 255), thickness=-1)
                    img_list.append(I)

                from utils.StitchedImage import StitchedImage
                merge = StitchedImage(img_list)
                cv2.imshow('center consistent', merge.image)
                cv2.waitKey()

        if np.sum(inlier) > 0:
            output['xyz'] = point3d

            # select optimal box wrt the center
            order = [1, 3, 0, 2]
            boxes_opti = list()
            for cid, (K, M) in enumerate(zip(K_list, M_list)):
                uv = cl.project(cl.trafo_coords(point3d, M), K)

                # find bbox with minimal distance to found center
                diff_l2 = np.sqrt(np.sum(np.square(box_centers[cid] - uv), -1))
                diff_combined = diff_l2 / np.sqrt(scores[cid] + 0.001)  # we want to pick something with low distance and high score
                ind = np.argmin(diff_combined)
                boxes_opti.append(boxes[cid, ind, order])
            output['boxes'] = np.stack(boxes_opti)

            return output

    # If we get here its time to use the fall back solution:
    # Use top scoring bbox in each frame independently
    boxes_opti = list()
    order = [1, 3, 0, 2]
    for box, score in zip(boxes, scores):
        ind = np.argmax(score)
        boxes_opti.append(box[ind, order])
    output['boxes'] = np.stack(boxes_opti)

    if verbose and logger is not None:
        logger.log('Using fallback solution: Best scoring box from each view, because of small amount of inliers.')

    return output