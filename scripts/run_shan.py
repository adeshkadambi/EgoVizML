from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
import pickle
import logging

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image

import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, vis_detections_filtered_objects_PIL, vis_detections_filtered_objects # (1) here add a function to viz
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/res101.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="models")
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="images")
  parser.add_argument('--cuda', dest='cuda', 
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      default=True)
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)
  parser.add_argument('--thresh_hand',
                      type=float, default=0.5,
                      required=False)
  parser.add_argument('--thresh_obj', default=0.5,
                      type=float,
                      required=False)

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
    im (ndarray): a color image in BGR order
    Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def load_faster_rcnn_model(args):
    """Load a trained model from a checkpoint."""
    model_dir = os.path.join(args.load_dir, args.net + "_handobj_100K", args.dataset)

    if not os.path.exists(model_dir):
        logging.error('There is no input directory for loading network from {}'.format(model_dir))
        raise Exception('There is no input directory for loading network from {}'.format(model_dir))

    load_name = os.path.join(model_dir, 'faster_rcnn_1_8_132028.pth')

    pascal_classes = np.array(['__background__', 'targetobject', 'hand'])
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]']

    # Initialize the network
    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    fasterRCNN.create_architecture()

    logging.info("Loading checkpoint {}".format(load_name))
    device = torch.device('cuda' if args.cuda > 0 else 'cpu')
    checkpoint = torch.load(load_name, map_location=device if args.cuda <= 0 else None)
    fasterRCNN.load_state_dict(checkpoint['model'])

    if 'pooling_mode' in checkpoint:
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    logging.info('Model loaded successfully!')
    return fasterRCNN, pascal_classes

def get_image_list(root_dir):
    img_list = []
    for root, dirs, files in os.walk(root_dir):
        if 'frames' in dirs:
            frames_path = os.path.join(root, 'frames')
            shan_path = os.path.join(root, 'shan')
            os.makedirs(shan_path, exist_ok=True)
            for file in os.listdir(frames_path):
                if file.endswith(".jpg"):
                    img_path = os.path.join(frames_path, file)
                    save_img_path = os.path.join(shan_path, file[:-4] + ".png")
                    save_pkl_path = os.path.join(shan_path, file[:-4] + "_shan.pkl")
                    img_list.append((img_path, save_img_path, save_pkl_path))
    return img_list

def main():
    args = parse_args()
    logging.info('Called with args: {}'.format(args))

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda
    np.random.seed(cfg.RNG_SEED)

    # load the model
    fasterRCNN, pascal_classes = load_faster_rcnn_model(args)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    box_info = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda > 0:
        cfg.CUDA = True
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        box_info = box_info.cuda()
        fasterRCNN.cuda()
    
    with torch.no_grad():
        fasterRCNN.eval()

        start = time.time()
        max_per_image = 100
        thresh_hand = args.thresh_hand
        thresh_obj = args.thresh_obj
        vis = args.vis

        # Get image directories
        logging.info('Loading images from {}'.format(args.image_dir))
        imglist = get_image_list(args.image_dir)
        num_images = len(imglist)

        logging.info('Loaded {} images.'.format(num_images))

        # main prediction loop
        for i in range(num_images):
            logging.info('Processing image {}/{}'.format(i + 1, num_images))
            total_tic = time.time()

            img_path, save_img_path, save_pkl_path = imglist[i]

            # load image
            im = cv2.imread(img_path)
            blobs, im_scales = _get_image_blob(im)
            assert len(im_scales) == 1, "Only single-image batch implemented"
            im_blob = blobs
            im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

            im_data_pt = torch.from_numpy(im_blob)
            im_data_pt = im_data_pt.permute(0, 3, 1, 2)
            im_info_pt = torch.from_numpy(im_info_np)

            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()
            box_info.resize_(1, 1, 5).zero_()

            det_tic = time.time()

            rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info) 

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            # extract predicted params
            contact_vector = loss_list[0][0] # hand contact state info
            offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
            lr_vector = loss_list[2][0].detach() # hand side info (left/right)

            # get hand contact 
            _, contact_indices = torch.max(contact_vector, 2)
            contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

            # get hand side 
            lr = torch.sigmoid(lr_vector) > 0.5
            lr = lr.squeeze(0).float()

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if args.class_agnostic:
                        if args.cuda > 0:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        if args.cuda > 0:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                        box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= im_scales[0]

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            detect_time = det_toc - det_tic
            misc_tic = time.time()
          
            # get hand and object detections
            obj_dets, hand_dets = None, None

            for j in range(1, len(pascal_classes)):
                if pascal_classes[j] == 'hand':
                    inds = (scores[:, j] > thresh_hand).nonzero().view(-1)
                elif pascal_classes[j] == 'targetobject':
                    inds = (scores[:, j] > thresh_obj).nonzero().view(-1)

                # If there is a detection
                if inds.numel() > 0:
                    cls_scores = scores[inds, j]
                    _, order = torch.sort(cls_scores, 0, descending=True)

                    cls_boxes = pred_boxes[inds]
                    if not args.class_agnostic:
                        cls_boxes = cls_boxes[:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]

                    if pascal_classes[j] == 'hand':
                        hand_dets = cls_dets.cpu().numpy()
                    if pascal_classes[j] == 'targetobject':
                        obj_dets = cls_dets.cpu().numpy()
            
            
            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            logging.info('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(num_images + 1, len(imglist), detect_time, nms_time))


            # save hand detections
            if vis:
                im2show = np.copy(im)
                im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, thresh_hand, thresh_obj)

                # save image
                os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
                im2show.save(save_img_path)

                # save predictions
                os.makedirs(os.path.dirname(save_pkl_path), exist_ok=True)
                
                with open(save_pkl_path, 'wb') as f:
                    preds = {'objects': obj_dets, 'hands': hand_dets}
                    pickle.dump(preds, f)

if __name__ == '__main__':
   main()
