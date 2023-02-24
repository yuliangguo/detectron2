# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import json
from iopath.common.file_io import PathManager

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import _create_text_labels
from detectron2.data import MetadataCatalog

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        default='/media/yuliangguo/data_ssd_4tb/Datasets/kitti/training/',
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='/media/yuliangguo/data_ssd_4tb/Datasets/kitti/training/pred_instance/',
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', 'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--camera-id',
                        help='the camera id from nuscenes',
                        default='image_2',
                        type=str)
    parser.add_argument('--save-vis',
                        help='output image visualization for debugging',
                        action='store_true')
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    args.input = os.path.join(args.input, args.camera_id)
    args.output = os.path.join(args.output, args.camera_id)

    if not os.path.exists(args.output):
        PathManager().mkdirs(args.output)

    if args.save_vis:
        vis_dir = os.path.join(args.output, 'vis')
        PathManager().mkdirs(vis_dir)

    cfg = setup_cfg(args)
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")

    demo = VisualizationDemo(cfg)

    if args.input:
        input_files = glob.glob(args.input + '/*.png')
        for path in tqdm.tqdm(input_files, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    # TODO: save prediction results in economic way
                    out_jsonfile = os.path.join(args.output, os.path.basename(path)[:-4]+'.json')

                    instances = predictions["instances"].to('cpu')
                    # save detections results in json
                    out_dict = {}
                    out_dict['boxes'] = instances.pred_boxes.tensor.tolist() if instances.has("pred_boxes") else None
                    out_dict['scores'] = instances.scores.tolist() if instances.has("scores") else None
                    out_dict['classes'] = instances.pred_classes.tolist() if instances.has("pred_classes") else None
                    out_dict['labels'] = _create_text_labels(out_dict['classes'], out_dict['scores'], metadata.get("thing_classes", None))
                    with open(out_jsonfile, 'w') as f:
                        json.dump(out_dict, f)
                    # save instance masks in images
                    masks = np.asarray(instances.pred_masks)
                    for ii, mask in enumerate(masks):
                        out_jsonfile = os.path.join(args.output, os.path.basename(path)[:-4] + f'_{ii}.png')
                        cv2.imwrite(out_jsonfile, mask.astype(np.uint8)*255)
                    # save visualization upon request
                    if args.save_vis:
                        vis_filename = os.path.join(vis_dir, os.path.basename(path))
                        visualized_output.save(vis_filename)
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                    visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
