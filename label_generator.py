import glob import cv2
import torch
import argparse
import imutils

from tracker.net import SiamRPNvot
from tracker.run_SiamRPN import SiamRPN_init, SiamRPN_track
from tracker.utils import get_axis_aligned_bbox, cxy_wh_2_rect


class LabelGenerator():
    """Generate labels in COCO format for object detection"""

    def __init__(self, bounding_box_initializer, image_display_method=None, **kwargs):
        self.load_tracker()
        self.get_initial_bounding_box = bounding_box_initializer


    def load_tracker(self):
        self.tracker = SiamRPNvot()
        self.tracker.load_state_dict(torch.load('tracker/SiamRPNVOT.model'))
        self.tracker.eval() #check docs
        if torch.cuda.is_available():
            self.tracker.cuda()


    def generate_labels(self, video, dataset_name, class_name):
        retval, first_frame = video.read()
        x, y, w, h = self.get_initial_bounding_box(first_frame)
        center_x, center_y = x + w/2, y + h/2
        object_position = np.array([center_x, center_y])
        object_size = np.array([w, h])
        tracker_state = SiamRPN_init(first_frame, object_position, object_size, self.tracker)
        frame_count = 0
        while True:
            retval, frame = video.read()
            if retval == False:
                break
            tracker_state = SiamRPN_track(tracker_state, frame)
            result = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            x, y, w, h = [int(i) for i in result]

            save_video_frame(frame, frame_count)
            save_label(class_name, frame_count, x, y, w, h)


    def save_video_frame(self):
        pass


    def save_label(self):
        pass
 

def get_bounding_box(image):
    cv2.show("COCO label generator", image)
    x, y, w, h = cv2.selectROI("COCO label generator", image)
    return (x, y, w, h)


def display_frame(frame, x, y, w, h):
    """display frame and draw a bounding box"""
    cv2.rectangle(frame, (x, y),  (x + w, y + h), (0, 255, 255), 2)
    cv2.imshow("COCO label generator", frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, help="path to input video file")
    ap.add_argument("-d", "--dataset-name", type=str,
                    help="name of dataset where labels should be saved")
    ap.add_argument("-c", "--class", type=str, help="class name of object in video")
    args = vars(ap.parse_args())

    label_generator = LabelGenerator(bounding_box_initializer=get_bounding_box)

    video = cv2.VideoCapture(args["video"])
    label_generator.generate_labels(video_frames, args["dataset-name"], args["class"])


if __name__ == '__main__':
    main()
