import glob 
import cv2
import torch
import argparse
import imutils
import os.path
import numpy as np
import json

from tracker.net import SiamRPNvot
from tracker.run_SiamRPN import SiamRPN_init, SiamRPN_track
from tracker.utils import get_axis_aligned_bbox, cxy_wh_2_rect


class LabelGenerator():
    """Generate labels in COCO format for object detection"""

    def __init__(self, bounding_box_initializer, image_display_method=None):
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
        first_frame = imutils.resize(first_frame, width=480)
        x, y, w, h = self.get_initial_bounding_box(first_frame)
        center_x, center_y = x + w/2, y + h/2
        object_position = np.array([center_x, center_y])
        object_size = np.array([w, h])
        tracker_state = SiamRPN_init(first_frame, object_position, object_size, self.tracker)
        frame_count = 0

        label_filename = "results/" + dataset_name + "/" + dataset_name + ".json"
        if os.path.isfile(label_filename):
            with open(label_filename, "r") as label_file:
                labels = json.load(label_file)
        else:
            labels = self.initialize_labels()
        category_id = self.add_category(labels, class_name)

        while True:
            frame_count += 1
            retval, frame = video.read()
            if retval == False:
                break
            frame = imutils.resize(frame, width=480)
            frame_height, frame_width = frame.shape[0], frame.shape[1]
            tracker_state = SiamRPN_track(tracker_state, frame)
            result = cxy_wh_2_rect(tracker_state['target_pos'], tracker_state['target_sz'])
            x, y, w, h = [int(i) for i in result]

            display_frame(frame, x, y, w, h)
            self.save_video_frame(frame, dataset_name, class_name, frame_count)
            self.add_label(category_id, frame_width, frame_height, frame_count, [x, y, w, h], labels)

        with open(label_filename, "w") as dataset_file:
            json.dump(labels, dataset_file)


    def initialize_labels(self):
        """
        Create basic structure for a COCO detection dataset json file.
        """
        return {
            "images": [],
            "categories": [],
            "annotations": []
        }


    def add_category(self, dataset, category_name):
        """
        Add category to dataset json if it doesn't exist and return id of the category.
        """
        max_id = -1
        for cat in dataset["categories"]:
           max_id = max(max_id, cat["id"])
           if cat["name"] == category_name: #category exists
               return cat["id"] 

        category_id = max_id + 1
        dataset["categories"].append({
            "id": category_id,
            "name": category_name,
            "supercategory": "none"
        })
        return category_id


    def save_video_frame(self, frame, dataset_name, class_name, frame_count):
        """
        Save a a single frame to disk.
        """
        if not os.path.isdir("results"):        
            os.mkdir("results")
        if not os.path.isdir("results/" + dataset_name):
            os.mkdir("results/" + dataset_name)
        cv2.imwrite("results/" + dataset_name + "/" + str(frame_count) + ".jpg", frame)


    def add_label(self, category_id, frame_width, frame_height, frame_count, bbox, dataset):
        """
        Save annotation and corresponding file info to json dataset.
        """
        frame_info = {
            "id": frame_count,
            "width": frame_width,
            "height": frame_height,
            "file_name": str(frame_count) + ".jpg"
        }
        dataset["images"].append(frame_info)

        annotation_info = {
            "area": bbox[2] * bbox[3],
            "imageid": frame_count,
            "bbox": bbox,
            "category_id": category_id
        }
        dataset["annotations"].append(annotation_info)


def get_bounding_box(image):
    cv2.imshow("COCO label generator", image)
    x, y, w, h = cv2.selectROI("COCO label generator", image)
    return (x, y, w, h)


def display_frame(frame, x, y, w, h):
    """display frame and draw a bounding box"""
    cv2.rectangle(frame, (x, y),  (x + w, y + h), (0, 255, 255), 2)
    cv2.imshow("COCO label generator", frame)
    cv2.waitKey(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, help="path to input video file")
    ap.add_argument("-d", "--dataset_name", type=str,
                    help="name of dataset where labels should be saved")
    ap.add_argument("-c", "--class", type=str, help="class name of object in video")
    args = vars(ap.parse_args())
    print(args)

    label_generator = LabelGenerator(bounding_box_initializer=get_bounding_box)

    video = cv2.VideoCapture(args["video"])
    label_generator.generate_labels(video, args["dataset_name"], args["class"])


if __name__ == '__main__':
    main()
