import glob 
import cv2
import torch
import argparse
import imutils
import os.path
import numpy as np
import json

from tracker.net import SiamRPNvot, SiamRPNotb
from tracker.run_SiamRPN import SiamRPN_init, SiamRPN_track
from tracker.utils import get_axis_aligned_bbox, cxy_wh_2_rect


class LabelGenerator():
    """Generate labels in COCO format for object detection"""

    def __init__(self, dataset_name, bounding_box_init_method, image_display_method):
        self.load_tracker()
        self.set_bounding_box = bounding_box_init_method
        self.display_image = image_display_method
        self.dataset_name = dataset_name 
        #load json dataset or create it if it doesn't exist
        self.label_filename = "results/" + dataset_name + "/" + dataset_name + ".json"
        if os.path.isfile(self.label_filename):
            with open(self.label_filename, "r") as label_file:
                self.labels = json.load(label_file)
        else:
            self.labels = self.initialize_labels()


    def load_tracker(self):
        self.tracker = SiamRPNotb()
        self.tracker.load_state_dict(torch.load('tracker/SiamRPNOTB.model'))
        self.tracker.eval() #check docs
        if torch.cuda.is_available():
            self.tracker.cuda()


    def generate_labels(self, video, class_name):
        retval, first_frame = video.read()
        first_frame = imutils.resize(first_frame, width=600)
        x, y, w, h = self.set_bounding_box(first_frame)
        center_x, center_y = x + w/2, y + h/2
        object_position = np.array([center_x, center_y])
        object_size = np.array([w, h])
        tracker_state = SiamRPN_init(first_frame, object_position, object_size, self.tracker)

        frame_id = 0
        if len(self.labels["images"]) > 0:
            frame_id = self.labels["images"][-1]["id"]
        category_id = self.add_category(class_name)
        
        count = 0
        paused = False
        while True:
            if not paused:
                frame_id += 1
                retval, frame = video.read()
                if retval == False:
                    break
                frame = imutils.resize(frame, width=600)
                frame_height, frame_width = frame.shape[0], frame.shape[1]
                tracker_state = SiamRPN_track(tracker_state, frame)
                result = cxy_wh_2_rect(tracker_state['target_pos'], tracker_state['target_sz'])
                x, y, w, h = [int(i) for i in result]
                
                if frame_id % 5 == 0:
                    print(frame_id)
                    self.save_image(frame, class_name, frame_id)
                    self.add_label(category_id, frame_width, frame_height,
                                   frame_id, [x, y, w, h])
                    count += 1
            key_pressed = self.display_image(frame, x, y, w, h)
            if key_pressed == ord("p"):
                paused = not paused
            elif key_pressed == ord("r"):
                #reset tracker:
                #select bbox for current frame and continue tracking
                x, y, w, h = self.set_bounding_box(frame)
                center_x, center_y = x + w/2, y + h/2
                object_position = np.array([center_x, center_y])
                object_size = np.array([w, h])
                tracker_state = SiamRPN_init(frame, object_position, object_size, self.tracker)
                paused = False

        with open(self.label_filename, "w") as dataset_file:
            json.dump(self.labels, dataset_file)
        print("{} images have been added to '{}' dataset".format(count, self.dataset_name))


    def label_image_manually(self, image, dataset_name, class_name):
        image = imutils.resize(image, width=600)
        x, y, w, h = self.set_bounding_box(image)
        center_x, center_y = x + w/2, y + h/2
        object_position = np.array([center_x, center_y])
        object_size = np.array([w, h])
        height, width = image.shape[0], image.shape[1]

        image_id = 0
        if len(self.labels["images"]) > 0:
            image_id = self.labels["images"][-1]["id"] + 1
        print(image_id)
        category_id = self.add_category(class_name)
        self.save_image(image, class_name, image_id)
        self.add_label(category_id, width, height, image_id, [x, y, w, h])


    def initialize_labels(self):
        """
        Create basic structure for a COCO detection dataset json file.
        """
        return {
            "images": [],
            "categories": [],
            "annotations": []
        }


    def add_category(self, category_name):
        """
        Add category to dataset json if it doesn't exist and return id of the category.
        """
        max_id = 0
        for cat in self.labels["categories"]:
           max_id = max(max_id, cat["id"])
           if cat["name"] == category_name: #category exists
               return cat["id"] 

        category_id = max_id + 1
        self.labels["categories"].append({
            "id": category_id,
            "name": category_name,
            "supercategory": "none"
        })
        return category_id


    def save_image(self, frame, class_name, frame_count):
        """
        Save a a single frame to disk.
        """
        if not os.path.isdir("results"):        
            os.mkdir("results")
        if not os.path.isdir("results/" + self.dataset_name):
            os.mkdir("results/" + self.dataset_name)
        cv2.imwrite("results/" + self.dataset_name + "/" + str(frame_count) + ".jpg", frame)


    def add_label(self, category_id, frame_width, frame_height, frame_count, bbox):
        """
        Save annotation and corresponding file info to json dataset.
        """
        frame_info = {
            "id": frame_count,
            "width": frame_width,
            "height": frame_height,
            "file_name": str(frame_count) + ".jpg"
        }
        self.labels["images"].append(frame_info)
        
        if len(self.labels["annotations"]) > 0:
            id = self.labels["annotations"][-1]["id"] + 1
        else:
            id = 0
        annotation_info = {
            "id": id,
            "area": bbox[2] * bbox[3],
            "image_id": frame_count,
            "bbox": bbox,
            "category_id": category_id
        }
        self.labels["annotations"].append(annotation_info)


    def print_dataset_stats(self):
        print("dataset:", self.dataset_name)
        print(" - {} categories".format(len(self.labels["categories"])))
        print(" - {} images".format(len(self.labels["images"])))
        print(" - {} annotations".format(len(self.labels["annotations"])))


def get_bounding_box(image):
    """
    Ask user to select ROI.
    If the selection was incorrect/canceled (c is pressed), ask for new selection.
    """
    redo_selection = (0,0,0,0)
    while True:
        cv2.imshow("COCO label generator", image)
        selection = cv2.selectROI("COCO label generator", image)
        if selection != redo_selection:
            break
    x, y, w, h = selection
    return (x, y, w, h)


def display_frame(frame, x, y, w, h):
    """display frame and draw a bounding box"""
    cv2.rectangle(frame, (x, y),  (x + w, y + h), (0, 255, 255), 1)
    cv2.imshow("COCO label generator", frame)
    return cv2.waitKey(1) #return key pressed by user


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, help="path to input video file")
    ap.add_argument("-d", "--dataset_name", type=str,
                    help="name of dataset where labels should be saved")
    ap.add_argument("-c", "--class", type=str, help="class name of object in video")
    ap.add_argument("-p", "--preview", action="store_true",
                    help="preview only, labels and frames will not be saved")
    ap.add_argument("--dir", type=str,
                    help="create labels for all files in given directory")
    ap.add_argument("--validation", action="store_true",
                    help="manually select all bounding boxes for validation")
    args = vars(ap.parse_args())
    print(args)

    label_generator = LabelGenerator(args["dataset_name"],
                                     bounding_box_init_method=get_bounding_box,
                                     image_display_method=display_frame)
    if args["dir"]:
        files = [args["dir"] + filename for filename in os.listdir(args["dir"])]
        if args["validation"]:
            for file in files:
                print(file)
                image = cv2.imread(file)
                label_generator.label_image_manually(image, args["dataset_name"], args["class"])
            print("All images in {}/ have been labeled.".format(args["dir"]))
        else:
            for file in files:
                video = cv2.VideoCapture(file)
                label_generator.generate_labels(video, args["dataset_name"], args["class"])
            print("Labels have been generated for all videos in {}/".format(args["dir"]))
    else:
        video = cv2.VideoCapture(args["video"])
        label_generator.generate_labels(video,  args["class"])
    label_generator.print_dataset_stats()


if __name__ == '__main__':
    main()
