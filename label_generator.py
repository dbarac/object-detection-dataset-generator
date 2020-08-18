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
        self.get_bounding_box = bounding_box_init_method
        self.display_image = image_display_method
        self.dataset_name = dataset_name 

        #load json dataset or create it if it doesn't exist
        self.dataset_filename  = "results/" + dataset_name + "/" + dataset_name + ".json"
        if os.path.isfile(self.dataset_filename):
            with open(self.dataset_filename, "r") as dataset_file:
                self.dataset = json.load(dataset_file)
        else:
            self.dataset = self.initialize_dataset()


    def load_tracker(self):
        self.tracker = SiamRPNotb()
        self.tracker.load_state_dict(torch.load('tracker/SiamRPNOTB.model'))
        self.tracker.eval()
        if torch.cuda.is_available():
            self.tracker.cuda()


    def generate_labels(self, video, class_name):
        """
        Add bounding box annotations to the dataset for a given video.

        After the user selects the initial bounding box, the tracker
        generates boxes for all frames in the rest of the video.
        Some of the frames and annotations are saved to the dataset.

        Tracking can be paused by pressing 'p'.
        By pressing 'r' the user can reset the tracker by selecting the bbox
        for the current frame.
        """
        retval, first_frame = video.read()
        first_frame = imutils.resize(first_frame, height=800)
        x, y, w, h = self.get_bounding_box(first_frame)
        center_x, center_y = x + w/2, y + h/2
        object_position = np.array([center_x, center_y])
        object_size = np.array([w, h])
        tracker_state = SiamRPN_init(first_frame, object_position, object_size, self.tracker)

        frame_id = 0
        if len(self.dataset["images"]) > 0:
            frame_id = self.dataset["images"][-1]["id"]
        category_id = self.add_category(class_name)
        
        count = 0
        paused = False
        while True:
            if not paused:
                frame_id += 1
                retval, frame = video.read()
                if retval == False:
                    break
                frame = imutils.resize(frame, height=800)
                frame_height, frame_width = frame.shape[0], frame.shape[1]
                tracker_state = SiamRPN_track(tracker_state, frame)
                result = cxy_wh_2_rect(tracker_state['target_pos'], tracker_state['target_sz'])
                x, y, w, h = [int(i) for i in result]
                
                if frame_id % 5 == 0:
                    print(frame_id)
                    self.save_image(frame, class_name, frame_id)
                    self.add_label(category_id, frame_width, frame_height,
                                   frame_id, [[x, y, w, h]])
                    count += 1
            key_pressed = self.display_image(frame, x, y, w, h)
            if key_pressed == ord("p"):
                paused = not paused
            elif key_pressed == ord("r"):
                #reset tracker:
                #select bbox for current frame and continue tracking
                x, y, w, h = self.get_bounding_box(frame)
                center_x, center_y = x + w/2, y + h/2
                object_position = np.array([center_x, center_y])
                object_size = np.array([w, h])
                tracker_state = SiamRPN_init(frame, object_position, object_size, self.tracker)
                paused = False

        with open(self.dataset_filename, "w") as dataset_file:
            json.dump(self.dataset, dataset_file)
        print("{} images have been added to '{}' dataset".format(count, self.dataset_name))


    def label_image_manually(self, image, class_name):
        """
        Select a bounding box for given image and add the
        annotation and image to the dataset.
        """
        image = imutils.resize(image, height=800)
        height, width = image.shape[0], image.shape[1]
        cv2.imshow("COCO label generator", image)
        cv2.waitKey(5)
        n_boxes = input("How many bounding boxes? (default = 1):") or 1
        n_boxes = int(n_boxes)
        bboxes = []
        for i in range(n_boxes):
            bboxes.append(self.get_bounding_box(image))

        image_id = 0
        if len(self.dataset["images"]) > 0:
            image_id = self.dataset["images"][-1]["id"] + 1
        print(image_id)
        category_id = self.add_category(class_name)
        self.save_image(image, class_name, image_id)

        self.add_label(category_id, width, height, image_id, bboxes)

        with open(self.dataset_filename, "w") as dataset_file:
            json.dump(self.dataset, dataset_file)


    def initialize_dataset(self):
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
        Add category to dataset json if it doesn't exist
        and return id of the category.
        """
        max_id = 0
        for cat in self.dataset["categories"]:
           max_id = max(max_id, cat["id"])
           if cat["name"] == category_name: #category exists
               return cat["id"] 

        category_id = max_id + 1
        self.dataset["categories"].append({
            "id": category_id,
            "name": category_name,
            "supercategory": "none"
        })
        return category_id


    def save_image(self, image, class_name, image_id):
        """
        Save a a single image to disk.
        """
        if not os.path.isdir("results"):        
            os.mkdir("results")
        if not os.path.isdir("results/" + self.dataset_name):
            os.mkdir("results/" + self.dataset_name)
        cv2.imwrite("results/" + self.dataset_name + "/" + str(image_id) + ".jpg", image)


    def add_label(self, category_id, image_width, image_height, image_id, bboxes):
        """
        Save bounding box annotation and file info to json dataset.
        """
        image_info = {
            "id": image_id,
            "width": image_width,
            "height": image_height,
            "file_name": str(image_id) + ".jpg"
        }
        self.dataset["images"].append(image_info)
        
        if len(self.dataset["annotations"]) > 0:
            annotation_id = self.dataset["annotations"][-1]["id"] + 1
        else:
            annotation_id = 0
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            annotation_info = {
                "id": annotation_id,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": bbox,
                "category_id": category_id
            }
            self.dataset["annotations"].append(annotation_info)
            annotation_id += 1


    def print_dataset_stats(self):
        print("dataset:", self.dataset_name)
        print(" - {} categories".format(len(self.dataset["categories"])))
        print(" - {} images".format(len(self.dataset["images"])))
        print(" - {} annotations".format(len(self.dataset["annotations"])))


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
    print(selection)
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
    ap.add_argument("-p", "--preview", action="store_true", #not implemented
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
                if not os.path.isfile(file):
                    continue
                print(file)
                image = cv2.imread(file)
                label_generator.label_image_manually(image, args["class"])
            print("All images in {} have been labeled.".format(args["dir"]))
        else:
            for file in files:
                video = cv2.VideoCapture(file)
                label_generator.generate_labels(video, args["class"])
            print("Labels have been generated for all videos in {}".format(args["dir"]))
    else:
        video = cv2.VideoCapture(args["video"])
        label_generator.generate_labels(video,  args["class"])
    label_generator.print_dataset_stats()


if __name__ == '__main__':
    main()
