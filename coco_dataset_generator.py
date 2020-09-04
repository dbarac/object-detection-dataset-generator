import argparse
import json
import os.path

import cv2
import torch
import imutils
import numpy as np

#check if necessary environment variables are set
env_vars = os.environ.copy()
if "PYSOTPATH" not in env_vars:
    print("Error: $PYSOTPATH=/path/to/pysot needs to be set to find pysot trackers and config")
    exit(1)
if "PYTHONPATH" not in env_vars:
    print("Error: /path/to/pysot needs to be in $PYTHONPATH find the pysot package")
    exit(1)

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

class COCODatasetGenerator():
    """Generates detection annotations in COCO format by tracking objects in videos."""

    def __init__(self, dataset_name, tracker_config, tracker_snapshot):
        """
        Args:
          - dataset_name (str): Where all images and annotations will be saved
          - tracker_config (str): Path to pysot config file for the tracker
          - tracker_snapshot (str): Path to .pth file of pysot tracker
        """
        self.dataset_name = dataset_name
        self.saved_frames = set()
        self.load_tracker(tracker_config, tracker_snapshot)
        #create dataset directory if it doesn't exist
        if not os.path.isdir("results"):        
            os.mkdir("results")
        if not os.path.isdir("results/" + self.dataset_name):
            os.mkdir("results/" + self.dataset_name)
        #load json dataset or create it if it doesn't exist
        self.dataset_filename  = "results/" + dataset_name + "/" + dataset_name + ".json"
        if os.path.isfile(self.dataset_filename):
            with open(self.dataset_filename, "r") as dataset_file:
                self.dataset = json.load(dataset_file)
        else:
            self.dataset = self.initialize_dataset()


    def load_tracker(self, tracker_config, tracker_snapshot):
        """Load the selected pysot tracker.

        Args:
          - tracker_config (str): Path to pysot config file for the tracker
          - tracker_snapshot (str): Path to .pth file of pysot tracker
        """
        cfg.merge_from_file(tracker_config)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda' if cfg.CUDA else 'cpu')
        model = ModelBuilder()
        model.load_state_dict(torch.load(tracker_snapshot))
        model.eval().to(device)
        self.tracker = build_tracker(model)


    def annotate_multi_object_video(self, video_path, class_name):
        """
        Make bounding box annotations for video with multiple objects
        of the same class.

        After selecting the number of objects, the tracker will
        run once for each of the objects.
        (pysot trackers do not support multi object tracking)

        Args:
          - video_path (str)
          - class_name (str)
        """
        video = cv2.VideoCapture(video_path)
        retval, first_frame = video.read()
        first_frame = imutils.resize(first_frame, height=800)
        del video
        print("How many objects to track?")
        cv2.imshow("COCO dataset generator", first_frame)
        cv2.waitKey(2)
        n_objects = int(input())
        for i in range(n_objects):
            print("Tracking object", i+1)
            self.annotate_video_frames(video_path, class_name, remember_saved_frames=True)


    def annotate_video_frames(self, video_path, class_name,
                              discard_blurry_frames=False, remember_saved_frames=False):
        """
        Add bounding box annotations to the dataset for a given video.

        After the user selects the initial bounding box, the tracker
        generates boxes for all frames in the rest of the video.
        Some of the frames and annotations are saved to the dataset.

        Tracking can be paused by pressing 'p'.
        By pressing 'r' the user can reset the tracker by selecting the bbox
        for the current frame.
        By pressing 'n' the user can end the tracking process for the video.

        Args:
          - video_path (str)
          - class_name (str)
          - discard_blurry_frames (bool): If true, blurry video frames won't be saved
          - remember_saved_frames (bool): Used when running multiple times on the same video
        """
        video = cv2.VideoCapture(video_path)
        category_id = self.add_category(class_name)
        retval, first_frame = video.read()
        first_frame = imutils.resize(first_frame, height=800)
        x, y, w, h = self.get_bounding_box(first_frame)
        self.tracker.init(first_frame, [x, y, w, h])

        frame_count = 0
        frames_to_skip = 10 #to avoid saving each frame
        saved_frames = 0
        valid_count = 0 #count of frames with low blur
        paused = False

        while True:
            if not paused:
                frame_count += 1
                retval, frame = video.read()
                if retval == False: #video is finished
                    break
                frame = imutils.resize(frame, height=800)
                outputs = self.tracker.track(frame)
                x, y, w, h = [int(i) for i in outputs["bbox"]]
                
                should_be_saved = False
                if discard_blurry_frames:
                    blurry = self.bbox_is_blurry(frame, [x, y, w, h])
                    if not blurry:
                        valid_count += 1
                        if valid_count % frames_to_skip == 0:
                            should_be_saved = True
                elif frame_count % frames_to_skip == 0:
                    should_be_saved = True

                if should_be_saved:
                    frame_name = os.path.splitext(os.path.split(video_path)[-1])[0] \
                                 + "_" + str(frame_count)
                    self.save_image(frame, frame_name)
                    frame_id = self.save_image_info(frame, frame_name)
                    self.save_bbox_info(frame_id, [[x, y, w, h]], category_id)
                    saved_frames += 1
                    if remember_saved_frames:
                        self.saved_frames.add(frame_name)
                    print(frame_name, "saved")

            #display image and process events
            key_pressed = self.display_image(frame, outputs)
            if key_pressed == ord("p"):
                paused = not paused
            elif key_pressed == ord("r"):
                #reset tracker:
                #select bbox for current frame and continue tracking
                x, y, w, h = self.get_bounding_box(frame)
                self.tracker.init(frame, [x, y, w, h])
                paused = False
            elif key_pressed == ord("n"): #next video
                break

        with open(self.dataset_filename, "w") as dataset_file:
            json.dump(self.dataset, dataset_file)
        print("{} images have been added to '{}' dataset".format(saved_frames, self.dataset_name))


    def bbox_is_blurry(self, frame, bbox):
        """
        Check amount of blur inside bbox area.

        High variance of laplacian value ==> lots of edges <==> low blur
        Threshold should be set depending on object type.

        Args:
          - frame (numpy.ndarray)
          - bbox (List[int])
        """
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w].copy()
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        roi_normalized = cv2.resize(roi_gray, dsize=(64,64))

        variance_of_laplacian = cv2.Laplacian(roi_normalized, cv2.CV_64F).var()
        print("Variance of laplacian:", variance_of_laplacian)
        threshold = 22.5
        if variance_of_laplacian < threshold:
            return True
        else:
            return False


    def annotate_image_manually(self, image_path, class_name):
        """
        Select bounding boxes for given image and add the
        annotation and image to the dataset.

        Args:
          - image_path (str)
          - class_name (str)
        """
        image = cv2.imread(image_path)
        image = imutils.resize(image, height=800)
        height, width = image.shape[0], image.shape[1]
        cv2.imshow("COCO dataset generator", image)
        cv2.waitKey(5)
        n_boxes = input("How many bounding boxes? (default = 1):") or 1
        n_boxes = int(n_boxes)
        bboxes = []
        for i in range(n_boxes):
            bboxes.append(self.get_bounding_box(image))

        category_id = self.add_category(class_name)
        image_name = os.path.splitext(os.path.split(image_path)[-1])[0]
        self.save_image(image, image_name)
        image_id = self.save_image_info(image, image_name)
        self.save_bbox_info(image_id, bboxes, category_id)

        with open(self.dataset_filename, "w") as dataset_file:
            json.dump(self.dataset, dataset_file)


    def initialize_dataset(self):
        """Create basic structure for a COCO detection dataset json file."""
        return {
            "images": [],
            "categories": [],
            "annotations": []
        }


    def add_category(self, category_name):
        """
        Add object category to dataset json if it doesn't exist
        and return id of the category.

        Args:
          - category_name (str)
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


    def save_image(self, image, image_name):
        """Save a a single image to results/dataset_name/.

        Args:
          - image (numpy.ndarray)
          - image_name (str)
        """
        if not os.path.isdir("results"):        
            os.mkdir("results")
        if not os.path.isdir("results/" + self.dataset_name):
            os.mkdir("results/" + self.dataset_name)
        cv2.imwrite("results/" + self.dataset_name + "/" + image_name + ".jpg", image)


    def save_image_info(self, image, image_name):
        """
        Save image info to json COCO dataset.
        Return image id.

        Args:
          - image (numpy.ndarray)
          - image_name (str)
        """
        if image_name in self.saved_frames:
            for image in self.dataset["images"]:
                if image["file_name"] == image_name + ".jpg":
                    return image["id"]

        height, width = image.shape[0], image.shape[1]
        image_id = 0
        if len(self.dataset["images"]) > 0:
            image_id = self.dataset["images"][-1]["id"] + 1

        image_info = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": image_name + ".jpg"
        }
        self.dataset["images"].append(image_info)
        return image_id


    def save_bbox_info(self, image_id, bboxes, category_id):
        """Save all image bounding boxes to json COCO dataset.

        Args:
          - image_id (int)
          - bboxes (List[List[int]])
          - category_id (int)
        """
        annotation_id = 0
        if len(self.dataset["annotations"]) > 0:
            annotation_id = self.dataset["annotations"][-1]["id"] + 1

        for bbox in bboxes:
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


    def get_bounding_box(self, image):
        """
        Ask user to select ROI.
        If the selection was incorrect/canceled (c is pressed), ask for new selection.

        Args:
          - image (numpy.ndarray)
        """
        redo_selection = (0,0,0,0)
        while True:
            cv2.imshow("COCO dataset generator", image)
            selection = cv2.selectROI("COCO dataset generator", image)
            if selection != redo_selection:
                break
        x, y, w, h = selection
        return (x, y, w, h)


    def display_image(self, image, outputs):
        """Draw outputs (masks, bboxes) and display image.

        Args:
          - image (numpy.ndarray)
          - outputs (Dict)
        """
        if 'polygon' in outputs:
            mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
            mask = mask.astype(np.uint8)
            mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
            image = cv2.addWeighted(image, 0.77, mask, 0.23, -1)
            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(image, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 255, 0), 1)
        else:
            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(image, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 255, 0), 1)
        cv2.imshow("COCO dataset generator", image)
        return cv2.waitKey(1) #return key pressed by user


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, help="path to input video file")
    ap.add_argument("-d", "--dataset", type=str,
                    help="name of dataset where annotations should be saved")
    ap.add_argument("-c", "--class", type=str, help="class name of object in video")
    ap.add_argument("--dir", type=str,
                    help="create annotations for all files in given directory")
    ap.add_argument("--manual", action="store_true",
                    help="manually select all bounding boxes (used for image directories)")
    ap.add_argument("--discard_blurry_frames", action="store_true",
                    help="discard frame if area inside the bbox is blurry (off by default)")
    ap.add_argument("--multi_object", action="store_true",
                    help="track multiple objects in the same video (run tracker once for each object)")
    ap.add_argument("--tracker_config", type=str,
                    help="config file for pysot tracker settings (set to SiamMask_E by default)",
                    default=os.path.expandvars("$PYSOTPATH/experiments/siammaske_r50_l3/config.yaml"))
    ap.add_argument("--tracker_snapshot", type=str,
                    help="pysot tracker .pth snapshot location (set to SiamMask_E by default)",
                    default=os.path.expandvars("$PYSOTPATH/experiments/siammaske_r50_l3/model.pth"))
    args = vars(ap.parse_args())
    
    print("Configuration:")
    print(" - video:", args["video"])
    print(" - dataset:", args["dataset"])
    print(" - class:", args["class"])
    print(" - directory:", args["dir"])
    print(" - multi object:", args["multi_object"])
    print(" - manual", args["manual"])
    print(" - discard blurry frames:", args["discard_blurry_frames"])
    print(" - tracker config:", args["tracker_config"])
    print(" - tracker model snapshot:", args["tracker_snapshot"])

    dataset_generator = COCODatasetGenerator(args["dataset"],
                                             args["tracker_config"], args["tracker_snapshot"])

    #run with given configuration
    if args["dir"]:
        files = [args["dir"] + filename for filename in os.listdir(args["dir"])]
        if args["manual"]:
            for i, image_file in enumerate(files):
                if not os.path.isfile(image_file):
                    continue
                print(image_file)
                dataset_generator.annotate_image_manually(image_file, args["class"])
                print("{}/{} images saved.".format(i+1, len(files)))
            print("All images in {} have been labeled.".format(args["dir"]))
        else:
            for i, video_file in enumerate(files):
                if args["multi_object"]:
                    dataset_generator.annotate_multi_object_video(video_file, args["class"])
                else:
                    dataset_generator.annotate_video_frames(video_file,  args["class"],
                                                            args["discard_blurry_frames"])
                print("{}/{} videos processed".format(i+1, len(files)))
            print("Labels have been generated for all videos in {}".format(args["dir"]))
    else:
        assert args["manual"] == False #manual annotation is only for image dirs
        video_file = args["video"]
        if args["multi_object"]:
            dataset_generator.annotate_multi_object_video(video_file, args["class"])
        else:
            dataset_generator.annotate_video_frames(video_file,  args["class"],
                                                    args["discard_blurry_frames"])
    dataset_generator.print_dataset_stats()


if __name__ == '__main__':
    main()
