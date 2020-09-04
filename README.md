# COCO object detection dataset generator

This tool uses a deep learning tracker to generate bounding box annotations for detection datasets.

## Prerequisites
This tool requires [pysot](https://github.com/STVIR/pysot) and the [siammask_e](https://github.com/baoxinchen/siammask_e)
add-on for pysot. Steps for installation are below.
1. Install requirements
   ```bash 
   $ pip install -r requirements.txt
   ```
2. Clone pysot repo and follow instructions in the pysot [INSTALL.md](https://github.com/STVIR/pysot/blob/master/INSTALL.md).
   The pip requirements above should cover everything needed for pysot so only the extension building step
   should be needed.
3. Clone SiamMask_E and run the install.sh script.
4. Download the siammask_r50_l3 .pth file from [PySOT Model Zoo](https://github.com/STVIR/pysot/blob/master/MODEL_ZOO.md)
   and place it in /path/to/pysot/experiments/siammaske_r50_l3/model.pth.
5. Export environment variables necessary for finding pysot:
   ```bash
   $ export PYTHONPATH=/path/to/pysot:$PYTHONPATH
   $ export PYSOTPATH=/path/to/pysot
   
   ```

## Some usage examples

Generate bounding boxes for a video file. Set the dataset name and
the class for the selected object. Images and a COCO json file will be
saved in the results/ directory:

```bash
$ python coco_dataset_generator.py --video example_data/train/VID_20200730_151459.mp4 --dataset uticnice --class uticnica
```

Generate boxes for all videos in the selected directory and discard blurry frames:
```bash
$ python coco_dataset_generator.py --dir example_data/train/ --dataset uticnice --class uticnica --discard_blurry_frames
```

Manually select bounding boxes for all images in a directory:

```bash
$ python coco_dataset_generator.py --manual --dir example_data/valid/ --dataset uticnice_valid --class uticnica
```

If the selected bounding box was incorrect press 'c' to cancel and select it again.
Tracking can be paused by pressing 'p'.
Pressing 'r' during tracking resets the tracker - user selects a bbox for the current frame so the tracker
can continue.
Pressing 'n' ends the current video.

This project was tested with Python 3.7.6 and package versions found in requirements.txt
