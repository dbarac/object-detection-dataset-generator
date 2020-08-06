# Object detection label generator

Use a deep learning tracker to generate bounding box annotations for detection datasets.

## Prerequisites

1. Download pretrained tracker weights 'SiamRPNOTB.model' from
  [Google drive](https://drive.google.com/drive/folders/1BtIkp5pB6aqePQGlMb2_Z7bfPy6XEj6H)
  and place the file in the tracker/ directory.

2. Install requirements (pytorch, opencv, numpy, imutils)
   ```bash 
   $ pip install -r requirements.txt
   ```

## Some examples

Generate bounding boxes for a video file. Set the dataset name and
the class for the selected object. Images and a COCO json file will be
saved in the results/ directory:

```bash
$ python label_generator.py --video example_data/train/VID_20200730_151459.mp4 --dataset uticnice --class uticnica
```

Generate boxes for all videos in the selected directory:
```bash
$ python label_generator.py --dir example_data/train/ --dataset uticnice --class uticnica
```

Manually select bounding boxes for all images in a directory:

```bash
$ python label_generator.py --validation --dir example_data/valid/ --dataset uticnice_valid --class uticnica
```

If the selected bounding box was incorrect press 'c' to cancel and select it again.
Tracking can be paused by pressing 'p'.
Pressing 'r' during tracking resets the tracker - user selects a bbox for the current frame so the tracker
can continue.

Tracker used: [DaSiamRPN](https://github.com/foolwood/DaSiamRPN)
