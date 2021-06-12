import addict
cfg = addict.Dict()


cfg.name_table = ['background', 
               'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


cfg.img_root_eval = 'C:\\dataset\\microsoft-coco\\val2017'
cfg.ann_file_eval = 'C:\\dataset\\microsoft-coco\\instances_val2017.json'
cfg.img_root_train = 'C:\\dataset\\microsoft-coco\\train2017'
cfg.ann_file_train = 'C:\\dataset\\microsoft-coco\\instances_train2017.json'


cfg.pad_n = 64


cfg.train.min_sizes = (-1, 128, 1024)
cfg.train.max_size = 1025
cfg.train.num_workers = 0
cfg.train.batch_size = 4


