import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import scipy.ndimage
from cfg_enable import *


def filter_annotation(anno, class_id_set, height, width, hw_th=1, area_th=1):
    anno = [obj for obj in anno if not obj.get('ignore', False)]
    anno = [obj for obj in anno if obj['iscrowd'] == 0] # filter crowd annotations
    anno = [obj for obj in anno if obj['area'] >= area_th]
    anno = [obj for obj in anno if all(o >= hw_th for o in obj['bbox'][2:])]
    anno = [obj for obj in anno if obj['category_id'] in class_id_set]
    _anno = []
    for obj in anno:
        xmin, ymin, w, h = obj['bbox']
        inter_w = max(0, min(xmin + w, width) - max(xmin, 0))
        inter_h = max(0, min(ymin + h, height) - max(ymin, 0))
        if inter_w * inter_h > 0: _anno.append(obj)
    return _anno


def x_flip(img, boxes=None, masks=None):
    img = img.transpose(Image.FLIP_LEFT_RIGHT) 
    w = img.width
    if boxes is not None and boxes.shape[0] != 0:
        xmin = w - boxes[:, 3] - 1
        xmax = w - boxes[:, 1] - 1
        boxes[:, 1] = xmin
        boxes[:, 3] = xmax
    if masks is not None and masks.shape[0] != 0:
        masks = masks[:, :, ::-1]
    return img, boxes, masks


def resize_img(img, min_size=641, max_size=1281, pad_n=64, boxes=None, masks=None):
    w, h = img.size
    smaller_size, larger_size = min(w, h), max(w, h)
    scale = min_size / float(smaller_size)
    if larger_size * scale > max_size:
        scale = max_size / float(larger_size)
    ow = round(w*scale)
    oh = round(h*scale)
    img = img.resize((ow, oh), Image.BILINEAR)
    pad_w, pad_h = (pad_n - ow % pad_n) + 1, (pad_n - oh % pad_n) + 1
    if pad_w >= pad_n: pad_w -= pad_n
    if pad_h >= pad_n: pad_h -= pad_n
    img = img.crop((0, 0, ow + pad_w, oh + pad_h))
    location = torch.FloatTensor([0, 0, oh-1, ow-1, h, w])
    if boxes is not None and boxes.shape[0] != 0:
        boxes = boxes*scale
    if masks is not None and masks.shape[0] != 0:
        masks = scipy.ndimage.zoom(masks, zoom=[1, scale, scale], order=0)
        masks_tmp = np.zeros((masks.shape[0], oh + pad_h, ow + pad_w))
        masks_tmp[:, :masks.shape[1], :masks.shape[2]] = masks
        masks = masks_tmp
    # for safe
    if boxes is not None:
        boxes[:, :2].clamp_(min=0)
        boxes[:, 2].clamp_(max=oh-2)
        boxes[:, 3].clamp_(max=ow-2)
        ymin_xmin, ymax_xmax = boxes.split([2, 2], dim=1)
        h_w = ymax_xmax - ymin_xmin + 1
        m = h_w.min(dim=1)[0] <= 1
        ymax_xmax[m] = ymin_xmin[m] + 1
        boxes = torch.cat([ymin_xmin, ymax_xmax], dim=1)
    return img, location, boxes, masks


class AspectRatioBasedSampler(data.Sampler):
    def __init__(self, dataset, batch_size, min_sizes, drop_last=True):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.min_sizes  = min_sizes
        if min_sizes[0] < 0:
            print('using multi-scale training')
            self.random_en = True
        else: self.random_en = False
        order = list(range(len(self.dataset)))
        order.sort(key=lambda x: self.dataset.image_aspect_ratio(x))
        self.groups = [[order[x % len(order)] for x in range(i, i+self.batch_size)]
                        for i in range(0, len(order), self.batch_size)]
        if drop_last: 
            self.len = len(self.dataset) // self.batch_size
        else:
            self.len = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        self.groups = self.groups[:self.len]

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            if self.random_en: 
                min_size = random.randint(self.min_sizes[1], self.min_sizes[2])
            else:
                min_size = random.choice(self.min_sizes)
            for item in group:
                yield item, min_size


class Dataset(torchvision.datasets.coco.CocoDetection):
    def __init__(self, train=False, normalize=True):
        self.train = train
        if train: super().__init__(cfg.img_root_train, cfg.ann_file_train)
        else: super().__init__(cfg.img_root_eval, cfg.ann_file_eval)
        self.normalizer = transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        self.normalize = normalize
        # name_table
        self.index_to_coco = [i for i in range(len(cfg.name_table))]
        self.coco_to_index = {}
        for i, cate in enumerate(self.coco.loadCats(self.coco.getCatIds())):
            name = cate['name']
            if name in cfg.name_table:
                index = cfg.name_table.index(name)
                self.index_to_coco[index] = cate['id']
                self.coco_to_index[cate['id']] = index
        # filter self.ids
        ids = []
        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)[0]
            height, width = img_info['height'], img_info['width']
            if min(height, width) < 32: continue
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anno = self.coco.loadAnns(ann_ids)
            if len(filter_annotation(anno, self.coco_to_index, height, width))>0:
                ids.append(img_id)
        self.ids = ids
    
    def __getitem__(self, data):
        '''
        Return:
        img:      F(3, h, w)
        location: F(6)
        boxes:    F(n, 4)
        labels:   L(n)
        masks:    F(n, h, w) 0 or 1
        segm:     F(num_class, h, w) 0 or 1
        '''
        idx, min_size = data
        img, anno = super().__getitem__(idx)
        anno = filter_annotation(anno, self.coco_to_index, img.size[1], img.size[0])
        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        xmin_ymin, w_h = boxes.split([2, 2], dim=1)
        xmax_ymax = xmin_ymin + w_h - 1
        xmin, ymin = xmin_ymin.split([1, 1], dim=1)
        xmax, ymax = xmax_ymax.split([1, 1], dim=1)
        boxes = torch.cat([ymin, xmin, ymax, xmax], dim=1)
        labels = [self.coco_to_index[obj['category_id']] for obj in anno]
        labels = torch.LongTensor(labels)
        masks = [self.coco.annToMask(obj) for obj in anno]
        masks = np.stack(masks)
        # clamp
        boxes[:, :2].clamp_(min=0)
        boxes[:, 2].clamp_(max=float(img.size[1])-1)
        boxes[:, 3].clamp_(max=float(img.size[0])-1)
        # transform
        if random.random() < 0.5: img, boxes, masks = x_flip(img, boxes, masks)
        img, location, boxes, masks = resize_img(img, min_size, cfg.train.max_size, 
                                                    cfg.pad_n, boxes, masks)
        img = transforms.ToTensor()(img)
        masks = torch.FloatTensor(masks)
        n, h, w = masks.shape
        segm = torch.zeros(len(cfg.name_table), h, w)
        for i in range(n):
            segm[int(labels[i])] += masks[i]
        segm.clamp_(max=1)
        if self.normalize: img = self.normalizer(img)
        return img, location, boxes, labels, masks, segm
    
    def collate_fn(self, data):
        '''
        Return:
        imgs:      F(b, 3, h, w)
        locations: F(b, 6)
        boxes:     F(b, max_n, 4)
        labels:    L(b, max_n)            bg:0
        masks:     F(b, max_n, max_h, max_w) bg:0, fg:1
        segm:      F(b, num_class, h, w) 0 or 1
        '''
        imgs, locations, boxes, labels, masks, segm = zip(*data)
        locations = torch.stack(locations)
        batch_num = len(imgs)
        max_h, max_w, max_n = 0, 0, 0
        for b in range(batch_num):
            if imgs[b].shape[1] > max_h: max_h = imgs[b].shape[1]
            if imgs[b].shape[2] > max_w: max_w = imgs[b].shape[2]
            if boxes[b].shape[0] > max_n: max_n = boxes[b].shape[0]
        imgs_t = torch.zeros(batch_num, 3, max_h, max_w)
        boxes_t = torch.zeros(batch_num, max_n, 4)
        labels_t = torch.zeros(batch_num, max_n).long()
        masks_t = torch.zeros(batch_num, max_n, max_h, max_w)
        segm_t = torch.zeros(batch_num, len(cfg.name_table), max_h, max_w)
        for b in range(batch_num):
            imgs_t[b, :, :imgs[b].shape[1], :imgs[b].shape[2]] = imgs[b]
            boxes_t[b, :boxes[b].shape[0]] = boxes[b]
            labels_t[b, :boxes[b].shape[0]] = labels[b]
            masks_t[b, :masks[b].shape[0], 
                :masks[b].shape[1], :masks[b].shape[2]] = masks[b]
            segm_t[b, :, :segm[b].shape[1], :segm[b].shape[2]] = segm[b]
        return {'imgs':imgs_t, 'locations':locations, 
                    'boxes':boxes_t, 'labels':labels_t, 'masks':masks_t, 
                    'segm':segm_t}
    
    def image_aspect_ratio(self, idx):
        image = self.coco.loadImgs(self.ids[idx])[0]
        return float(image['width']) / float(image['height'])
    
    def transform_inference_img(self, img_pil):
        img_pil, location, _, _ = resize_img(img_pil, self.min_size, cfg.eval.max_size)
        img = transforms.ToTensor()(img_pil)
        if self.normalize: img = self.normalizer(img)
        img = img.unsqueeze(0)
        return img, location
    
    def make_loader(self):
        batch_size = cfg.train.batch_size
        sampler = AspectRatioBasedSampler(self, batch_size, cfg.train.min_sizes, 
                        drop_last=cfg.train.get('drop_last', True))
        return data.DataLoader(self, batch_size=batch_size, 
                    sampler=sampler, num_workers=cfg.train.num_workers, 
                    collate_fn=self.collate_fn)


COLOR_TABLE = [
    (256,0,0), (0,256,0), (0,0,256), 
    (255,0,255), (255,106,106),(139,58,58),(205,51,51),
    (139,0,139),(139,0,0),(144,238,144),(0,139,139)
] * 100


def draw_bbox_text(drawObj, ymin, xmin, ymax, xmax, text, color, bd=1):
    drawObj.rectangle((xmin, ymin, xmax, ymin+bd), fill=color)
    drawObj.rectangle((xmin, ymax-bd, xmax, ymax), fill=color)
    drawObj.rectangle((xmin, ymin, xmin+bd, ymax), fill=color)
    drawObj.rectangle((xmax-bd, ymin, xmax, ymax), fill=color)
    strlen = len(text)
    drawObj.rectangle((xmin, ymin, xmin+strlen*6+5, ymin+12), fill=color)
    drawObj.text((xmin+3, ymin), text)


def show_instance(img, boxes, labels, masks=None, name_table=None, scores=None, 
                    file_name=None, matplotlib=False):
    '''
    img:      FloatTensor(3, H, W) or PIL
    boxes:    FloatTensor(N, 4)
    labels:   LongTensor(N) 0:bg
    masks:    FloatTensor(N, H, W) or None
    scores:   FloatTensor(N) or None
    file_name: 'out.bmp' or None
    '''
    if boxes.shape[0] == 0: return
    if not isinstance(img, Image.Image):
        img = transforms.ToPILImage()(img)
    # sort
    hw = boxes[:, 2:] - boxes[:, :2]
    area = hw[:, 0] * hw[:, 1] # N
    select = area.sort(descending=True)[1] # L(N)
    # blend mask
    if masks is not None:
        img_mask = torch.zeros(3, masks.shape[1], masks.shape[2])
        for i in range(select.shape[0]):
            i = int(select[i])
            m = masks[i] == 1 # H,W
            color = COLOR_TABLE[i]
            img_mask[0, m] = color[0]
            img_mask[1, m] = color[1]
            img_mask[2, m] = color[2]
        img_mask = img_mask / 257.0
        img_mask = transforms.ToPILImage()(img_mask)
        img = Image.blend(img, img_mask, 0.4)
    # draw bbox
    drawObj = ImageDraw.Draw(img)
    for i in range(select.shape[0]):
        i = int(select[i])
        lb = int(labels[i])
        if lb > 0: # fg
            box = boxes[i]
            if scores is None:
                draw_bbox_text(drawObj, box[0], box[1], box[2], box[3], name_table[lb],
                    color=COLOR_TABLE[i])
            else:
                str_score = str(float(scores[i]))[:5]
                str_out = name_table[lb] + ': ' + str_score
                draw_bbox_text(drawObj, box[0], box[1], box[2], box[3], str_out, 
                    color=COLOR_TABLE[i])
    if file_name is not None:
        img.save(file_name)
    else:
        if matplotlib:
            plt.imshow(img, aspect='equal')
            plt.show()
        else: img.show()


if __name__ == '__main__':
    dataset = Dataset(False, False)
    loader = dataset.make_loader()
    for data in loader:
        imgs, locations, boxes, labels, masks, segm = data['imgs'], \
            data['locations'], data['boxes'], data['labels'], data['masks'], data['segm']
        print('imgs:', imgs.shape)
        print('locations:', locations.shape)
        print('boxes:', boxes.shape)
        print('labels:', labels.shape)
        print('masks', masks.shape)
        print('segm', segm.shape)
        b = random.randint(0, cfg.train.batch_size-1)
        show_instance(imgs[b], boxes[b], labels[b], masks[b], 
            name_table=cfg.name_table)
        # segm = transforms.ToPILImage()(segm[0, 57])
        # segm.show()
        break