
import torch
import random
import cv2
import glob
import numpy as np
from PIL import Image, ImageDraw
import os
import math
import matplotlib.pyplot as plt

from .utils import xywhn2xyxy
from .utils import random_perspective
from .utils import augment_hsv


hyp = {
    'mosaic': 1.0,
    
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    
    'fliplr': 0.5,
    
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4
}


class DatasetYolov5(torch.utils.data.Dataset):

    def __init__(self, path, img_size=640, augment=True, hyp=hyp):
        '''
        '''
        self.path = path
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        
        self.mosaic = self.augment
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        
        self.img_files, self.lab_files = self.load_files(path)
        self.n = len(self.img_files)
        self.indices = range(self.n)

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __len__(self, ):
        return self.n
    
    def __getitem__(self, idx):
        
        idx = self.indices[idx]
        # mosaic = self.mosaic and random.random() < self.hyp['mosaic']
        
        if random.random() < self.hyp['mosaic']:
            # [h w c]
            # [cls, x, y, x, y]
            img, labels = self.load_mosaic(idx)

            # cls x y x y -> cls x y w h

            labels[:, [3, 4]] = labels[:, [3, 4]] - labels[:, [1, 2]] 
            labels[:, [1, 2]] = labels[:, [1, 2]] + labels[:, [3, 4]] / 2.
            labels[:, 1:] /= self.img_size

            # if random.random() < self.hyp['fliplr']:
            #     img4 = np.fliplr(img4)
            #     if len(labels):
            #         labels[:, 1] = 1. - labels[:, 1]

            labels_out = np.zeros((labels.shape[0], 6))
            labels_out[:, 1:] = labels

            img_out = (img / 255. - self.mean) / self.std
            img_out = img_out[np.newaxis, ...].transpose(0, 3, 1, 2)
            img_out = np.ascontiguousarray(img_out)

        else:
            pass

        return img_out, labels_out
            
 
    @staticmethod
    def collate_fn(batch):
        '''add image index
        '''
        imgs, labels = zip(*batch)
        for i, lab in enumerate(labels):
            lab[:, 0] = i
        imgs = np.concatenate(imgs, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        imgs = torch.tensor(imgs, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return imgs, labels


    def load_files(self, path, label=True):
        # load image files
        files = []
        for p in path if isinstance(path, list) else [path, ]:
            files.extend(glob.glob(os.path.join(p, '*.jpg')))
        
        img_files = [os.path.abspath(p) for p in sorted(files)]
        lab_files = [p.replace('/images/', '/labels/').replace('.jpg', '.txt') for p in img_files]
        
        _img_files = []
        _lab_files = []
        
        for im, lab in zip(img_files, lab_files):
            if os.path.exists(im) and os.path.exists(lab):
                _img_files.append(im)
                _lab_files.append(lab)
        
        # print(len(_img_files), len(_lab_files))
        # print(_img_files[0], )
        # print(_lab_files[0], )
        
        return _img_files, _lab_files
            
        
    def load_mosaic(self, idx):
        '''load mosaic
        '''
        labels4 = []
        s = self.img_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  
        indices = [idx] + [self.indices[random.randint(0, self.n - 1)] for _ in range(3)]

        for i, idx in enumerate(indices):
            
            img, _, (h, w) = self.load_image(idx)
            
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b] 
            
            padw = x1a - x1b
            padh = y1a - y1b

            labels = self.load_label(idx)
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)
                labels4.append(labels)
            
        labels4 = np.concatenate(labels4, 0)
        for x in labels4[:, 1:]:
            np.clip(x, 0, 2 * s - 1, out=x)
        
        # Augment
        if True:
            img4, labels4 = random_perspective(img=img4, targets=labels4,
                                                degrees=self.hyp['degrees'],
                                                translate=self.hyp['translate'],
                                                scale=self.hyp['scale'],
                                                shear=self.hyp['shear'],
                                                perspective=self.hyp['perspective'],
                                                border=self.mosaic_border)  # border to remove

        if self.hyp['hsv_h'] or self.hyp['hsv_s'] or self.hyp['hsv_v']:
            augment_hsv(img4, self.hyp['hsv_h'], self.hyp['hsv_s'], self.hyp['hsv_v'])

        for x in labels4[:, 1:]:
            np.clip(x, 0, s-1, out=x)

        return img4, labels4
        
            
    def load_image(self, index):
        '''load image
        '''
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            
        return img[:,:,::-1], (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
        
        
    def load_label(self, idx):
        '''load label
        '''
        with open(self.lab_files[idx], 'r') as f:
            l = [x.split() for x in f.read().strip().splitlines()]
        return np.array(l, dtype=np.float32)
    
    

    def show(self, idx=None, img=None, bbox=None):
        
        from PIL import Image, ImageDraw
        
        if idx is not None:
            img, bbox = self.load_mosaic(idx)

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        
        for _, b in enumerate(bbox):
            draw.rectangle(tuple(b[1:]), outline='red', width=2)
        
        # axes = plt.subplots(1, 2, figsize=(10, 5))
        # plt.imshow(img)
        
        return img
    



