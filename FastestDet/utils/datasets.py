import os
import cv2
import numpy as np

import torch
import random


def random_crop(image, boxes):
    height, width, _ = image.shape
    # random crop imgage
    cw, ch = random.randint(int(width * 0.75), width), random.randint(int(height * 0.75), height)
    cx, cy = random.randint(0, width - cw), random.randint(0, height - ch)

    roi = image[cy:cy + ch, cx:cx + cw]
    roi_h, roi_w, _ = roi.shape

    output = []
    for box in boxes:
        index, category = box[0], box[1]
        bx, by = box[2] * width, box[3] * height
        bw, bh = box[4] * width, box[5] * height

        bx, by = (bx - cx) / roi_w, (by - cy) / roi_h
        bw, bh = bw / roi_w, bh / roi_h

        output.append([index, category, bx, by, bw, bh])

    output = np.array(output, dtype=float)

    return roi, output


def random_narrow(image, boxes):
    height, width, _ = image.shape
    # random narrow
    cw, ch = random.randint(width, int(width * 1.25)), random.randint(height, int(height * 1.25))
    cx, cy = random.randint(0, cw - width), random.randint(0, ch - height)

    background = np.ones((ch, cw, 3), np.uint8) * 128
    background[cy:cy + height, cx:cx + width] = image

    output = []
    for box in boxes:
        index, category = box[0], box[1]
        bx, by = box[2] * width, box[3] * height
        bw, bh = box[4] * width, box[5] * height

        bx, by = (bx + cx) / cw, (by + cy) / ch
        bw, bh = bw / cw, bh / ch

        output.append([index, category, bx, by, bw, bh])

    output = np.array(output, dtype=float)

    return background, output


def collate_fn(batch):
    img, label = zip(*batch)
    for i, l in enumerate(label):
        if l.shape[0] > 0:
            l[:, 0] = i
    return torch.stack(img), torch.cat(label, 0)


class TensorDataset():
    def __init__(self, path, img_width, img_height, aug=False):
        assert os.path.exists(path), "%s文件路径错误或不存在" % path

        self.aug = aug
        self.img_path = path
        self.label_path = path.replace("images", "labels")
        self.img_width = img_width
        self.img_height = img_height
        self.img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif']
        self.image_paths = []
        self.label_paths = []

        # 遍历图片文件夹
        for img_name in os.listdir(self.img_path):
            base, img_type = os.path.splitext(img_name)

            if img_type not in self.img_formats:
                raise Exception("img type error:%s" % img_type)

            img_path = os.path.join(self.img_path, img_name)
            label_name = f"{base}.txt"
            label_path = os.path.join(self.label_path, label_name)
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"缺少标签文件: {label_path}")

            self.image_paths.append(img_path)
            self.label_paths.append(label_path)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label_path = self.label_paths[index]

        # 加载图片
        img = cv2.imread(img_path)

        # 加载label文件
        if os.path.exists(label_path):
            label = []
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    l = line.strip().split(" ")
                    label.append([0, l[0], l[1], l[2], l[3], l[4]])
            label = np.array(label, dtype=np.float32)

            if label.shape[0]:
                assert label.shape[1] == 6, '> 5 label columns: %s' % label_path
                # assert (label >= 0).all(), 'negative labels: %s'%label_path
                # assert (label[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s'%label_path
        else:
            raise Exception("%s is not exist" % label_path)

            # 是否进行数据增强
        if self.aug:
            if random.randint(1, 10) % 2 == 0:
                img, label = random_narrow(img, label)
            else:
                img, label = random_crop(img, label)

        img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)

        # debug
        # for box in label:
        #     bx, by, bw, bh = box[2], box[3], box[4], box[5]
        #     x1, y1 = int((bx - 0.5 * bw) * self.img_width), int((by - 0.5 * bh) * self.img_height)
        #     x2, y2 = int((bx + 0.5 * bw) * self.img_width), int((by + 0.5 * bh) * self.img_height)
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.imwrite("debug.jpg", img)

        img = img.transpose(2, 0, 1)

        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    data = TensorDataset("datasets/apricot/images/train", img_width=640, img_height=640)
    img, label = data.__getitem__(0)
    print(img.shape)
    print(label.shape)
