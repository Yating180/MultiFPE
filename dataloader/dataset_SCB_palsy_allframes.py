import os.path
from numpy.random import randint
from torch.utils import data
import glob
import os
from dataloader.video_transform import *
import numpy as np
import torchvision.transforms as transforms


def parse_landmarks(lines):
    landmarks, left_iris, right_iris = [], [], []
    sections = {'# 68 facial Landmarks [x,y] {': landmarks,
                '# Left iris [x,y,r] {': left_iris,
                '# Right iris [x,y,r] {': right_iris}

    current_section = None
    for line in lines:
        line = line.strip()
        if line in sections:
            current_section = sections[line]
            continue
        if line == "# }":
            current_section = None
            continue
        if current_section is not None:
            current_section.append(tuple(map(int, line.split(','))) if current_section == landmarks else int(line))

    return landmarks, left_iris, right_iris


def line_intersection(A, C, B, D):
    x1, y1 = A[0], A[1]
    x2, y2 = C[0], C[1]
    x3, y3 = B[0], B[1]
    x4, y4 = D[0], D[1]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if torch.isclose(denom, torch.tensor(0.0, dtype=torch.float32)):
        if torch.isclose((y2 - y1) * (x3 - x1), (y3 - y1) * (x2 - x1), atol=1e-6):
            midpoint_x = (x1 + x2 + x3 + x4) / 4
            midpoint_y = (y1 + y2 + y3 + y4) / 4
            return torch.stack([torch.round(midpoint_x), torch.round(midpoint_y)])
        else:
            midpoint_1_x, midpoint_1_y = (x1 + x2) / 2, (y1 + y2) / 2
            midpoint_2_x, midpoint_2_y = (x3 + x4) / 2, (y3 + y4) / 2
            center_x = (midpoint_1_x + midpoint_2_x) / 2
            center_y = (midpoint_1_y + midpoint_2_y) / 2
            return torch.stack([torch.round(center_x), torch.round(center_y)])

    x_num = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    y_num = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)

    x_intersection = x_num / denom
    y_intersection = y_num / denom

    x_intersection_int = torch.round(x_intersection)
    y_intersection_int = torch.round(y_intersection)

    intersection_tensor = torch.stack([x_intersection_int, y_intersection_int])

    return intersection_tensor


def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    landmarks, left_iris, right_iris = parse_landmarks(lines)

    landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)
    left_iris_tensor = line_intersection(landmarks_tensor[37], landmarks_tensor[40], landmarks_tensor[38], landmarks_tensor[41])
    right_iris_tensor = line_intersection(landmarks_tensor[43], landmarks_tensor[46], landmarks_tensor[44], landmarks_tensor[47])

    return landmarks_tensor, left_iris_tensor, right_iris_tensor


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[4]) - 1


class VideoDataset(data.Dataset):
    def __init__(self, list_file, mode, image_transform,static_image_transform, image_size):
        self.list_file = list_file
        self.image_transform = image_transform
        self.static_image_transform = static_image_transform
        self.image_size = image_size
        self.mode = mode
        self._parse_list()

    def _parse_list(self):
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        self.video_list = [VideoRecord(item) for item in tmp]
        print(f'video number: {len(self.video_list)}')

    def __getitem__(self, index):
        record = self.video_list[index]
        return self.get(record)

    def get(self, record):
        video_name = record.path.split('/')[-1]

        video_frames_path = glob.glob(os.path.join(record.path, '*.jpg'))
        video_frames_path.sort()

        assert len(video_frames_path) == 32, f"The number of frames in video {video_name} is not equal to 32"

        images = list()
        static_image = list()
        landmarks_list = list()
        left_iris_list = list()
        right_iris_list = list()
        static_landmarks_list = list()
        static_left_iris_list = list()
        static_right_iris_list = list()

        static_frame_path = record.path.replace('frames', 'static_frame')
        static_frame_files = glob.glob(os.path.join(static_frame_path, '*.jpg'))
        static_frame = Image.open(static_frame_files[0]).convert('RGB')
        static_image.append(static_frame)

        static_txts_path = record.path.replace('frames', 'static_txts')
        static_txts_files = glob.glob(os.path.join(static_txts_path, '*.txt'))
        static_landmarks, static_left_iris, static_right_iris = load_data(static_txts_files[0])
        static_landmarks_list.append(static_landmarks)
        static_left_iris_list.append(static_left_iris)
        static_right_iris_list.append(static_right_iris)

        for p in range(32):
            frame_name = os.path.basename(video_frames_path[p]).split('.')[0]
            img = Image.open(video_frames_path[p]).convert('RGB')
            images.append(img)
            txt_path = os.path.join(record.path.replace('frames', 'txts'), frame_name + '.txt')
            landmarks_tensor, left_iris_tensor, right_iris_tensor = load_data(txt_path)
            landmarks_list.append(landmarks_tensor)
            left_iris_list.append(left_iris_tensor)
            right_iris_list.append(right_iris_tensor)

        if self.image_transform:
            images = self.image_transform(images)

        if self.static_image_transform:
            static_image = [self.static_image_transform(img) for img in static_image]

        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))
        static_image = torch.stack(static_image)
        static_image = torch.reshape(static_image, (-1, 3, self.image_size, self.image_size))

        landmarks_tensor = torch.stack(landmarks_list)
        left_iris_tensor = torch.stack(left_iris_list)
        right_iris_tensor = torch.stack(right_iris_list)
        static_landmarks_tensor = torch.stack(static_landmarks_list)
        static_left_iris_tensor = torch.stack(static_left_iris_list)
        static_right_iris_tensor = torch.stack(static_right_iris_list)

        return images, static_image, landmarks_tensor, left_iris_tensor, right_iris_tensor, static_landmarks_tensor, static_left_iris_tensor, static_right_iris_tensor, record.label

    def __len__(self):
        return len(self.video_list)


def train_data_loader(data_set):
    image_size = 112
    train_image_transforms = torchvision.transforms.Compose([GroupRandomSizedCrop(image_size),
                                                             GroupRandomHorizontalFlip(),
                                                             Stack(),
                                                             ToTorchFormatTensor()])
    static_image_transforms = transforms.Compose([transforms.Resize((image_size, image_size)),
                                                  transforms.ToTensor()])

    train_data = VideoDataset(list_file=f"./annotation/{data_set}_train.txt",
                              mode='train',
                              image_transform=train_image_transforms,
                              static_image_transform=static_image_transforms,
                              image_size=image_size)
    return train_data


def test_data_loader(data_set):
    image_size = 112
    test_image_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                           Stack(),
                                                           ToTorchFormatTensor()])

    static_image_transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                                 transforms.ToTensor()])

    test_data = VideoDataset(list_file=f"./annotation/{data_set}_test.txt",
                             mode='test',
                             image_transform=test_image_transform,
                             static_image_transform=static_image_transform,
                             image_size=image_size)
    return test_data

