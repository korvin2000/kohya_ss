import threading
from typing import *
import torchvision.transforms as transforms
import numpy as np
import cv2
import torch


def fire_in_thread(f, *args, **kwargs):
    threading.Thread(target=f, args=args, kwargs=kwargs).start()


###########################
##### IDENTIFIER #####
###########################
def find_identifier(image_key: str):
    file_last_index = image_key.rfind("/")
    file_name = image_key[file_last_index+1:]
    return file_name.split('_')[0]


###########################
##### MULTIVIEW IMAGE #####
###########################
# 최종 이미지 크기
IMAGE_SIZE = 768

def custom_padding(img, expected_width, expected_height, image_type="numpy"):
    # image_type: numpy, tensor
    # input: image
    if image_type == "numpy":
        # img.shape: (height, width, channel)
        (img_height, img_width, _) = img.shape
        delta_width = max(expected_width - img_width, 0)
        delta_height = max(expected_height - img_height, 0)
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        result_image = 255 * np.ones((expected_height, expected_width, 3), dtype=np.uint8)
        start_x = pad_width
        end_x = pad_width + img_width
        start_y = pad_height
        end_y = pad_height + img_height
        result_image[start_y: end_y, start_x: end_x] = img
        return result_image
    elif image_type == "tensor":
        [img_width, img_height] = transforms.functional.get_image_size(img)
        delta_width = max(expected_width - img_width, 0)
        delta_height = max(expected_height - img_height, 0)
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        # (left, top, right, bottom)
        pad_transform = transforms.Pad((pad_width, pad_height, pad_width, pad_height), fill=255, padding_mode='constant')
        return pad_transform(img)

def make_multiview_image(real_image_list, image_type="numpy"):
    image_len = len(real_image_list)
    if image_len == 1:
        image = real_image_list[0]
    elif image_len == 2:
        image = make_multiview_image_2(real_image_list, image_type)
    elif image_len == 3:
        image = make_multiview_image_3(real_image_list, image_type)
    elif image_len == 4:
        image = make_multiview_image_4(real_image_list, image_type)
    else:
        assert False
 
    return image
    

def make_multiview_image_2(real_image_list, image_type="numpy"):
    # MAKE TO 1X2 MULTIVIEW
    # image_type: numpy, (tensor)
    # input: list[image]
    # output: image
    SMALL_IMAGE_SIZE = IMAGE_SIZE // 2
    if image_type == "numpy":
        image_list = []
        for real_image in real_image_list:
            image = real_image
            (img_height, img_width, _) = image.shape
            img_width, img_height = int(img_width), int(img_height)
            longer_size = max(img_width, img_height)
            image = custom_padding(image, longer_size, longer_size, image_type=image_type)

            # resize
            image = cv2.resize(image, dsize=(SMALL_IMAGE_SIZE, SMALL_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
            image_list.append(image)

        # 1*2 view
        total_image = 255 * np.ones((SMALL_IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        # [x,y] shape
        coordinate_list = [[0, 0], [SMALL_IMAGE_SIZE, 0]]
        for i, image_item in enumerate(image_list):
            start_x = coordinate_list[i][0]
            end_x = start_x + SMALL_IMAGE_SIZE
            start_y = coordinate_list[i][1]
            end_y = start_y + SMALL_IMAGE_SIZE
            total_image[start_y: end_y, start_x: end_x] = image_item
        return total_image
    

def make_multiview_image_3(real_image_list, image_type="numpy"):
    # MAKE TO 1-2 MULTIVIEW
    # image_type: numpy, (tensor)
    # input: list[image]
    # output: image
    SMALL_IMAGE_SIZE = IMAGE_SIZE // 2
    if image_type == "numpy":
        image_list = []
        for index, real_image in enumerate(real_image_list):
            image = real_image
            (img_height, img_width, _) = image.shape
            img_width, img_height = int(img_width), int(img_height)
            longer_size = max(img_width, img_height)
            if index == 0:
                image = custom_padding(image, longer_size, longer_size * 2, image_type=image_type)
                # resize
                image = cv2.resize(image, dsize=(SMALL_IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
                (img_height, img_width, _) = image.shape
            else:
                image = custom_padding(image, longer_size, longer_size, image_type=image_type)
                # resize
                image = cv2.resize(image, dsize=(SMALL_IMAGE_SIZE, SMALL_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
            image_list.append(image)

        # 1-2 view
        total_image = 255 * np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        coordinate_list = [[0, 0], [SMALL_IMAGE_SIZE, 0], [SMALL_IMAGE_SIZE, SMALL_IMAGE_SIZE]]
        size_list = [[SMALL_IMAGE_SIZE, IMAGE_SIZE], [SMALL_IMAGE_SIZE, SMALL_IMAGE_SIZE], [SMALL_IMAGE_SIZE, SMALL_IMAGE_SIZE]]
        for i, image_item in enumerate(image_list):
            start_x = coordinate_list[i][0]
            end_x = start_x + size_list[i][0]
            start_y = coordinate_list[i][1]
            end_y = start_y + size_list[i][1]
            total_image[start_y: end_y, start_x: end_x] = image_item
        return total_image


def make_multiview_image_4(real_image_list, image_type="numpy"):
    # MAKE TO 2X2 MULTIVIEW
    # image_type: numpy, tensor
    # input: list[image]
    # output: image
    SMALL_IMAGE_SIZE = IMAGE_SIZE // 2
    if image_type == "numpy":
        image_list = []
        for real_image in real_image_list:
            image = real_image
            (img_height, img_width, _) = image.shape
            img_width, img_height = int(img_width), int(img_height)
            longer_size = max(img_width, img_height)
            image = custom_padding(image, longer_size, longer_size, image_type=image_type)

            # resize
            image = cv2.resize(image, dsize=(SMALL_IMAGE_SIZE, SMALL_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
            image_list.append(image)

        # 2*2 view
        total_image = 255 * np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        coordinate_list = [[0, 0], [SMALL_IMAGE_SIZE, 0], [0, SMALL_IMAGE_SIZE], [SMALL_IMAGE_SIZE, SMALL_IMAGE_SIZE]]
        for i, image_item in enumerate(image_list):
            start_x = coordinate_list[i][0]
            end_x = start_x + SMALL_IMAGE_SIZE
            start_y = coordinate_list[i][1]
            end_y = start_y + SMALL_IMAGE_SIZE
            total_image[start_y: end_y, start_x: end_x] = image_item
        return total_image
    elif image_type == "tensor":
        image_list = []
        for real_image in real_image_list:
            image = real_image

            [img_width, img_height] = transforms.functional.get_image_size(image)
            img_width, img_height = int(img_width), int(img_height)
            longer_size = max(img_width, img_height)
            image = custom_padding(image, longer_size, longer_size, image_type=image_type)

            # resize
            resize_transform = transforms.Resize((SMALL_IMAGE_SIZE, SMALL_IMAGE_SIZE))
            image = resize_transform(image)
            image_list.append(image)

        # 2*2 view
        # cat tensor
        sum_image0 = torch.cat(image_list[:2], dim=1)
        sum_image1 = torch.cat(image_list[2:], dim=1)
        total_image = torch.cat([sum_image0, sum_image1], dim=2)
        return total_image
