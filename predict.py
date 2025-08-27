#-------------------------------------#
#   运行predict.py可以生成图片
#   生成1x1的图片和5x5的图片
#-------------------------------------#
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ddpm import Diffusion_stage1, Diffusion_stage2
import cv2
import numpy as np
import torch
from utils.utils import get_lr, show_pred, show_result
from PIL import Image
import time
from natsort import natsorted
from skimage.metrics import peak_signal_noise_ratio as psnr

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

input_shape = (64,64)

lower_room = np.array([10,10,10])  # living_room
upper_room = np.array([255,255,255])
lower_pink = np.array([0, 20, 225])
upper_pink = np.array([5, 30, 237])
lower_pink_BGR = np.array([210, 210, 230])
upper_pink_BGR = np.array([220, 220, 240])
lower_gray = np.array([20, 10, 240])
upper_gray = np.array([30, 20, 250])
lower_yellow = np.array([22, 78, 245])
upper_yellow = np.array([32, 88, 255])
lower_blue = np.array([97, 43, 245])
upper_blue = np.array([107, 53, 255])
lower_green = np.array([28, 91, 210])
upper_green = np.array([38, 101, 220])
def num_room(pic, low, upper):
    mask_grey = cv2.inRange(pic, low, upper)
    contours, _ = cv2.findContours(mask_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 10:
            n += 1
    return n

def postprocess_output(x):
    x *= 0.5
    x += 0.5
    x *= 255
    return x

def read_image(img_name):
    image = Image.open(os.path.join(img_name)).convert('RGB')
    image = np.array(image)
    image = image.astype(np.float64) / 255.0
    return image

def read_img_stage2(path_layout, predict_num):
    for i in range(predict_num):
        source_layout = cv2.imread(path_layout[i])
        source_layout = cv2.resize(source_layout, (64, 64), interpolation=cv2.INTER_AREA)
        source_layout = cv2.cvtColor(source_layout, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source_layout = source_layout.astype(np.float32) / 255.0
        source_layout = np.expand_dims(source_layout.transpose(2, 0, 1), axis=0)
        #source = np.concatenate((source_layout), axis=1)
        source = source_layout

        if i == 0:
            sources = source
        else:
            sources = np.concatenate((sources, source), axis=0)
    sources = torch.from_numpy(sources).cuda()
    return sources

def read_img_stage1(path_balcony, path_bedroom, path_kitchen, path_living_room, path_toilet, predict_num):
    for i in range(predict_num):
        source_living_room = cv2.imread(path_living_room)
        num_livingroom = num_room(source_living_room, lower_room, upper_room)
        source_living_room = cv2.resize(source_living_room, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA)
        source_balcony = cv2.imread(path_balcony)
        num_balcony = num_room(source_balcony, lower_room, upper_room)
        source_balcony = cv2.resize(source_balcony, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA)
        source_kitchen = cv2.imread(path_kitchen)
        num_kitchen = num_room(source_kitchen, lower_room, upper_room)
        source_kitchen = cv2.resize(source_kitchen, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA)
        source_bedroom = cv2.imread(path_bedroom)
        num_bedroom = num_room(source_bedroom, lower_room, upper_room)
        source_bedroom = cv2.resize(source_bedroom, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA)
        source_toilet = cv2.imread(path_toilet)
        num_toilet = num_room(source_toilet, lower_room, upper_room)
        source_toilet = cv2.resize(source_toilet, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA)
        num_rooms = [num_livingroom, num_bedroom, num_toilet, num_kitchen, num_balcony]

        # Do not forget that OpenCV read images in BGR order.
        source_living_room = cv2.cvtColor(source_living_room, cv2.COLOR_BGR2RGB)
        source_balcony = cv2.cvtColor(source_balcony, cv2.COLOR_BGR2RGB)
        source_kitchen = cv2.cvtColor(source_kitchen, cv2.COLOR_BGR2RGB)
        source_bedroom = cv2.cvtColor(source_bedroom, cv2.COLOR_BGR2RGB)
        source_toilet = cv2.cvtColor(source_toilet, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source_living_room = source_living_room.astype(np.float32) / 255.0
        source_living_room = source_living_room.transpose(2, 0, 1)
        source_balcony = source_balcony.astype(np.float32) / 255.0
        source_balcony = source_balcony.transpose(2, 0, 1)
        source_kitchen = source_kitchen.astype(np.float32) / 255.0
        source_kitchen = source_kitchen.transpose(2, 0, 1)
        source_bedroom = source_bedroom.astype(np.float32) / 255.0
        source_bedroom = source_bedroom.transpose(2, 0, 1)
        source_toilet = source_toilet.astype(np.float32) / 255.0
        source_toilet = source_toilet.transpose(2, 0, 1)
        source = np.concatenate((source_living_room, source_balcony, source_kitchen, source_bedroom, source_toilet), axis=0)

        source = np.expand_dims(source, axis=0)
        #source = torch.from_numpy(source).cuda()
        if i == 0:
            sources = source
        else:
            sources = np.concatenate((sources, source), axis=0)
    sources = torch.from_numpy(sources).cuda()
    return sources, num_rooms

def remove_noise(img):
    lower = np.array([0, 245, 245], dtype=np.uint8)
    upper = np.array([10, 255, 255], dtype=np.uint8)
    mask = np.all((img >= lower) & (img <= upper), axis=-1)
    img[mask] = [255, 255, 255]

    lower = np.array([245, 245, 0], dtype=np.uint8)
    upper = np.array([255, 255, 10], dtype=np.uint8)
    mask = np.all((img >= lower) & (img <= upper), axis=-1)
    img[mask] = [255, 255, 255]

    lower = np.array([245, 0, 245], dtype=np.uint8)
    upper = np.array([255, 10, 255], dtype=np.uint8)
    mask = np.all((img >= lower) & (img <= upper), axis=-1)
    img[mask] = [255, 255, 255]

    return img


if __name__ == "__main__":
    #--------------Stage 1-----------------
    ddpm_stage1 = Diffusion_stage1()
    path_balcony = f'./test/stage1_input/0_balcony.png'
    path_bedroom = f'./test/stage1_input/0_bedroom.png'
    path_kitchen = f'./test/stage1_input/0_kitchen.png'
    path_living_room = f'./test/stage1_input/1_living_room.png'
    path_toilet = f'./test/stage1_input/0_toilet.png'
    save_path_stage1 = f'./test/stage1_output/'
    predict_num = 8
    Room_Judgment = True
    # --------------Stage 2-----------------
    path_base = './test/stage1_output'
    save_path_stage2 = f'./test/stage2_output/'
    ddpm_stage2 = Diffusion_stage2()
    contour_index = 0
    path_gai_base = './test/stage1_output'
    test_img = './test/stage2_psnr.png'

    # --------------Stage 1 Start-----------------
    start_time = time.time()
    souce, num_rooms = read_img_stage1(path_balcony, path_bedroom, path_kitchen, path_living_room, path_toilet, predict_num)
    test_images_output = show_pred(predict_num, ddpm_stage1.net, "cuda", source=souce)
    for img_index in range(predict_num):
        test_images = postprocess_output(test_images_output[img_index].cpu().data.numpy().transpose(1, 2, 0))
        test_images = np.uint8(test_images)
        test_images[test_images == 255] = 0
        test_images_origin = test_images.copy()
        if Room_Judgment:
            test_images = cv2.cvtColor(test_images, cv2.COLOR_BGR2RGB)
            test_images_HSV = cv2.cvtColor(test_images, cv2.COLOR_BGR2HSV)
            num_livingroom = num_room(test_images_HSV, lower_gray, upper_gray)
            num_bedroom = num_room(test_images_HSV, lower_yellow, upper_yellow)
            num_toilet = num_room(test_images_HSV, lower_blue, upper_blue)
            num_kitchen = num_room(test_images_HSV, lower_pink, upper_pink)
            num_balcony = num_room(test_images_HSV, lower_green, upper_green)

            if num_livingroom == num_rooms[0] and num_bedroom == num_rooms[1] and num_toilet == num_rooms[
                2] and num_kitchen == num_rooms[3] and num_balcony == num_rooms[4]:
                Image.fromarray(test_images_origin).save(f'{save_path_stage1}{img_index}.png')
                #cv2.imwrite(save_path_1x1, test_images)
                #break
        else:
            Image.fromarray(test_images_origin).save(f'{save_path_stage1}{img_index}.png')
    print("Stage1 Generate image Done")

    # --------------Stage 2 Start-----------------
    img_all = natsorted(os.listdir(path_gai_base))
    path_layout = []
    for i in range(predict_num):
        path_layout.append(f'{path_base}/{img_all[i]}')
    souce = read_img_stage2(path_layout, predict_num)
    img_generation = show_pred(predict_num, ddpm_stage2.net, "cuda", source=souce)
    for j in range(predict_num):
        if not os.path.exists(save_path_stage2):
            os.mkdir(save_path_stage2)
        save_path_1x1 = f'{save_path_stage2}/{j}.png'
        test_images = postprocess_output(img_generation[j].cpu().data.numpy().transpose(1, 2, 0))
        test_images = np.uint8(test_images)
        test_images_origin = test_images.copy()
        test_images_origin = remove_noise(test_images_origin)
        test_images_HSV = cv2.cvtColor(test_images, cv2.COLOR_BGR2HSV)
        Image.fromarray(test_images_origin).save(f'{save_path_1x1}')
    psnr_image = read_image(f'{test_img}')
    for index_stage2 in range(predict_num):
        test_img_path = f'{save_path_stage2}/{index_stage2}.png'
        sample_image = read_image(test_img_path)
        psnr_value = psnr(psnr_image, sample_image, data_range=255)
        if psnr_value < 60:
            os.remove(test_img_path)
    print("Stage2 Generate image Done")
