import os
import cv2
import numpy as np
import argparse

from image_recorder import ImageRecoder


def capture_one_object(Image_Recorder, output_dir):
    print("press 's' to save")
    count = 0

    img_per_obj = 8
    image_list = []
    while True:
        image_dict = Image_Recorder.get_images()
        color_image = image_dict['main_cam']
        depth_image = image_dict['main_cam_depth']


        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1)

        if key == ord('s'):
            count += 1
            image_list.append(color_image)
            # 保存图像

        if count == img_per_obj:
            break

    print('collect done, saving image')
    for i in range(len(image_list)):
        rgb_filename = os.path.join(output_dir, f'{i}.png')
        cv2.imwrite(rgb_filename, image_list[i])


def main(output_dir):
    Image_Recorder = ImageRecoder()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ask the user to input the object name
    while True:
        obj_name = input("Enter the object name: ")

        obj_dir = os.path.join(output_dir, obj_name)
        if not os.path.exists(obj_dir):
            os.makedirs(obj_dir)

        capture_one_object(Image_Recorder, obj_dir)

        flag = input("Enter 'c' to continue, and any other key to quit:")
        if flag == 'c':
            continue
        else:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', action='store', type=str, help='Output directory.', default=None, required=True)

    args = vars(parser.parse_args())
    output_dir = args['output_dir']
    main(output_dir)

