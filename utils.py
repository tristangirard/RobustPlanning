import os
import cv2


def file_exists(file_name):
    return os.path.isfile(file_name)


def dir_exists(dir_name):
    return os.path.isdir(dir_name)


def create_directory(directory_name):
    try:
        os.mkdir(directory_name)
    except FileExistsError:
        pass


def images_to_video(file_name):
    dir_path = 'recordings'
    create_directory(dir_path)
    i = 0
    images = []
    while file_exists('recordings/{}-{}.png'.format(file_name, i)):
        images.append(cv2.imread('recordings/{}-{}.png'.format(file_name, i)))
        os.remove('recordings/{}-{}.png'.format(file_name, i))
        i += 1
    height, width, channels = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('recordings/{}.mp4'.format(file_name), fourcc, 2.0, (width, height))
    images += [images[-1]] * 3
    for image in images:
        out.write(image)
    out.release()
    cv2.destroyAllWindows()
