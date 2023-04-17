
import cv2
import glob
import argparse

count = 0

# print(len(glob.glob("../WaymoCOCO/data/waymococo_full/trainFrames/train_*_00000_camera1.jpg")))
# print(len(glob.glob("../WaymoCOCO/data/waymococo_full/valFrames/val_*_00000_camera1.jpg")))
# print(len(glob.glob("../WaymoCOCO/data/waymococo_full/testFrames/test_*_00000_camera1.jpg")))


def set_video(video_name):
    fps = 10
    video_width = 1920
    video_height = 1280
    size = (video_width, video_height)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video = cv2.VideoWriter(video_name, fourcc, fps, size)
    return video


parser = argparse.ArgumentParser(description='')
parser.add_argument('--MODE', default='train',
                    help='MODE can be all, train, val, and test')

## Parse arguments
args = parser.parse_args()


if args.MODE == 'all' or args.MODE == 'train':
    for vid_num in range(len(glob.glob("../WaymoCOCO/data/waymococo_full/trainFrames/train_*_00000_camera1.jpg"))):
        vid_name = '../waymo_clips/train/train_{:05d}.mp4'.format(vid_num)
        print (vid_name)
        video = set_video(vid_name)
        # print(len(glob.glob("../WaymoCOCO/data/waymococo_full/trainFrames/train_{:05d}_*_camera1.jpg".format(vid_num))))
        # break
        for frame_num in range(len(glob.glob("../WaymoCOCO/data/waymococo_full/trainFrames/train_{:05d}_*_camera1.jpg".format(vid_num)))):
            img = cv2.imread("../WaymoCOCO/data/waymococo_full/trainFrames/train_{:05d}_{:05d}_camera1.jpg".format(vid_num,frame_num))
            video.write(img)
        video.release()
        # break


if args.MODE == 'all' or args.MODE == 'val':
    for vid_num in range(len(glob.glob("../WaymoCOCO/data/waymococo_full/valFrames/val_*_00000_camera1.jpg"))):
        vid_name = '../waymo_clips/val/val_{:05d}.mp4'.format(vid_num)
        print (vid_name)
        video = set_video(vid_name)
        # print(len(glob.glob("../WaymoCOCO/data/waymococo_full/valFrames/val_{:05d}_*_camera1.jpg".format(vid_num))))
        # break
        for frame_num in range(len(glob.glob("../WaymoCOCO/data/waymococo_full/valFrames/val_{:05d}_*_camera1.jpg".format(vid_num)))):
            img = cv2.imread("../WaymoCOCO/data/waymococo_full/valFrames/val_{:05d}_{:05d}_camera1.jpg".format(vid_num,frame_num))
            video.write(img)
        video.release()
        # break


if args.MODE == 'all' or args.MODE == 'test':
    for vid_num in range(len(glob.glob("../WaymoCOCO/data/waymococo_full/testFrames/test_*_00000_camera1.jpg"))):
        vid_name = '../waymo_clips/test/test_{:05d}.mp4'.format(vid_num)
        print (vid_name)
        video = set_video(vid_name)
        # print(len(glob.glob("../WaymoCOCO/data/waymococo_full/testFrames/test_{:05d}_*_camera1.jpg".format(vid_num))))
        # break
        for frame_num in range(len(glob.glob("../WaymoCOCO/data/waymococo_full/testFrames/test_{:05d}_*_camera1.jpg".format(vid_num)))):
            img = cv2.imread("../WaymoCOCO/data/waymococo_full/testFrames/test_{:05d}_{:05d}_camera1.jpg".format(vid_num,frame_num))
            video.write(img)
        video.release()
        # break
            
    








