import cv2
from moviepy.editor import VideoFileClip
import numpy as np
import autosp

videos = ['./datasets/Dense/jan28.avi',
          './datasets/Sunny/april21.avi',
          './datasets/Urban/march9.avi']

pos_annotations = ['./datasets/Dense/pos_annot.dat',
          './datasets/Sunny/pos_annot.dat',
          './datasets/Urban/pos_annot.dat']

output_videos = ['./datasets/Dense/jan28_outlined.mp4',
          './datasets/Sunny/april21_outlined.mp4',
          './datasets/Urban/march9_outlined.mp4']

output_images = ['./datasets/Dense/jan28_{}_{}.png',
          './datasets/Sunny/april21_{}_{}.png',
          './datasets/Urban/march9_{}_{}.png']

output_non_car_images = './datasets/Non_Vehicles/{}.png'

index = 0
index_non_car = 0

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=3):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def parse_annotation_data(filepath):
    annotated_data = []
    with open(filepath, 'r') as f:
        for line in f:
            data = line.split('\t')
            num_cars = int(data[1])
            cars = []
            for i in range(num_cars):
                car_coord = map(int, data[i + 2].split())
                cars.append(car_coord)
            annotated_data.append(cars)
    return annotated_data


def process_image(img, annotated_data, output_images):
    global index

    imcopy = np.copy(img)

    if index >= len(annotated_data):
        return img

    for i, car in enumerate(annotated_data[index]):
        startx, starty = car[0], car[1]
        endx, endy = car[0] + car[2], car[1] + car[3]

        cropped = imcopy[starty:endy, startx:endx]
        if cropped.shape[0] > 30 and cropped.shape[1] > 30:
            # Only take in images with more than 30 x 30 pixels
            cropped = cv2.resize(cropped, (64, 64))
            cv2.imwrite(output_images.format(index, i), cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

        # img = draw_boxes(img, [((startx, starty), (endx, endy))])

    index += 1
    if index % 30 == 0:
        process_image_for_non_car(img, annotated_data, output_non_car_images)

    return img

def process_image_for_non_car(img, annotated_data, output_images):
    global index_non_car
    global index

    imcopy = np.copy(img)
    if index >= len(annotated_data):
        return

    car_windows = []
    for i, car in enumerate(annotated_data[index]):
        startx, starty = car[0], car[1]
        endx, endy = car[0] + car[2], car[1] + car[3]
        car_windows.append(((startx,starty),(endx,endy)))

    all_windows = autosp.sliding_windows(img,(None,None),(None,None),(64,64),(0.5,0.5))
    for i in xrange(0, len(all_windows)):
        window = all_windows[i]

        is_car = False
        for car_window in car_windows:
            if rectangles_intersect(window, car_window):
                is_car = True
        if not is_car:
            startx = window[0][0]
            starty = window[0][1]
            endx = window[1][0]
            endy = window[1][1]

            cropped = imcopy[starty:endy, startx:endx]
            cv2.imwrite(output_images.format(index_non_car), cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

            index_non_car += 1



def rectangles_intersect(rectA, rectB):
    a = rectA[0][0]
    b = rectA[0][1]
    c = rectA[1][0]
    d = rectA[1][1]

    e = rectB[0][0]
    f = rectB[0][1]
    g = rectB[1][0]
    h = rectB[1][1]

    return not (e > c or f > d or g < a or h < b)

if __name__ == "__main__":

    for i in range(len(videos)):
        annotated_data = parse_annotation_data(pos_annotations[i])

        video = VideoFileClip(videos[i])
        annot_vid = video.fl_image(lambda x: process_image(x, annotated_data, output_images[i]))
        annot_vid.write_videofile(output_videos[i], audio=False)
        index = 0
