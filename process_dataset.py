import cv2
from moviepy.editor import VideoFileClip
import numpy as np

videos = ['./datasets/Dense/jan28.avi',
          './datasets/Sunny/april21.avi',
          './datasets/Urban/march9.avi']

pos_annotations = ['./datasets/Dense/pos_annot.dat',
          './datasets/Sunny/pos_annot.dat',
          './datasets/Urban/pos_annot.dat']

output_videos = ['./datasets/Dense/jan28_outlined.mp4',
          './datasets/Sunny/april21_outlined.mp4',
          './datasets/Urban/march9_outlined.mp4']

output_images = ['./datasets/Dense/jan28_{}.jpeg',
          './datasets/Sunny/april21_{}.jpeg',
          './datasets/Urban/march9_{}.jpeg']

index = 0

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

    for car in annotated_data[index]:
        startx, starty = car[0], car[1]
        endx, endy = car[0] + car[2], car[1] + car[3]

        cropped = imcopy[starty:endy, startx:endx]
        if cropped.shape[0] > 30 and cropped.shape[1] > 30:
            # Only take in images with more than 30 x 30 pixels
            cropped = cv2.resize(cropped, (64, 64))
            cv2.imwrite(output_images.format(index), cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

        img = draw_boxes(img, [((startx, starty), (endx, endy))])

    index += 1

    return img


if __name__ == "__main__":

    for i in range(len(videos)):
        annotated_data = parse_annotation_data(pos_annotations[i])

        video = VideoFileClip(videos[i])
        annot_vid = video.fl_image(lambda x: process_image(x, annotated_data, output_images[i]))
        annot_vid.write_videofile(output_videos[i], audio=False)
        index = 0

