from collections import deque
import pickle

from train_classifier import extract_hog_features
from train_classifier import extract_bin_spacial_features
from train_classifier import extract_color_hist_features

import cv2
import numpy as np

from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

### TODO: Tweak these parameters and see how the results change.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell #16
cell_per_block = 1 # HOG cells per block #2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [300, None] # Min and max in y to search in slide_window()


svc = None
scaler = None

def extract_features(image, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_features, hog_features):

    file_features = []
    
    if spatial_feat:
        file_features.append(extract_bin_spacial_features(image))
    if hist_features:
        file_features.append(extract_color_hist_features(image))
    if hog_features:
        file_features.append(extract_hog_features(image))
    
    return np.concatenate(file_features)



def sliding_windows(img, x_range, y_range, window_size, xy_overlap_percent):
    windows = []
    
    x_start = x_range[0];
    x_end = x_range[1];
    y_start = y_range[0];
    y_end = y_range[1];
    
    # check values and use full image size if parameter is not given
    if (x_start == None):
        x_start = 0;
    if (x_end == None):
        x_end = img.shape[1];
    if (y_start == None):
        y_start = 0;
    if (y_end == None):
        y_end = img.shape[0];

    # window span
    x_span = x_end - x_start;
    y_span = y_end - y_start;

    # how much to move the window by each step
    x_step_size = window_size[0] * (1 - xy_overlap_percent[0])
    y_step_size = window_size[1] * (1 - xy_overlap_percent[1])

    # the region that overlaps
    x_overlap_region = window_size[0] * xy_overlap_percent[0]
    y_overlap_region = window_size[1] * xy_overlap_percent[1]

    # num of times the window fits in the span
    x_num_steps = np.int((x_span - x_overlap_region) / x_step_size)
    y_num_steps = np.int((y_span - y_overlap_region) / y_step_size)

    for y in range(y_num_steps):
        for x in range(x_num_steps):
            window_start = (np.int(x * x_step_size + x_start), np.int(y * y_step_size + y_start))
            window_end = (np.int(window_start[0] + window_size[0]), np.int(window_start[1] + window_size[1]))
            windows.append((window_start, window_end))

    return windows


def search_windows(img, windows, clf, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_features, hog_features):
    on_windows = []
    for window in windows:
        startx = window[0][0]
        starty = window[0][1]
        endx = window[1][0]
        endy = window[1][1]

        cropped = img[starty:endy, startx:endx]
        test_img = cv2.resize(cropped, (64, 64))
        features = extract_features(test_img, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_features=hist_features, hog_features=hog_features)

        # transform features to be fed into classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = clf.predict(test_features)

        if prediction == 1:
            on_windows.append(window)

    startx = window[0][0]
    starty = window[0][1]
    endx = window[1][0]
    endy = window[1][1]

    cropped = img[starty:endy, startx:endx]
    test_img = cv2.resize(cropped, (64, 64))
    features = extract_features(test_img, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_features=hist_features, hog_features=hog_features)
    
    # transform features to be fed into classifier
    test_features = scaler.transform(np.array(features).reshape(1, -1))
    prediction = clf.predict(test_features)
            
    return on_windows


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def get_classified_windows(image):
    # Returns windows that classifier claims to be a car
    
    xy_window = [(64,64), (96,96)]#, (128,128), (256,256)]
#     xy_window = [(32,32), (64,64), (128,128), (256,256)]
    y_start_stop = [[300, None], [300, None]]#, [300, None], [300, None]]

    windows_temp = []
    for i in range(len(xy_window)):
        windows = sliding_windows(image, [None, None], y_start_stop[i],
                            xy_window[i], (0.75, 0.75))
        windows = search_windows(image, windows, svc, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel,
                                        spatial_feat, hist_feat, hog_feat)
        windows_temp.append(windows)
        
    #Flatten windows_temp
    windows_final = sum(windows_temp, [])
    return windows_final


def apply_heat(heatmap, windows, heat_threshold):
    
    for window in windows:
        startx, starty = window[0][0], window[0][1]
        endx, endy = window[1][0], window[1][1]
        # Add heat value of 1 to all pixels inside classified 'true' window
        heatmap[starty:endy, startx:endx] += 1
    
    # Use threshold to zero out values
    heatmap[heatmap <= heat_threshold] = 0
    return heatmap


def draw_cars(img, labels):
    # Labels contains ([Image Array], number of heat blobs)
    for car_number in range(1, labels[1]+1):
        # Get pixels for the corresponding labeled heat
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Get the boundary of the heat
        outline = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, outline[0], outline[1], (0,0,255), 6)

    return img


# Number of frames to average out from
NUM_FRAMES = 10
# Heat values must be over threshold to be valid
HEAT_THRESHOLD = 20
windows_in_frames = deque([])


def process_frame(image, num_frames=NUM_FRAMES, heat_threshold=HEAT_THRESHOLD):
    global windows_in_frames

    classified_windows = get_classified_windows(image)
    windows_in_frames.append(classified_windows)

    if len(windows_in_frames) > num_frames:
        windows_in_frames.popleft()
    windows_in_frames_flattened = sum(windows_in_frames, [])

    heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)
    heatmap = apply_heat(heatmap, windows_in_frames_flattened, heat_threshold)
    labels = label(heatmap)
    final_image = draw_cars(image, labels)

    return final_image


if __name__ == "__main__":

    with open('classifier.p', 'rb') as f:
        data = pickle.load(f)

    svc = data['classifier']
    scaler = data['scaler']

    video = "datasets/Sunny/april21.avi"
    video = VideoFileClip(video)

    detection_video = video.fl_image(lambda x: process_frame(x, NUM_FRAMES))
    detection_video.write_videofile("vehicle_detection.mp4", audio=False)
