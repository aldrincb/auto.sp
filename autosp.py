import glob
import os
from skimage.feature import hog
import cv2
import numpy

# Load car and not car data
cars = []
not_cars = []

# Get car paths
car_paths = ["./udacity-dataset/vehicles/GTI_Far"
"./udacity-dataset/vehicles/GTI_Left",
"./udacity-dataset/vehicles/GTI_MiddleClose",
"./udacity-dataset/vehicles/GTI_Right",
"./udacity-dataset/vehicles/KITTI_extracted"]

for i in xrange(0, len(car_paths)):
    car_path = car_paths[i]
    for file in glob.glob(os.path.join(car_path, "*.png")):
        cars.append(file)

# Get not car paths
not_car_paths = ["./udacity-dataset/non-vehicles/Extras",
"./udacity-dataset/non-vehicles/GTI"]

for i in xrange(0, len(not_car_paths)):
    not_car_path = not_car_paths[i]
    for file in glob.glob(os.path.join(not_car_path, "*.png")):
        not_cars.append(file)

# Extract HOG for features
def extract_hog_features(files, channel=0, orientations=9, pixels_per_cell=8, cells_per_block=2, transform_sqrt=False, visualise=False, feature_vector=True):
    features = []

    for i in xrange(0, len(files)):
        file = files[i]
        img = cv2.imread(file)
        if channel == "ALL":
            for i in xrange(0, img.shape[2]):
                hog_features = hog(img[:,:,channel], orientations=orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell), cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=transform_sqrt, visualise=visualise, feature_vector=feature_vector)
        else:
            hog_features = hog(img[:,:,channel], orientations=orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell), cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=transform_sqrt, visualise=visualise, feature_vector=feature_vector)

        # Append the new feature vector to the features list
        features.append(hog_features)

    return features

def extract_bin_spacial_features(files, size=(32, 32)):
    features = []

    for i in xrange(0, len(files)):
        file = files[i]
        image = cv2.imread(file)

        features.append(cv2.resize(image, size).ravel())

    return features

def extract_color_hist_features(files, nbins=32, bins_range=(0,256)):
    features = []

    for i in xrange(0, len(files)):
        file = files[i]
        image = cv2.imread(file)

        # Compute the histogram of the color channels separately
        channel1_hist = numpy.histogram(image[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = numpy.histogram(image[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = numpy.histogram(image[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = numpy.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        features.append(hist_features)

    return features

def extract_features(files, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_features, hog_features):
    features = []
    if spatial_feat:
        print(len(extract_bin_spacial_features(files)))
    if hist_features:
        print(len(extract_color_hist_features(files)))
    if hog_features:
        print(len(extract_hog_features(files)))

def normalize_features(raw_features):
    normalized_features = []
    return normalized_features
