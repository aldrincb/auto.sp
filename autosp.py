import glob
import os
import time


import cv2
import numpy as np

from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from neuralnetwork import Neural_Network
from neuralnetwork import trainer

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
def extract_hog_features(img, channel=0, orientations=9, pixels_per_cell=8, cells_per_block=2, transform_sqrt=False, visualise=False, feature_vector=True):
    if channel == "ALL":
        for j in xrange(0, img.shape[2]):
            hog_features = hog(img[:,:,j], orientations=orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell), cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=transform_sqrt, visualise=visualise, feature_vector=feature_vector)
            hog_features = np.ravel(hog_features)
            print hog_features
    else:
        hog_features = hog(img[:,:,channel], orientations=orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell), cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=transform_sqrt, visualise=visualise, feature_vector=feature_vector)

    return hog_features

def extract_bin_spacial_features(image, size=(32, 32)):
    return cv2.resize(image, size).ravel()


def extract_color_hist_features(image, nbins=32, bins_range=(0,256)):

    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(image[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(image[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(image[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features




def extract_features(files, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_features, hog_features):
    features = []

    for file in files:
        file_features = []
        image = cv2.imread(file)
        if spatial_feat:
            file_features.append(extract_bin_spacial_features(image))
        if hist_features:
            file_features.append(extract_color_hist_features(image))
        if hog_features:
            file_features.append(extract_hog_features(image))

        features.append(np.concatenate(file_features))

    return features



def normalize_features(raw_features):
    normalized_features = []
    return normalized_features

def search_windows(img, windows, clf, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_features, hog_features):
    on_windows = []
    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = extract_features([test_img], color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)[0]

        # transform features to be fed into classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = clf.predict(test_features)

        if prediction == 1:
            on_windows.append(window)

    return on_windows


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



if __name__ == "__main__":
    car_features = extract_features(cars, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel,
                                        spatial_feat, hist_feat, hog_feat)

    not_car_features = extract_features(not_cars, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel,
                                        spatial_feat, hist_feat, hog_feat)

    print "Done extracting features..."

    X = np.vstack((car_features, not_car_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    print "number of inputs: ", len(scaled_X)
    print "number of features: ", len(scaled_X[0])

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print ('Configured Using:',orient,'orientations',
           pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
    print ('Feature vector length:', len(X_train[0]))


    # TODO: We want to implement our own classifier.
    svc = LinearSVC()
    t1 = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()

    print ("{} seconds to train classifier...".format(round(t2-t1, 5)))

    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    NN = Neural_Network(Lambda=0.0001)
    T = trainer(NN)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)


    X_train = np.array(X_train, dtype=float)
    X_test = np.array(X_test, dtype=float)
    y_train = np.array(y_train.reshape(-1,1), dtype=float)
    y_test = np.array(y_test.reshape(-1,1), dtype=float)

    T.train(X_train,y_train,X_test,y_test)
