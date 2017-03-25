import glob
import os

# Load car and not car data
cars = []
not_cars = []

# Get car paths
car_paths = ["/udacity-dataset/vehicles/GTI_Far",
"/udacity-dataset/vehicles/GTI_Left",
"/udacity-dataset/vehicles/GTI_MiddleClose",
"/udacity-dataset/vehicles/GTI_Right",
"/udacity-dataset/vehicles/KITTI_extracted"]

for i in xrange(0, len(car_paths)):
    car_path = car_paths[i]
    os.chdir(car_path)
    for file in glob.glob("*.png"):
        cars.append(file)

# Get not car paths
not_car_paths = ["/udacity-dataset/non-vehicles/Extras",
"/udacity-dataset/non-vehicles/GTI"]

for i in xrange(0, len(not_car_paths)):
    not_car_path = not_car_paths[i]
    os.chdir(not_car_path)
    for file in glob.glob("*.png"):
        not_cars.append(file)
