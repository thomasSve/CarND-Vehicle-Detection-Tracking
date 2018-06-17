import os
import os.path as osp
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split

from config import cfg

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis = False, feature_vec = True):
    # Call with two outputs if vis = True
    if vis == True:
        features, hog_image = hog(img, orientations = orient,
                                  pixels_per_cell = (pix_per_cell, pix_per_cell),
                                  cells_per_block = (cell_per_block, cell_per_block),
                                  transform_sqrt = False, visualise = vis, feature_vector = feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations = orient,
                                  pixels_per_cell = (pix_per_cell, pix_per_cell),
                                  cells_per_block = (cell_per_block, cell_per_block),
                                  transform_sqrt = False, visualise = vis, feature_vector = feature_vec)
        return features

# Downsample the image
def bin_spatial(img, size = (32, 32)):
    return cv2.resize(img, size).ravel()

def color_hist(img, nbins = 32): # bins_range(0, 256)
    channel1_hist = np.histogram(img[:, :, 0], bins = nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins = nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins = nbins)

    hist_feature = np.hstack((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_feature

def convert_colorspace(img, color_space):
    if color_space == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif color_space == 'RBG':
        return cv2.cvtColor(img, cv2.COLOR_RGB2RBG)

def single_img_features(img, color_space = "RGB", spatial_size = (32, 32), hist_bins = 32, orient = 9, pix_per_cell = 8, cell_per_block = 2, hog_channel = 0, spatial_feat = True, hist_feat = True, hog_feat = True):
    # Iterate through the list of images
    img_features = []
    
    # apply color inversion if other than 'RGB'
    if color_space != 'RGB':
        feat_img = convert_colorspace(img, color_space)
    else:
        feat_img = np.copy(img)

    if spatial_feat:
        img_features.append(bin_spatial(feat_img, size = spatial_size))

    if hist_feat:
        img_features.append(color_hist(feat_img, nbins = hist_bins))

    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []

            for channel in range(feat_img.shape[2]):
                hog_features.append(get_hog_features(feat_img[:, :, channel], orient, pix_per_cell, cell_per_block, vis = False, feature_vec = True))
                    
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feat_img[:, :, hog_channel], orient, pix_per_cell, cell_per_block, vis = False, feature_vec = True)

        img_features.append(hog_features)
        
    return np.concatenate(img_features)

def extract_features(imgs, color_space = "RGB", spatial_size = (32, 32), hist_bins = 32, orient = 9, pix_per_cell = 8, cell_per_block = 2, hog_channel = 0, spatial_feat = True, hist_feat = True, hog_feat = True):
    # Create a list to append features to
    features = []

    # Iterate through the list of images
    for img_file in imgs:
        file_feat = []
        img = cv2.imread(img_file)
        
        # apply color inversion if other than 'RGB'
        file_features = single_img_features(img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

        features.append(file_features)

    return features

def slide_window(img, x_start_stop = [None, None], y_start_stop = [None, None], xy_window = (64, 64), xy_overlap = (0.5, 0.5)):
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / ny_pix_per_step) - 1
    
    # Initiate a list to append window positions to
    window_list = []

    # Loop through x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[0]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))

    return window_list

def draw_boxes(img, bboxes, color = (0, 0, 255), thick = 6):
    # Make copy
    imcopy = np.copy(img)
    
    # Iterate over bounding boxes
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        
    # Return image copy with boxes drawn
    return imcopy

# Search for vehicles in the windows
def search_windows(img, windows, clf, scaler, color_space = 'RGB', spatial_size = (32, 32), hist_bins = 32, hist_range = (0, 256), orient = 9, pix_per_cell = 8, cell_per_block = 2, hog_channel = 0, spatial_feat = True, hist_feat = True, hog_feat = True):
    # Create an empty list to recieve positive detection windows
    on_windows = []
    
    # Iterate over all windows in the list
    for window in windows:
        # Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        # Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        # Predict using classifier
        prediction = clf.predict(test_features)

        # If prediction positive (vehicle found), save the window
        if prediction:
            on_windows.append(window)

    return windows

def generate_datasets(vehicles, non_vehicles):
    # Get features for both datasets
    print("Generating features...")
    vehicle_features = extract_features(vehicles, color_space=cfg.color_space, 
                        spatial_size = (cfg.spatial_size_x, cfg.spatial_size_y),
                        hist_bins=cfg.hist_bins, 
                        orient = cfg.orient, pix_per_cell = cfg.pix_per_cell, 
                        cell_per_block = cfg.cell_per_block, 
                        hog_channel = cfg.hog_channel, spatial_feat = cfg.spatial_feat, 
                        hist_feat = cfg.hist_feat, hog_feat = cfg.hog_feat)
    
    notvehicle_features = extract_features(non_vehicles, color_space=cfg.color_space, 
                        spatial_size = (cfg.spatial_size_x, cfg.spatial_size_y),
                        hist_bins=cfg.hist_bins, 
                        orient = cfg.orient, pix_per_cell = cfg.pix_per_cell, 
                        cell_per_block = cfg.cell_per_block, 
                        hog_channel = cfg.hog_channel, spatial_feat = cfg.spatial_feat, 
                        hist_feat = cfg.hist_feat, hog_feat = cfg.hog_feat)
    
    # Merge together feature sets into one dataset
    X = np.vstack((vehicle_features, notvehicle_features)).astype(np.float64)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X) 

    # Define the labels vector
    y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(notvehicle_features))))

    # Split randomly into tran and test sets
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=22)


    return X_train, X_test, y_train, y_test, X_scaler
    
def load_image_lists():
    # Load vehicle images
    vehicles = glob.glob(osp.join(cfg.VEHICLES_DIR, '*', '*'))
    print('Number of vehicle images found: {}'.format(len(vehicles)))

    # Load non-vehicles images
    non_vehicles = glob.glob(osp.join(cfg.NON_VEHICLES_DIR, '*', '*'))
    print('Number of non-vehicles images found: {}'.format(len(non_vehicles)))

    return vehicles, non_vehicles

def train_classifier(X_train, y_train, X_test, y_test, loss = 'hinge'):
    print("Started training a linear SVC...")
    svc = LinearSVC(loss=loss)
    svc.fit(X_train, y_train)
    print("Successfully trained SVC classifier")
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    return svc

def show_img(img):
    if len(img.shape)==3:
        # Color BGR image
        plt.figure()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
    else:
        # Grayscale image
        plt.figure()
        plt.imshow(img, cmap='gray')

    plt.show()
    
def display_example(clf, X_scaler):
    for image_p in glob.glob('test_images/test*.jpg'):
        image = cv2.imread(image_p)
        draw_image = np.copy(image)
        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 640], 
                    xy_window=(128, 128), xy_overlap=(0.85, 0.85))
        hot_windows = []
        hot_windows += (search_windows(image, windows, clf, X_scaler, color_space=cfg.color_space, 
                        spatial_size = (cfg.spatial_size_x, cfg.spatial_size_y), hist_bins=cfg.hist_bins, 
                        orient = cfg.orient, pix_per_cell = cfg.pix_per_cell, 
                        cell_per_block = cfg.cell_per_block, 
                        hog_channel = cfg.hog_channel, spatial_feat = cfg.spatial_feat, 
                        hist_feat = cfg.hist_feat, hog_feat = cfg.hog_feat))
        
        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    
        show_img(window_img)
        
def main():
    # Load images
    vehicles, non_vehicles = load_image_lists()

    # Generate the datasets
    X_train, X_test, y_train, y_test, X_scaler = generate_datasets(vehicles, non_vehicles)

    # Train classifier
    clf = train_classifier(X_train, y_train, X_test, y_test)

    # Display example images on classifier
    display_example(clf, X_scaler)
    

if __name__ == '__main__':
    main()
