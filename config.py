import os.path as osp
from easydict import EasyDict as edict
__C = edict()

cfg = __C

__C.DATA_DIR = 'data'
__C.VEHICLES_DIR = osp.join(__C.DATA_DIR, 'vehicles')
__C.NON_VEHICLES_DIR = osp.join(__C.DATA_DIR, 'non-vehicles')

# Define the parameters for feature extraction
__C.color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
__C.orient = 8 # HOG orientations
__C.pix_per_cell = 8 # HOG pixels per cell
__C.cell_per_block = 2 # HOG cells per block
__C.hog_channel = 0 # Can be 0, 1, 2, or "ALL"
__C.spatial_size_x = 16 # Spatial binning dimensions x
__C.spatial_size_y = 16 # Spatial binning dimensions y
__C.hist_bins = 32 # Number of histogram bins
__C.spatial_feat = True # Spatial features on or off
__C.hist_feat = True # Histogram features on or off
__C.hog_feat = True # HOG features on or off
