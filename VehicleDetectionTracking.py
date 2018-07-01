

from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip


#load the database

# Divide up into cars and notcars
car_images = glob.glob('vehicles/*/*.png')
non_car_images = glob.glob('non-vehicles/*/*.png')

# Store all car image files 
cars = []

# Store all not car image files
notcars = []

for image in car_images:
    cars.append(image)
for image in non_car_images:
    notcars.append(image)

print("Total car images ", len(cars))
print("Total non-car images ", len(notcars))

# Generate a random index to look at a car image
ind = np.random.randint(0, len(cars))
ind_notcar = np.random.randint(0, len(notcars))
# Read in the image
#print(cars[ind])

# Read png image in the scale of 0 to 1.
image = mpimg.imread(cars[ind])


# Read png image in the scale of 0 to 1
not_car_image = mpimg.imread(notcars[ind_notcar])

feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
feature_notcar_image = cv2.cvtColor(not_car_image, cv2.COLOR_RGB2YCrCb)




'''
Define a function to return HOG features
'''
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

'''
Define a function to compute binned color features using cv2.resize and then 
flattened the array using ravel
Return the horizontally stacked feature vector.
'''
def bin_spatial(img, size=(32, 32)):
    colorY = cv2.resize(img[:,:,0], size).ravel()
    colorCr = cv2.resize(img[:,:,1], size).ravel()
    colorCb = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((colorY, colorCr, colorCb))

# Visualize Spatially Binned Features

def convert_feature_color(img, color_space = 'RGB'):
    if color_space == 'RGB':
        feature_image = np.copy(img)
    if color_space == 'HSV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if color_space == 'LUV':
        #print("Hit LUV")
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if color_space == 'HLS':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if color_space == 'YUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if color_space == 'YCrCb':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    return feature_image

'''
Define a function to compute color histogram features if Image is on scale 0 to 1,
need to convert that image to scale 0 to 255 before the image can be given to this function
'''
def color_hist(img, nbins=32, bins_range=(0, 256), eachchannel_histogram=False):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    # Generating bin centers
    bin_edges = channel1_hist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0: len(bin_edges) -1])/2
    #hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    hist_features = np.hstack((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    
    if eachchannel_histogram == True:
        # Return the individual histograms, bin_centers and feature vector
        return channel1_hist, channel2_hist, channel3_hist, bin_centers, hist_features
    
    return hist_features
	

'''
Define a function to extract features from a list of images
Have this function call bin_spatial() and color_hist()
'''
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=12, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            # Apply Spatial Binning
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
#orient = 9
orient = 12
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

spatial_size = (32,32)
hist_bins = 32





t=time.time()



car_features = extract_features(cars, color_space=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
notcar_features = extract_features(notcars, color_space=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)

print('car_features: ', len(car_features) , 'notcar_features', len(notcar_features))
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)

# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features)))) #.reshape(1,-1)
print('Feature vector X Length: ', len(X) , 'scaled_X shape: ', scaled_X.shape, 'Label vector Y shape: ', y.shape)

# Randomized training data everytime the following 2 line executes and also 
# split the train and test sets in 20%
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
print('X_train', X_train.shape)

# Use a linear SVC 
clf = LinearSVC()
# Check the training time for the SVC
t=time.time()
clf.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', clf.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

print('DONE')


'''
Convert an RGB image to the desired colore space i.e. COLOR_RGB2YCrCb etc.
'''
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

'''
find_cars function will extract features using hog sub-sampling, spatial bin and
color histogram and will make predictions. 
Return a list of detected boxes as part of the car detection on a given image
'''
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, \
              cell_per_block, spatial_size, hist_bins):
    
    # Initialize a list to append window positions to
    window_list = []
    
    #draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    #nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(20,nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))#.reshape(1,-1)
            #print("hog_features shape:", hog_features.shape)
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get spatial bin features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            
            #Get color histogram feature  
            hist_features = color_hist(subimg, nbins=hist_bins)
        
            #print("spatial_features shape:", spatial_features.shape)
            #print("hist_features shape:", hist_features.shape)

            # Scale features and make a prediction
            total_feature = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            #print("total_feature", total_feature.shape)
            test_features = X_scaler.transform(total_feature)    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                
                # Calculate window position
                startx = xbox_left
                endx = xbox_left+win_draw
                starty = ytop_draw+ystart
                endy = ytop_draw+win_draw+ystart
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
   
    return window_list

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for boxs in bbox_list:
        for box in boxs:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap # Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img



class frame():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        
        self.box_list = []

    '''
         Append the fit coefficient into. If there are more than 10 fit coefficients 
         on current_fit remove the old one (the first one ), fifo/.
         Average the remaining and store it in self.best_fit
    '''
    def add_boxes(self, box):

        self.box_list.append(box)
        if len(self.box_list) > 10:
            self.box_list  = self.box_list[-10:]


frame_ctx = frame()

def pipeline(img):
    
    frames = 0
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    

    ystart = 400
    ystop = 656  
    scale = 1.5
    windows = find_cars(img, ystart, ystop, scale, clf, X_scaler, orient, \
                pix_per_cell, cell_per_block, spatial_size, hist_bins)
    if len(windows) > 0:
        frame_ctx.add_boxes(windows)

        
    
    ystart = 408
    ystop = 656  
    scale = 2
    windows = find_cars(img, ystart, ystop, scale, clf, X_scaler, orient, \
            pix_per_cell, cell_per_block, spatial_size, hist_bins)
    if len(windows) > 0:
        frame_ctx.add_boxes(windows)
    

    ystart = 400
    ystop = 500  
    scale = 1
    
    windows = find_cars(img, ystart, ystop, scale, clf, X_scaler, orient, \
            pix_per_cell, cell_per_block, spatial_size, hist_bins)
    if len(windows) > 0:
        frame_ctx.add_boxes(windows)


    # Add heat to each box in box list
    heat = add_heat(heat,frame_ctx.box_list)
    
    # Apply threshold to help remove false positives
    # I selected the threshold value based on multiple iteration of the track
    # with the above 3 different scale sliding window box detection and visually 
    # observed the True and False frames

    heat = apply_threshold(heat, 1 + len(frame_ctx.box_list)//2)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img



def processVDT_image(img):
    output = pipeline(img)
    return output

