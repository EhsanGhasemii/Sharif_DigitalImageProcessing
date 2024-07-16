import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.io as skio
plt.rcParams['figure.figsize'] = (20, 10)


# Part one - Define the main function==============================================================
def find_shifting(chanel_1, chanel_2, x_lim=(-20, 150), y_lim=(0, 100), step=4, last_process='ON'):
    '''
    chanel_1 is reference and find proper x_ and y_ shifting to set chanel_2 to chanel_1
    '''
    
    # crop from center of chanel_1 and chanel_2 to reduce our computation time
    resolution_divider = 8
    height = chanel_1.shape[0]
    width = chanel_1.shape[1]
    
    # set the boundary of x_range and y_range for moving chanel to in chanel 1
    x_boundary = range(x_lim[0], x_lim[1], step)
    y_boundary = range(y_lim[0], y_lim[1], step)
    
    # define shift score list
    shift_score_current = [(i, j) for i in x_boundary for j in y_boundary]
    shift_score_current = np.array(shift_score_current)
    
    # main body : 
    while resolution_divider > 1:
        
        # crop image depend on resolution divider
        x1 = int((0.5 - 0.5 / resolution_divider) * height)
        x2 = int((0.5 + 0.5 / resolution_divider) * height)
        y1 = int((0.5 - 0.5 / resolution_divider) * width)
        y2 = int((0.5 + 0.5 / resolution_divider) * width)
        c_1 = chanel_1[x1:x2, y1:y2]
        c_2 = chanel_2[x1:x2, y1:y2]
        
        # just do computing in center of our channels
        height_1 = c_1.shape[0]
        width_1 = c_1.shape[1]
        x_1 = int(0.25 * height_1)
        x_2 = int(0.75 * height_1)
        y_1 = int(0.25 * width_1)
        y_2 = int(0.75 * width_1)
        c1 = c_1[x_1:x_2, y_1:y_2]
        
        # do our process for resolution divider : [8, 4, 2]
        shift_score_new = []
        for i in range(shift_score_current.shape[0]):
            x = int(shift_score_current[i, 0])
            y = int(shift_score_current[i, 1])
            img = np.roll(c_2, (x, y), axis=(0, 1))
            img = img[x_1:x_2, y_1:y_2]
            
            # calculate score of every shifting
            score = np.sum((c1 - img) ** 4)
            my_tuple = (x, y, score)
            shift_score_new.append(my_tuple)
        shift_score_new = np.array(shift_score_new)
        shift_score_new = shift_score_new[shift_score_new[:, 2].argsort()]
        
        # printting result
        print('resolution divider : ', resolution_divider)
        print('shape of shift score : ', shift_score_new.shape)
        
        # ready to go to next step
        my_len = int(np.ceil(shift_score_new.shape[0] / 4))
        shift_score_current = shift_score_new[:my_len, :]
        resolution_divider /= 2

        # printting result
        print('chanel 1 size : ', c_1.shape)
        print('chanel 2 size : ', c_2.shape)    
        print('the best shifting is : ', shift_score_new[0, :])
        print('==========================================')
        
    # mode last_process to set the best possible one.
    if last_process == 'ON':
        my_len = int(np.ceil(shift_score_new.shape[0] / (step-1) ** 2))
        shift_score_new = shift_score_new[:my_len, :]
        shift_score_current = shift_score_new
        shift_score_new = []
        for i in range(shift_score_current.shape[0]):
            x = int(shift_score_current[i, 0])
            y = int(shift_score_current[i, 1])
            for j in range(1, step):
                for k in range(1, step):
                    x1 = x + j
                    y1 = y + k
                    img = np.roll(c_2, (x1, y1), axis=(0, 1))
                    img = img[x_1:x_2, y_1:y_2]
                    score = np.sum((c1 - img) ** 4)
                    my_tuple = (x1, y1, score)
                    shift_score_new.append(my_tuple)
        shift_score_new = np.array(shift_score_new)
        shift_score_new = shift_score_new[shift_score_new[:, 2].argsort()]
        print('last process mode is "on" and shape of shift score : ', shift_score_new.shape)
        print('the best shifting is : ', shift_score_new[0, :])
        print('==========================================')
                
    return shift_score_new[0, :2]


# Part two - Girls image==============================================================
# read our image
image = cv2.imread('Girls.tif', 0)             # Girls : g : [-15, 10], r : [11, 15]
                                               # Train : g : [43, 0]  , r : [88, 30]
                                               # Museum : g : [78, 19], r : [149, 25]

# convert image type to float
image = image.astype(float)                  

# create our (r, g, b) channels
height = int(np.floor(image.shape[0] / 3))
image_b = image[:height]
image_g = image[height:2*height]
image_r = image[2*height:3*height]

# calculate best shiftting for green and red chanel
g_shift = find_shifting(image_b,
                        image_g,
                        x_lim=(-20, 20),
                        y_lim=(0, 20),
                        step=2,
                        last_process='ON').astype(int)
print('the best shifting for green chanel to set to blue one is : ', g_shift)
r_shift = find_shifting(image_b,
                        image_r,
                        x_lim=(-20, 20),
                        y_lim=(0, 20),
                        step=2,
                        last_process='ON').astype(int)
print('the best shifting for green chanel to set to blue one is : ', r_shift)

# shiftting our 2 channels
image_g_shifted = np.roll(image_g, g_shift, axis=(0, 1))
image_r_shifted = np.roll(image_r, r_shift, axis=(0, 1))

# concatenating 3 final channels to create an united image
image_final = np.dstack([image_r_shifted, image_g_shifted, image_b]).astype(int)
print(image_final.shape)

# crop margin not set boarders
my_len = 130
mask = np.ones((image_final.shape[0] - 2 * my_len, image_final.shape[1] - 2 * my_len, image_final.shape[2]))
mask = np.pad(mask , ((my_len, my_len), (my_len, my_len), (0, 0)))
image_final = (image_final * mask).astype(int)

# showing final image
skio.imshow(image_final)
skio.show()

# save our result
result_07 = np.float32(image_final)
result_07 = cv2.cvtColor(result_07, cv2.COLOR_RGB2BGR)
cv2.imwrite('Girls_result_07.jpg', result_07)


# Part three - Train image==============================================================
# read our image
image = cv2.imread('Train.tif', 0)             # Girls : g : [-15, 10], r : [11, 15]
                                               # Train : g : [43, 0]  , r : [88, 30]
                                               # Museum : g : [78, 19], r : [149, 25]

# convert image type to float
image = image.astype(float)                  

# create our (r, g, b) channels
height = int(np.floor(image.shape[0] / 3))
image_b = image[:height]
image_g = image[height:2*height]
image_r = image[2*height:3*height]

# calculate best shiftting for green and red chanel
g_shift = find_shifting(image_b,
                        image_g,
                        x_lim=(0, 100),
                        y_lim=(0, 40),
                        step=6,
                        last_process='ON').astype(int)
print('the best shifting for green chanel to set to blue one is : ', g_shift)
r_shift = find_shifting(image_b,
                        image_r,
                        x_lim=(0, 100),
                        y_lim=(0, 40),
                        step=6,
                        last_process='ON').astype(int)
print('the best shifting for green chanel to set to blue one is : ', r_shift)

# shiftting our 2 channels
image_g_shifted = np.roll(image_g, g_shift, axis=(0, 1))
image_r_shifted = np.roll(image_r, r_shift, axis=(0, 1))

# concatenating 3 final channels to create an united image
image_final = np.dstack([image_r_shifted, image_g_shifted, image_b]).astype(int)
print(image_final.shape)

# crop margin not set boarders
my_len = 120
mask = np.ones((image_final.shape[0] - 3 * my_len, image_final.shape[1] - 2 * my_len, image_final.shape[2]))
mask = np.pad(mask , ((2 * my_len, my_len), (my_len, my_len), (0, 0)))
image_final = (image_final * mask).astype(int)

# showing final image
skio.imshow(image_final)
skio.show()

# save our result
result_06 = np.float32(image_final)
result_06 = cv2.cvtColor(result_06, cv2.COLOR_RGB2BGR)
cv2.imwrite('Train_result_06.jpg', result_06)


# Part four - Museum image==============================================================
# read our image
image = cv2.imread('Museum.tif', 0)            # Girls : g : [-15, 10], r : [11, 15]
                                               # Train : g : [43, 0]  , r : [88, 30]
                                               # Museum : g : [78, 19], r : [149, 25]

# convert image type to float
image = image.astype(float)                  

# create our (r, g, b) channels
height = int(np.floor(image.shape[0] / 3))
image_b = image[:height]
image_g = image[height:2*height]
image_r = image[2*height:3*height]

# calculate best shiftting for green and red chanel
g_shift = find_shifting(image_b,
                        image_g,
                        x_lim=(50, 200),
                        y_lim=(0, 30),
                        step=6,
                        last_process='ON').astype(int)
print('the best shifting for green chanel to set to blue one is : ', g_shift)
r_shift = find_shifting(image_b,
                        image_r,
                        x_lim=(50, 200),
                        y_lim=(0, 200),
                        step=6,
                        last_process='ON').astype(int)
print('the best shifting for green chanel to set to blue one is : ', r_shift)

# shiftting our 2 channels
image_g_shifted = np.roll(image_g, g_shift, axis=(0, 1))
image_r_shifted = np.roll(image_r, r_shift, axis=(0, 1))

# concatenating 3 final channels to create an united image
image_final = np.dstack([image_r_shifted, image_g_shifted, image_b]).astype(int)
print('shape of final image : ', image_final.shape)

# crop margin not set boarders
my_len = 140
mask = np.ones((image_final.shape[0] - 2 * my_len, image_final.shape[1] - 2 * my_len, image_final.shape[2]))
mask = np.pad(mask , ((my_len, my_len), (my_len, my_len), (0, 0)))
image_final = (image_final * mask).astype(int)

# showing final image
skio.imshow(image_final)
skio.show()

# save our result
result_05 = np.float32(image_final)
result_05 = cv2.cvtColor(result_05, cv2.COLOR_RGB2BGR)
cv2.imwrite('Museum_result_05.jpg', result_05)