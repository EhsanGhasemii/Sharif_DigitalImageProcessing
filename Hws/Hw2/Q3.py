# import our needed libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random 
from numba import jit
import time 
plt.rcParams['figure.figsize'] = (20, 10)

# Define our needed functions -------------------------------
# function for reading an image from its path
def reading_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# showing an image by matplotlib library
def showing_image(image): 
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.show()

# function for creating a hole in an image
def hole_creating(img_in, pixels):
    img_out = img_in.copy()
    for pixel in pixels:
        img_out[pixel[0]:pixel[2], pixel[1]:pixel[3]] = 0
    return img_out

# function for pick a slide to tecture synthesis
def pick_random_slide(img_in, w=50, l=50, hole_zone=[]):
    img_w = img_in.shape[0]
    img_l = img_in.shape[1]
    
    # check to be not in a hole
    in_hole = True
    while in_hole:
        
        # pick a random (x, y)
        x = random.randint(0, img_w - w)
        y = random.randint(0, img_l - l)
        
        # check is in hole or not? 
        in_hole = False
        for hole in hole_zone:
            if (x + w > hole[0]) & (y + l > hole[1]) & (hole[0] + hole[2] > x) & (hole[1] + hole[3] > y):
                in_hole = True
    
    img = img_in[x:x+w, y:y+l]
    return img

# function for pick best slide 
def pick_best_slide(img_in, x_loc, y_loc, w=50, l=50, w2=20,
                    l2=20, hole_zone=[], random_p=20000, k=4, step=4, Th1=1.1, x1=100,
                    y1=100, x2=1500-100, y2=1500-100, neigbors=['up']):
    # create a list including all possibles pixels to check matching
    pixels = [(i, j) for i in range(x1, x2, k) for j in range(y1, y2, k)]
    pixels_len = int(np.ceil((x2 - x1) / k) * np.ceil((y2 - y1) / k))
#     print('pixels len : ', pixels_len)
        
    # create location of image
    img_loc_up = img_in[x_loc-w2:x_loc, y_loc-l2:y_loc+l+l2]
    img_loc_down = img_in[x_loc+w:x_loc+w+w2, y_loc-l2:y_loc+l+l2]
    img_loc_left = img_in[x_loc-w2:x_loc+w+w2, y_loc-l2:y_loc]
    img_loc_right = img_in[x_loc-w2:x_loc+w+w2, y_loc+l:y_loc+l+l2]

    
    # set initial variables
    min_score = np.inf
    best_slide = (0, 0)
    
    # loop over pixels randomly
    for i in range(np.min((pixels_len, random_p))):
        
        # check to not to be in a hole
        in_hole = True
        while in_hole:
            
            # pick a random pixel to check matching 
            indx = random.randint(0, pixels_len - i - 1)
            pixel = pixels.pop(indx)
            pixel_x = pixel[0]
            pixel_y = pixel[1]
            
            # check is in hole or not? 
            in_hole = False
            for hole in hole_zone:
                if (pixel_x + w > hole[0]) & (pixel_y + l > hole[1]) & (hole[0] + hole[2] > pixel_x) & (hole[1] + hole[3] > pixel_y):
                    in_hole = True
                    
            if in_hole:
                pixels_len -=1
            
            
        
        img_up = img_in[pixel_x-w2:pixel_x, pixel_y-l2:pixel_y+l+l2]
        img_down = img_in[pixel_x+w:pixel_x+w+w2, pixel_y-l2:pixel_y+l+l2]
        img_left = img_in[pixel_x-w2:pixel_x+w+w2, pixel_y-l2:pixel_y]
        img_right = img_in[pixel_x-w2:pixel_x+w+w2, pixel_y+l:pixel_y+l+l2]
        
        # Cross Corelation to check matching
        score = 0
        for neigbor in neigbors:
            if neigbor == 'up':
                score += (np.sum((img_up - img_loc_up)**2))
            elif neigbor == 'down':
                score += (np.sum((img_down - img_loc_down)**2))
            elif neigbor == 'left':
                score += (np.sum((img_left - img_loc_left)**2))
            else:
                score += (np.sum((img_right - img_loc_right)**2))
        
        
        if score < min_score:
            best_slide = (pixel_x, pixel_y)
            min_score = score
            
        # if it is close to the location check its neighbors
        if score < Th1 * min_score:
            for i in range(1, k, step):
                for j in range(1, k, step):
                    x = pixel_x + i
                    y = pixel_y + j
                    
                    img_up = img_in[pixel_x-w2:pixel_x, pixel_y-l2:pixel_y+l+l2]
                    img_down = img_in[pixel_x+w:pixel_x+w+w2, pixel_y-l2:pixel_y+l+l2]
                    img_left = img_in[pixel_x-w2:pixel_x+w+w2, pixel_y-l2:pixel_y]
                    img_right = img_in[pixel_x-w2:pixel_x+w+w2, pixel_y+l:pixel_y+l+l2]
                    
                    
                    score = 0
                    for neigbor in neigbors:
                        if neigbor == 'up':
                            score += (np.sum((img_up - img_loc_up)**2))
                        elif neigbor == 'down':
                            score += (np.sum((img_down - img_loc_down)**2))
                        elif neigbor == 'left':
                            score += (np.sum((img_left - img_loc_left)**2))
                        elif neigbor == 'right':
                            score += (np.sum((img_right - img_loc_right)**2))
                            
                            
                    if score < min_score:
                        best_slide = (x, y)
                        min_score = score
        
    return best_slide
    

# this function is for hole filling by help of texture synthesis
def texture_synthesis(img_texture, x1, y1, x2, y2, w=50, l=50, w2=20, l2=20,
                      org_x1=100, org_y1=100, org_x2=1500-100, org_y2=1500-100,
                      hole_zone=[], random_p=20000, Cmode="Random", watch_creating=False):
#     img_out = np.zeros(shape=img_texture.shape)
#     img_out = img_out.astype(np.uint8)
    img_out = img_texture.copy()
    
    # initial first parameter
    mode = 0
    mode_c = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    mode_current = 0 
    i = 1
    j = 1
    di = mode_c[mode][0]
    dj = mode_c[mode][1]
    a, b = np.meshgrid(range(x1, x2, w), range(y1, y2, l))
    c = np.zeros(shape=a.shape)
    y_len = c.shape[0]
    x_len = c.shape[1]
    state = np.dstack((a, b, c))
    
    # pad the array
    pad = ((1, 1), (1, 1), (0, 0))
    state = np.pad(state, pad, mode='constant', constant_values=1)
    
    # check all the blocks till any one be unseened
    counter = 0
    while not state[j][i][2]: 
        
        # set the variables
        x = int(state[j][i][0])
        y = int(state[j][i][1])
        
        # print the way 
#         print('x : ', x)
#         print('y : ', y)
#         print('i : ', i)
#         print('j : ', j)
        
        # mark this block that have slide now
        state[j][i][2] = 1

        # check for the end of each line 
        if (state[j+dj][i+di][2] == 1):
            mode += 1
            mode %= 4
            
            # update di and dj
            di = mode_c[mode][0]
            dj = mode_c[mode][1]
        

        # put the picked image in the specified location
        if Cmode == "Random":
            img = pick_random_slide(img_out, w=np.min((w, x2-x)), l=np.min((l, y2-y)), hole_zone=hole_zone)
            
        elif Cmode == "Best":
            
            # define neighbors
            neigbors = []
            if state[j][i+1][2] == 1:
                neigbors.append('down')
            if state[j][i-1][2] == 1:
                neigbors.append('up')
            if state[j+1][i][2] == 1:
                neigbors.append('right')
            if state[j-1][i][2] == 1:
                neigbors.append('left')
                
            
            (x_best, y_best) = pick_best_slide(img_out, x_loc=x, y_loc=y, w=np.min((w, x2-x)),
                                               l=np.min((l, y2-y)), w2=w2, l2=l2, hole_zone=hole_zone,
                                               random_p=random_p, x1=org_x1, y1=org_y1, x2=org_x2, y2=org_y2, neigbors=neigbors)
            img = img_out[x_best:x_best+np.min((w, x2-x)), y_best:y_best+np.min((l, y2-y))]
            
        img_out[x:x+np.min((w, x2-x)), y:y+np.min((l, y2-y))] = img
        
        # showing the image 
        if watch_creating:
            showing_image(img_out)
        
        # go to next slide 
        i += di
        j += dj
        mode_current = mode
        counter += 1
        
        # report to user
        print('slide ', counter, ' is done!')
    print('--------------')
    return img_out














# Reading our images
img_org = reading_image('Swimmer.jpg')

# creating holes
hole_zone = []
hole_zone.append((700, 700, 1200, 1000))
img1 = hole_creating(img_org, hole_zone)

# Showing the images
showing_image(img1)

# saving the image
Q3_hole = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
cv2.imwrite('Q3_hole.jpg', Q3_hole)

















# run the algorithm
img2 = texture_synthesis(img1, 700, 700, 1200, 1000, w=100, l=100, hole_zone=hole_zone, Cmode="Random", watch_creating=True)
showing_image(img2)
result13_random = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
cv2.imwrite('result13_random.jpg', result13_random)























# run the algorithm
img2 = texture_synthesis(img1, 700, 700, 1200, 1000, w=5, l=5, hole_zone=hole_zone, Cmode="Best", watch_creating=False)
showing_image(img2)
result13 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
cv2.imwrite('result13.jpg', result13)





















# run the algorithm
img2 = texture_synthesis(img1, 700, 700, 1200, 1000, w=50, l=50, hole_zone=hole_zone, Cmode="Best", watch_creating=False)
showing_image(img2)
result13_2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
cv2.imwrite('result13_2.jpg', result13_2)


















# Reading our images
img_org = reading_image('HainesEagle.jpg')

# creating holes
hole_zone = []
hole_zone.append((420, 570, 620, 660))
hole_zone.append((900, 880, 1100, 980))
hole_zone.append((1500, 750, 1700, 840))
hole_zone.append((1950, 700, 2150, 800))
hole_zone.append((1270, 3210, 1470, 3300))
hole_zone.append((1530, 3750, 1750, 3850))
hole_zone.append((1700, 4050, 1900, 4150))






img1 = hole_creating(img_org, hole_zone)

# Showing the images
showing_image(img1)

# saving the image
Q3_hole2 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
cv2.imwrite('Q3_hole2.jpg', Q3_hole2)


















img2 = img1.copy()
# run the algorithm
for hole in hole_zone: 
    img2 = texture_synthesis(img2, hole[0], hole[1], hole[2], hole[3], w=50, l=50, w2=20, l2=20, hole_zone=hole_zone, Cmode="Random", watch_creating=False)


result12_random = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
cv2.imwrite('result12_random.jpg', result12_random)
showing_image(img2)





























img2 = img1.copy()
# run the algorithm
for hole in hole_zone: 
    img2 = texture_synthesis(img2, hole[0], hole[1], hole[2], hole[3], w=20, l=20, w2=10, l2=10,
                             org_x1=320, org_y1=1400, org_x2=1900, org_y2=2100, hole_zone=hole_zone,
                             random_p=20000, Cmode="Best", watch_creating=False)

result12 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
cv2.imwrite('result12.jpg', result12)
showing_image(img2)











































