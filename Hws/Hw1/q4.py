# import our needed libraries
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio

# define our needed function
def power_law_transformation(image, y=0.6):    
    trnsf = 255 * (image / 255)**y
    return trnsf.astype(int)

# read our image
image = cv2.imread('Poinsettia.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# seperate our r, g, b channels
image_r = image[:, :, 0]
image_g = image[:, :, 1]
image_b = image[:, :, 2]

# check mean of every chanel
print("r mean : ", np.mean(image_r))
print("g mean : ", np.mean(image_g))
print("b mean : ", np.mean(image_b))

# main process
tre = 55
image_g = np.maximum(image_g, image_r)
image_g = image_g.astype(int) 
image_r[image_r > tre] = 0 

# change r, b chanel
image_r = image_r + 20
image_b = power_law_transformation(image_b, 0.95)

# create final image
image_final = np.dstack([image_r, image_g, image_b]).astype(int)

# check mean of every chanel after these changes
print('r mean : ', np.mean(image_r))
print('g mean : ', np.mean(image_g))
print('b mean : ', np.mean(image_b))

# show final image
skio.imshow(image_final)
skio.show()

# save our result
result_22 = np.float32(image_final)
result_22 = cv2.cvtColor(result_22, cv2.COLOR_RGB2BGR)
cv2.imwrite('result_22.jpg', result_22)