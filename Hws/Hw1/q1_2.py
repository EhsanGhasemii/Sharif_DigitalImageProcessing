# q1-2
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter

# Create histogram of an image with this function
def creat_histogram(image):    
    resolution = 256
    counter = dict(Counter(image.ravel()))
    indices = np.array(list(counter.keys()))
    val = np.array(list(counter.values()))
    image_hist = np.zeros(resolution)
    image_hist[indices] = val
    return image_hist

# This function calculate the best mapping for histogram equalization
def histogram_equalization(image_hist):
    resolution = 255
    image_mapping = np.floor(resolution * np.cumsum(image_hist) / np.sum(image_hist)).astype(int)
    return image_mapping

# Read our image
image = cv2.imread('LuebeckCityGate.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

# Calculate 3 channels hsitogram of our image
image_r_hist = creat_histogram(image[:, :, 0])
image_g_hist = creat_histogram(image[:, :, 1])
image_b_hist = creat_histogram(image[:, :, 2])

# Showing histograms 
fig, axs = plt.subplots(1, 3)
axs[0].plot(image_r_hist)
axs[0].set_title("red chanel histogram")
axs[1].plot(image_g_hist)
axs[1].set_title("green chanel histogram")
axs[2].plot(image_b_hist)
axs[2].set_title("blue chanel histogram")
plt.show()

# Using histogram_equalization() to find the mapping for this part of assignment
image_r_mapping = histogram_equalization(image_r_hist)
image_g_mapping = histogram_equalization(image_g_hist)
image_b_mapping = histogram_equalization(image_b_hist)

# Apply mapping to 3 channels seperately
image_r_chanel = image_r_mapping[image[:, :, 0]]
image_g_chanel = image_g_mapping[image[:, :, 1]]
image_b_chanel = image_b_mapping[image[:, :, 2]]

# Concatenate 3 channels to create a unit image
image_r_chanel = image_r_chanel.reshape(image.shape[0], image.shape[1], 1)
image_g_chanel = image_g_chanel.reshape(image.shape[0], image.shape[1], 1)
image_b_chanel = image_b_chanel.reshape(image.shape[0], image.shape[1], 1)
final_image = np.concatenate((image_r_chanel, image_g_chanel, image_b_chanel), axis=2)

# Showing result
plt.figure()
plt.imshow(final_image)
plt.show()

# Create histograms for final image
final_r_hist = creat_histogram(final_image[:, :, 0])
final_g_hist = creat_histogram(final_image[:, :, 1])
final_b_hist = creat_histogram(final_image[:, :, 2])

# Showing histograms 
fig, axs = plt.subplots(1, 3)
axs[0].plot(final_r_hist)
axs[0].set_title("red chanel histogram")
axs[1].plot(final_g_hist)
axs[1].set_title("green chanel histogram")
axs[2].plot(final_b_hist)
axs[2].set_title("blue chanel histogram")
plt.show()

# Saving our result
fig.savefig('result_03.jpg')
result_04 = np.concatenate((image_b_chanel, image_g_chanel, image_r_chanel), axis=2)
cv2.imwrite('result_04.jpg', result_04)