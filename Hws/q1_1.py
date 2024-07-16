# https://automaticaddison.com/difference-between-histogram-equalization-and-histogram-matching/

# Q1-1
import numpy as np
import matplotlib.pyplot as plt
import cv2
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

# Function for Histogram matching between original image and specified image
def histogram_matching(original_hist, specified_hist):
    resolution = 255
    original_cdf = np.floor(resolution * np.cumsum(original_hist) / np.sum(original_hist))
    specified_cdf = np.floor(resolution * np.cumsum(specified_hist) / np.sum(specified_hist))
    x, y = np.meshgrid(original_cdf, specified_cdf)
    mapping = np.argmin(np.abs(x - y), axis=0)
    return mapping

# Reading our images
image_original = cv2.imread('Cecropia.jpg')
image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
plt.imshow(image_original)
plt.show()

plt.figure()
image_specified = cv2.imread('GeorgiaCypress.jpg')
image_specified = cv2.cvtColor(image_specified, cv2.COLOR_BGR2RGB)
plt.imshow(image_specified)
plt.show()

# Creat our histograms with our functino
image_original_r_hist = creat_histogram(image_original[:, :, 0])
image_original_g_hist = creat_histogram(image_original[:, :, 1])
image_original_b_hist = creat_histogram(image_original[:, :, 2])

image_specified_r_hist = creat_histogram(image_specified[:, :, 0])
image_specified_g_hist = creat_histogram(image_specified[:, :, 1])
image_specified_b_hist = creat_histogram(image_specified[:, :, 2])

# Showing our histograms 
fig, axs = plt.subplots(1, 3)
axs[0].plot(image_original_r_hist)
axs[0].set_title("red chanel of original image")
axs[1].plot(image_original_g_hist)
axs[1].set_title("green chanel of original image")
axs[2].plot(image_original_b_hist)
axs[2].set_title("blue chanel of original image")
plt.show()

fig, axs = plt.subplots(1, 3)
axs[0].plot(image_specified_r_hist)
axs[0].set_title("red chanel of specified image")
axs[1].plot(image_specified_g_hist)
axs[1].set_title("green chanel of specified image")
axs[2].plot(image_specified_b_hist)
axs[2].set_title("blue chanel of specified image")
plt.show()

# Creating right mapping for pixels of original image
final_image_r_mapping = histogram_matching(image_original_r_hist, image_specified_r_hist)
final_image_g_mapping = histogram_matching(image_original_g_hist, image_specified_g_hist)
final_image_b_mapping = histogram_matching(image_original_b_hist, image_specified_b_hist)

# Creating our final image chanels
final_image_r_chanel = final_image_r_mapping[image_original[:, :, 0]]
final_image_g_chanel = final_image_g_mapping[image_original[:, :, 1]]
final_image_b_chanel = final_image_b_mapping[image_original[:, :, 2]]

# Concatinating these 3 channls
final_image_r_chanel = final_image_r_chanel.reshape(image_original.shape[0], image_original.shape[1], 1)
final_image_g_chanel = final_image_g_chanel.reshape(image_original.shape[0], image_original.shape[1], 1)
final_image_b_chanel = final_image_b_chanel.reshape(image_original.shape[0], image_original.shape[1], 1)
final_image = np.concatenate((final_image_r_chanel, final_image_g_chanel, final_image_b_chanel), axis=2)

# Showing result
plt.imshow(final_image)
plt.show()

# Creating histogram for final image
image_final_r_hist = creat_histogram(final_image[:, :, 0])
image_final_g_hist = creat_histogram(final_image[:, :, 1])
image_final_b_hist = creat_histogram(final_image[:, :, 2])

# Showing histogram of final image
fig, axs = plt.subplots(1, 3)
axs[0].plot(image_final_r_hist)
axs[0].set_title("red chanel of specified image")
axs[1].plot(image_final_g_hist)
axs[1].set_title("green chanel of specified image")
axs[2].plot(image_final_b_hist)
axs[2].set_title("blue chanel of specified image")
plt.show()

# Saving our result 
fig.savefig('result_01.jpg')
result_02 = np.concatenate((final_image_b_chanel, final_image_g_chanel, final_image_r_chanel), axis=2)
cv2.imwrite('result_02.jpg', result_02)
