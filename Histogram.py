import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
path = r"C:\Users\hp\Pictures\Screenshots\damdata.png"  
image=cv2.imread(path)

if image is None:
    print("Error: Unable to load image.")
    exit()

blue=image[:,:,0]
green=image[:,:,1]
red=image[:,:,2]
blue_hist=[0]*256
green_hist=[0]*256
red_hist=[0]*256
height=image.shape[0]
width=image.shape[1]
colors=image.shape[2]
for i in range(height):  
    for j in range(width):
        blue_hist[blue[i, j]]+=1
        green_hist[green[i, j]]+=1
        red_hist[red[i, j]]+=1

plt.figure(figsize=(10, 6))
plt.plot(blue_hist, color='blue', label='Blue Channel')
plt.plot(green_hist, color='green', label='Green Channel')
plt.plot(red_hist, color='red', label='Red Channel')

plt.title('Color Histogram (Manually Computed)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.grid()
plt.show()
