import cv2
import numpy as np
import matplotlib.pyplot as plt  
path=r"C:\Users\hp\Desktop\coding\dip\bone_scint.pgm" 
image=cv2.imread(path)
invert=cv2.imread(path)
height=image.shape[0]
width=image.shape[1]
color=image.shape[2]
print(f"height:{height}")
print(f"width:{width}")
print(f"channels:{color}")
for i in range(height):  
    for j in range(width):
        for k in range(color):
            invert[i, j, k]=255-image[i, j, k]

def compute_histogram(img):
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]
    
    blue_hist = [0] * 256
    green_hist = [0] * 256
    red_hist = [0] * 256
    
    height, width = img.shape[:2]

    for i in range(height):  
        for j in range(width):
            blue_hist[blue[i, j]] += 1
            green_hist[green[i, j]] += 1
            red_hist[red[i, j]] += 1
    
    return blue_hist, green_hist, red_hist

# Compute histograms
blue_hist_orig, green_hist_orig, red_hist_orig = compute_histogram(image)
blue_hist_sub, green_hist_sub, red_hist_sub = compute_histogram(invert)
plt.figure(figsize=(10, 6))

# Subsampled Image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(invert, cv2.COLOR_BGR2RGB))
plt.title(f"inverted Image ({invert.shape[1]}x{invert.shape[0]})")
plt.axis("on")

# Histogram of Subsampled Image
plt.subplot(1, 2, 2)
plt.plot(blue_hist_sub, color='blue', label='Blue Channel')
plt.plot(green_hist_sub, color='green', label='Green Channel')
plt.plot(red_hist_sub, color='red', label='Red Channel')
plt.title("Histogram of inverted Image")
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


