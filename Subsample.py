import cv2
import numpy as np
import matplotlib.pyplot as plt  

# Load the image
path = r"C:\Users\hp\Desktop\coding\dip\bone_scint.pgm"
image = cv2.imread(path)

# Subsample with factor 0.5
scale_factor = 0.5
color = image.shape[2]
new_height = int(image.shape[0] * scale_factor)
new_width = int(image.shape[1] * scale_factor)
subsampled = np.zeros((new_height, new_width, color), dtype=np.uint8)

for i in range(new_height):
    for j in range(new_width):
        orig_i = int(i / scale_factor)
        orig_j = int(j / scale_factor)
        for k in range(color):  
            subsampled[i][j][k] = image[orig_i, orig_j, k]

# Function to compute color histogram
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
blue_hist_sub, green_hist_sub, red_hist_sub = compute_histogram(subsampled)

# === Page 1: Original Image & Histogram ===
plt.figure(figsize=(10, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f"Original Image ({image.shape[1]}x{image.shape[0]})")
plt.axis("on")

# Histogram of Original Image
plt.subplot(1, 2, 2)
plt.plot(blue_hist_orig, color='blue', label='Blue Channel')
plt.plot(green_hist_orig, color='green', label='Green Channel')
plt.plot(red_hist_orig, color='red', label='Red Channel')
plt.title("Histogram of Original Image")
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# === Page 2: Subsampled Image & Histogram ===
plt.figure(figsize=(10, 6))

# Subsampled Image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(subsampled, cv2.COLOR_BGR2RGB))
plt.title(f"Subsampled Image ({subsampled.shape[1]}x{subsampled.shape[0]})")
plt.axis("on")

# Histogram of Subsampled Image
plt.subplot(1, 2, 2)
plt.plot(blue_hist_sub, color='blue', label='Blue Channel')
plt.plot(green_hist_sub, color='green', label='Green Channel')
plt.plot(red_hist_sub, color='red', label='Red Channel')
plt.title("Histogram of Subsampled Image")
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

