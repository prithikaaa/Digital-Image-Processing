import cv2
import numpy as np
import matplotlib.pyplot as plt


path = r"C:\Users\hp\Desktop\coding\dip\bone_scint.pgm"
image = cv2.imread(path)

def contrast_stretching(image):
    min_val, max_val = np.min(image), np.max(image)
    stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return stretched

def log_transform(image):
    image = image.astype(np.float64) 
    epsilon = 1e-5 
    c = 255 / (np.log(1 + np.max(image) + epsilon))  
    log_image = c * np.log(1 + image + epsilon)
    return np.uint8(np.clip(log_image, 0, 255)) 


def gamma_transform(image, gamma=1.0):
    gamma_corrected = np.power(image / 255.0, gamma) * 255
    return np.uint8(gamma_corrected)

def display_results(image, blue_hist, green_hist, red_hist, title):
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"{title} ({image.shape[1]}x{image.shape[0]})")
    plt.axis("on")

    plt.subplot(1, 2, 2)
    plt.plot(blue_hist, color='blue', label='Blue Channel')
    plt.plot(green_hist, color='green', label='Green Channel')
    plt.plot(red_hist, color='red', label='Red Channel')
    plt.title("Histogram")
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid()


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

contrast_image = contrast_stretching(image)
log_image = log_transform(image)
gamma_image = gamma_transform(image, gamma=0.5)  
blue_hist_orig, green_hist_orig, red_hist_orig = compute_histogram(image)
blue_hist_con, green_hist_con, red_hist_con = compute_histogram(contrast_image)
blue_hist_log, green_hist_log, red_hist_log = compute_histogram(log_image)
blue_hist_power, green_hist_power, red_hist_power = compute_histogram(gamma_image)
display_results(image, blue_hist_orig, green_hist_orig, red_hist_orig, "Original")
display_results(contrast_image, blue_hist_con, green_hist_con, red_hist_con, "Contrast sketched")
display_results(log_image, blue_hist_log, green_hist_log, red_hist_log, "log transform")
display_results(gamma_image, blue_hist_power, green_hist_power, red_hist_power, "Power law (γ=0.5)")
plt.show() 
