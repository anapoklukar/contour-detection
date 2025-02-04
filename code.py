import math
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys
from matplotlib.backends.backend_pdf import PdfPages

# gaussian_kernel: Function to generate a 2D Gaussian kernel
def gaussian_kernel(size, sigma):
    # Initialize the kernel as a 2D list filled with zeros
    kernel = [[0 for _ in range(size)] for _ in range(size)]
    
    # Calculate the center index of the kernel
    center = size // 2
    
    # Variable to keep track of the total sum for normalization
    total = 0
    
    # Populate the kernel using the Gaussian function
    for x in range(size):
        for y in range(size):
            # Compute the exponent for the Gaussian function
            exponent = -((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)
            
            # Calculate the Gaussian value and assign it to the kernel
            kernel[x][y] = math.exp(exponent) / (2 * math.pi * sigma ** 2)
            
            # Accumulate the sum of all kernel values for normalization
            total += kernel[x][y]
    
    # Normalize the kernel so that the sum of all elements equals 1
    for x in range(size):
        for y in range(size):
            kernel[x][y] /= total

    return kernel

# apply_kernel: Function to apply a 2D kernel to a 2D image
def apply_kernel(image, kernel):
    # Determine the size of the kernel
    kernel_size = len(kernel)
    
    # Calculate the padding needed based on the kernel size
    pad = kernel_size // 2
    
    # Get the dimensions of the input image
    width, height = len(image[0]), len(image)
    
    # Initialize the result image with zeros, having the same dimensions as the input image
    result = [[0 for _ in range(width)] for _ in range(height)]

    # Iterate over each pixel in the input image, excluding the padded border
    for x in range(pad, height - pad):
        for y in range(pad, width - pad):
            # Variable to accumulate the convolution result for the current pixel
            value = 0
            
            # Iterate over the kernel elements
            for kx in range(kernel_size):
                for ky in range(kernel_size):
                    # Apply the kernel to the corresponding image region
                    value += image[x + kx - pad][y + ky - pad] * kernel[kx][ky]
            
            # Assign the calculated value to the corresponding pixel in the result
            result[x][y] = value

    return result

# gradient: Function to compute the gradient magnitude and angle of an image
def gradient(image):
    # Define Sobel operators for gradient computation in the x and y directions
    Gx = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]  # Horizontal edge detection kernel

    Gy = [[1, 2, 1],
          [0, 0, 0],
          [-1, -2, -1]]  # Vertical edge detection kernel

    # Get the dimensions of the input image
    width, height = len(image[0]), len(image)

    # Apply Sobel kernels to compute gradients in the x and y directions
    grad_x = apply_kernel(image, Gx)  # Gradient in the x-direction
    grad_y = apply_kernel(image, Gy)  # Gradient in the y-direction

    # Initialize matrices for gradient magnitude and angle
    magnitude = [[0 for _ in range(width)] for _ in range(height)]
    angle = [[0 for _ in range(width)] for _ in range(height)]

    # Compute gradient magnitude and angle for each pixel
    for x in range(height):
        for y in range(width):
            # Calculate the gradient magnitude using the Pythagorean theorem
            mag = math.sqrt(grad_x[x][y] ** 2 + grad_y[x][y] ** 2)
            magnitude[x][y] = mag
            
            # Calculate the gradient angle in degrees using the arctangent function
            angle[x][y] = math.degrees(math.atan2(grad_y[x][y], grad_x[x][y]))

    # Return the gradient magnitude and angle as 2D lists
    return magnitude, angle

# non_maximum_suppression: Function to perform non-maximum suppression
def non_maximum_suppression(magnitude, angle):
    # Get the dimensions of the gradient magnitude image
    width, height = len(magnitude[0]), len(magnitude)

    # Initialize the result image with zeros
    result = [[0 for _ in range(width)] for _ in range(height)]

    # Iterate over the interior of the image, ignoring the border pixels
    for x in range(1, height - 1):
        for y in range(1, width - 1):
            # Map the angle to the range [0, 180)
            angle_dir = angle[x][y] % 180
            current = magnitude[x][y]  # Current pixel's gradient magnitude

            # Determine the two neighboring pixels to compare based on the angle direction
            if (0 <= angle_dir < 22.5) or (157.5 <= angle_dir <= 180):
                # Horizontal direction (neighbors to the left and right)
                neighbor1, neighbor2 = magnitude[x][y - 1], magnitude[x][y + 1]
            elif 22.5 <= angle_dir < 67.5:
                # Diagonal direction (top-right and bottom-left neighbors)
                neighbor1, neighbor2 = magnitude[x - 1][y + 1], magnitude[x + 1][y - 1]
            elif 67.5 <= angle_dir < 112.5:
                # Vertical direction (neighbors above and below)
                neighbor1, neighbor2 = magnitude[x - 1][y], magnitude[x + 1][y]
            else:
                # Diagonal direction (top-left and bottom-right neighbors)
                neighbor1, neighbor2 = magnitude[x - 1][y - 1], magnitude[x + 1][y + 1]

            # Suppress the current pixel if it is not greater than its neighbors
            if current >= neighbor1 and current >= neighbor2:
                result[x][y] = current  # Retain the current pixel value
            # Otherwise, it remains 0 (default initialization)

    return result

# hysteresis_thresholding: Function to apply hysteresis thresholding
def hysteresis_thresholding(image, low, high):
    # Get the dimensions of the input image
    width, height = len(image[0]), len(image)

    # Initialize the result image with zeros
    result = [[0 for _ in range(width)] for _ in range(height)]

    # Define intensity values for strong and weak edges
    strong, weak = 255, 75

    # Iterate over each pixel in the image
    for x in range(height):
        for y in range(width):
            if image[x][y] >= high:
                # Classify pixels with intensity >= high as strong edges
                result[x][y] = strong
            elif low <= image[x][y] < high:
                # Classify pixels with intensity between low and high as weak edges
                result[x][y] = weak
            # Pixels below the low threshold remain 0 (suppressed)

    return result

# edge_linking: Function to link weak edges to strong edges
def edge_linking(image):
    # Get the dimensions of the input image
    width, height = len(image[0]), len(image)

    # Define intensity values for strong and weak edges
    strong, weak = 255, 75

    # Iterate over the interior of the image, avoiding the border pixels
    for x in range(1, height - 1):
        for y in range(1, width - 1):
            # Check if the current pixel is a weak edge
            if image[x][y] == weak:
                # Check the 8-connected neighborhood for any strong edge
                if any(image[nx][ny] == strong for nx in range(x - 1, x + 2) for ny in range(y - 1, y + 2)):
                    # If connected to a strong edge, promote this pixel to a strong edge
                    image[x][y] = strong
                else:
                    # Otherwise, suppress the weak edge
                    image[x][y] = 0

    return image

# dynamic_thresholds: Function to compute dynamic thresholds based on gradient magnitude
def dynamic_thresholds(magnitude, low_ratio=0.50):
    # Flatten the 2D magnitude matrix into a 1D array
    mag_values = [magnitude[x][y] for x in range(len(magnitude)) for y in range(len(magnitude[0]))]
    
    # Convert to a NumPy array for easier computation
    mag_values = np.array(mag_values)
    
    # Compute a dynamic high threshold based on the mean and standard deviation
    mean_val = np.mean(mag_values)
    std_dev = np.std(mag_values)
    high_thresh = mean_val + 2 * std_dev  # Mean + 2 * standard deviation
    
    # Ensure the high threshold is within valid bounds
    high_thresh = min(high_thresh, mag_values.max())
    
    # Compute the low threshold as a fraction of the high threshold
    low_thresh = low_ratio * high_thresh
    
    return high_thresh, low_thresh


# Canny Edge Detection Function
def canny_edge_detection(image, kernel_size=5, sigma=1):
    
    # Step 1: Gaussian Smoothing
    kernel = gaussian_kernel(kernel_size, sigma) # Generate Gaussian kernel
    smoothed = apply_kernel(image, kernel) # Apply Gaussian kernel

    # Step 2: Gradient Calculation
    magnitude, angle = gradient(smoothed) # Compute gradient magnitude and angle

    # Step 3: Non-Maximum Suppression
    thin_edges = non_maximum_suppression(magnitude, angle)
    
    # Compute dynamic thresholds based on the gradient magnitude
    high_thresh, low_thresh = dynamic_thresholds(magnitude)
    
    # Step 4: Hysteresis Thresholding
    thresholded = hysteresis_thresholding(thin_edges, low_thresh, high_thresh)

    # Step 5: Edge Linking
    edges = edge_linking(thresholded)
   
    # Display the original image, image after detecting the edges, image after edge linking
    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(thin_edges, cmap='gray')
    plt.title('Detected Edges')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Linking')
    plt.axis('off')
    
    # Save the image as PNG file
    plt.imsave('edge_linking.png', edges, cmap='gray')
    plt.show()
    
    return edges

# evaluate_metrics: Function to evaluate Precision, Sensitivity, and F1 Score
def evaluate_metrics(edges, edges_cv):
    # Mask for pixels with value >= 255 (edge pixels)
    mask = edges >= 255  # Same mask applies for both edges and edges_cv
    
    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = np.sum(np.logical_and(edges >= 255, edges_cv >= 255))  # Correctly detected edge pixels
    FP = np.sum(np.logical_and(~mask, edges_cv >= 255))          # False edge detections (non-edge pixels detected as edges)
    FN = np.sum(np.logical_and(mask, edges_cv < 255))            # Missed edge pixels (true edges not detected)

    # Precision, Sensitivity, and F1 Score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    return precision, sensitivity, f1_score

# process_images_in_folder: Function to process images in a folder
def process_images_in_folder(image_folder, low_thresholds=range(10, 400, 10), high_thresholds=range(40, 500, 10), max_images=10):
    best_f1_overall = 0
    best_image = ""
    avg_precision = 0
    avg_sensitivity = 0
    avg_f1_score = 0
    total_images = 0

    # Walk through the directory to get all images
    image_count = 0  # To keep track of the number of images processed
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
                if image_count >= max_images:
                    break  # Stop if we've processed the maximum number of images
                
                image_path = os.path.join(root, file) # Get the full path of the image
                
                # Load the image using Pillow (PIL) and preprocess
                img = Image.open(image_path).convert("L")  # Convert to grayscale
                
                img_data = list(img.getdata())  # Get pixel values
                width, height = img.size  # Get image dimensions
                img_matrix = [img_data[i * width:(i + 1) * width] for i in range(height)]  # Convert to 2D matrix

                # Sigma size is 0.5% of the minimum dimension
                sigma = int(0.005 * min(width, height))

                # Kernel size is the smallest odd integer greater than 6 times the sigma value
                kernel_size = int(6 * sigma) + 1

                # Run custom Canny Edge Detection
                edges = canny_edge_detection(img_matrix, kernel_size=kernel_size, sigma=sigma)

                # Load the image using OpenCV
                image_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image_cv = cv2.GaussianBlur(image_cv, (kernel_size, kernel_size), sigma)  # Apply Gaussian blur

                # Initialize variables to track the best metrics and corresponding thresholds
                best_precision = 0
                best_sensitivity = 0
                best_f1 = 0

                # Iterate over different pairs of thresholds
                for low_thresh in low_thresholds:
                    for high_thresh in high_thresholds:
                        # Apply Canny edge detection using OpenCV with current thresholds
                        edges_cv = cv2.Canny(image_cv, low_thresh, high_thresh)
                        edges_cv = np.array(edges_cv)
                        edges = np.array(edges)

                        # Evaluate Precision, Sensitivity, and F1 score for the current thresholds
                        precision, sensitivity, f1_score = evaluate_metrics(edges, edges_cv)
                        
                        # Update the best metrics and thresholds if a better result is found
                        if f1_score > best_f1:
                            best_f1 = f1_score
                            best_precision = precision
                            best_sensitivity = sensitivity
                            best_thresholds = (low_thresh, high_thresh)

                # Track overall statistics
                avg_precision += best_precision
                avg_sensitivity += best_sensitivity
                avg_f1_score += best_f1
                total_images += 1

                # Check if this image is the best performing image
                if best_f1 > best_f1_overall:
                    best_f1_overall = best_f1
                    best_image = image_path

                image_count += 1

                if image_count >= max_images:
                    break  # Stop processing after the first 10 images

        if image_count >= max_images:
            break  # Stop if we've processed the maximum number of images

    # Calculate the average Precision, Sensitivity, and F1 score
    if total_images > 0:
        avg_precision /= total_images
        avg_sensitivity /= total_images
        avg_f1_score /= total_images

    # Output the overall statistics
    print(f"\nOverall Statistics across all images:")
    print(f"Average Precision: {avg_precision}")
    print(f"Average Sensitivity: {avg_sensitivity}")
    print(f"Average F1 Score: {avg_f1_score}")


# process_single_image: Function to process a single image
def process_single_image(image_path, low_thresholds=range(10, 400, 10), high_thresholds=range(40, 500, 10)):

    # Load the image using Pillow (PIL) and preprocess
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img_data = list(img.getdata())  # Get pixel values
    width, height = img.size  # Get image dimensions
    img_matrix = [img_data[i * width:(i + 1) * width] for i in range(height)]  # Convert to 2D matrix

    # Sigma size is 0.5% of the minimum dimension
    sigma = int(0.005 * min(width, height))

    # Kernel size is the smallest odd integer greater than 6 times the sigma value
    kernel_size = int(6 * sigma) + 1

    # Run custom Canny Edge Detection
    edges = canny_edge_detection(img_matrix, kernel_size=kernel_size, sigma=sigma)

    # Load the image using OpenCV
    image_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_cv = cv2.GaussianBlur(image_cv, (kernel_size, kernel_size), sigma)  # Apply Gaussian blur

    # Initialize variables to track the best metrics and corresponding thresholds
    best_precision = 0
    best_sensitivity = 0
    best_f1 = 0
    best_thresholds = (0, 0)

    # Iterate over different pairs of thresholds
    for low_thresh in low_thresholds:
        for high_thresh in high_thresholds:
            # Apply Canny edge detection using OpenCV with current thresholds
            edges_cv = cv2.Canny(image_cv, low_thresh, high_thresh)
            edges_cv = np.array(edges_cv)
            edges = np.array(edges)

            # Evaluate Precision, Sensitivity, and F1 score for the current thresholds
            precision, sensitivity, f1_score = evaluate_metrics(edges, edges_cv)

            # Update the best metrics and thresholds if a better result is found
            if f1_score > best_f1:
                best_f1 = f1_score
                best_precision = precision
                best_sensitivity = sensitivity
                best_thresholds = (low_thresh, high_thresh)
                
    # Plot the result from my custom Canny Edge Detection and best result by OpenCV's Canny Edge Detection
    edges_cv = cv2.Canny(image_cv, best_thresholds[0], best_thresholds[1])
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(edges, cmap='gray')
    plt.title('Custom Canny Edge Detection')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(edges_cv, cmap='gray')
    plt.title('OpenCV Canny Edge Detection')
    plt.axis('off')
    plt.show()

    # Output the statistics for the processed image
    print(f"\nProcessed Image: {image_path}")
    print(f"Precision: {best_precision}")
    print(f"Sensitivity: {best_sensitivity}")
    print(f"F1 Score: {best_f1}")

if __name__ == "__main__":
    image_path = sys.argv[1]  # Get the image path from the command line argument
    process_single_image(image_path)
