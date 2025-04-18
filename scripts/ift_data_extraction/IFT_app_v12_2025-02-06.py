import os  # Importing the operating system interface module to enable cross‐platform file and folder operations, such as creating directories and joining file paths, which is necessary for organizing data outputs
import csv  # Importing the module that provides functionality to read from and write to comma-separated value files, allowing structured data storage and exchange with other software systems
import cv2  # Importing the computer vision library that supplies image processing algorithms and tools for tasks such as edge detection, contour analysis, and drawing shapes on images
import numpy as np  # Importing the numerical computing library that supports multi‐dimensional arrays and a wide range of mathematical functions, used here for processing image data and performing numerical computations
import matplotlib.pyplot as plt  # Importing the plotting library used to generate visual representations of data, such as graphs and annotated images, for analysis and presentation purposes
from math import sqrt  # Importing the square root function to perform mathematical calculations needed for geometric computations, such as determining circle radii from fitted parameters
from datetime import datetime  # Importing the date and time functionality to obtain the current timestamp, which is used for timestamping files and folders for traceability of processed data
from scipy import interpolate  # Importing the interpolation submodule to create smooth curves from discrete data points, crucial for reconstructing droplet profiles from edge-detected coordinates
from scipy.integrate import solve_ivp  # Importing the function that numerically solves initial value problems for ordinary differential equations, used here to simulate droplet shapes via the Young–Laplace equation
from scipy.optimize import minimize_scalar  # Importing the optimization function for finding the minimum of a scalar function, which is applied to tune physical model parameters to experimental data

# Defining a class to manage folders in the file system for organizing results by date and measurement time
class FolderManager:
    @staticmethod
    def create_date_folder(base_path):
        # Generating a string that represents the current date in a standardized year-month-day format,
        # which facilitates chronological organization of output files
        date_str = datetime.now().strftime('%Y-%m-%d')
        # Combining the base directory with the date string to form a complete path where output will be stored,
        # ensuring that data from different dates do not mix
        date_folder = os.path.join(base_path, date_str)
        # Creating the directory and any required intermediate directories; if the directory already exists, do not raise an error,
        # which ensures idempotent behavior when running multiple analyses on the same day
        os.makedirs(date_folder, exist_ok=True)
        # Returning the constructed directory path so that subsequent functions can save their outputs there
        return date_folder

    @staticmethod
    def create_measurement_folder(base_path, hour_str):
        # Constructing a folder name that includes the current time (hour, minute, second) to uniquely identify the measurement session,
        # which aids in organizing data acquired at different times during the same day
        folder_name = f"Measurement_{hour_str}"
        # Combining the base path with the newly constructed folder name to generate the full path for the measurement session,
        # ensuring that each measurement session is stored in its own separate folder for clarity
        folder_path = os.path.join(base_path, folder_name)
        # Creating the folder (and any necessary parent directories) without error if it already exists,
        # which guarantees that the data saving process will not be interrupted by file system errors
        os.makedirs(folder_path, exist_ok=True)
        # Returning the full path of the measurement folder so that the main processing function can use it for output file storage
        return folder_path

# Defining a class for detecting and classifying features within images using contour analysis and edge detection methods
class DetectionClassifier:
    def __init__(self, min_contour_area=300, line_length_threshold=20):
        # Storing the minimum area threshold that distinguishes significant image regions from noise,
        # which helps in ignoring irrelevant small objects during image analysis
        self.min_contour_area = min_contour_area
        # Storing the minimum required length for a detected line segment to be considered a valid feature,
        # ensuring that only sufficiently long structures are used for further classification
        self.line_length_threshold = line_length_threshold

    def classify(self, contours, image_resized):
        # Initializing a variable to keep track of the lowest vertical coordinate detected among valid contours,
        # which is used later to determine the region of interest in the image
        lowest_y = 0
        # Setting a flag to indicate whether a candidate needle (or similar feature) has been detected by analyzing contours,
        # which is essential for branching the processing logic based on detection outcome
        needle_detected = False
        # Iterating over all contour objects obtained from image segmentation,
        # which represent continuous boundaries detected in the image that may correspond to physical objects
        for contour in contours:
            # Measuring the area enclosed by the current contour to filter out insignificant regions,
            # thereby ensuring that only contours representing potentially meaningful features are further considered
            if cv2.contourArea(contour) > self.min_contour_area:
                # Indicating that a candidate feature has been found, triggering further processing for needle detection
                needle_detected = True
                # Drawing the contour on the scaled image in a visually distinct color to aid in debugging and verification,
                # which makes it easier to visually confirm that the algorithm is detecting the expected regions
                cv2.drawContours(image_resized, [contour], -1, (0, 255, 0), 1)
                # Finding the maximum vertical coordinate in the current contour by examining each point,
                # which helps in determining the lowest part of the detected feature and in segmenting the image appropriately
                lowest_y = max(lowest_y, max(contour, key=lambda p: p[0][1])[0][1])
        # If no valid contours have been found that meet the area criteria,
        # the function returns an early classification indicating that the needle was not detected, thus preventing further unnecessary processing
        if not needle_detected:
            return 'Needle not detected', image_resized, lowest_y
        # Adjusting the identified vertical coordinate slightly by a fixed offset,
        # which helps to ensure that subsequent image processing focuses on a region just below the detected feature
        adjusted_y = lowest_y + 4
        # Drawing a horizontal line across the image at the adjusted vertical position using a contrasting color,
        # which visually demarcates the region of interest for later edge detection and feature extraction steps
        cv2.line(image_resized, (0, adjusted_y), (image_resized.shape[1], adjusted_y), (0, 0, 255), 1)
        # Cropping the image to isolate the region below the drawn line,
        # since this area is likely to contain the droplet or stream that needs to be further analyzed
        below_red_line = image_resized[adjusted_y + 1:, :, :]
        # Converting the cropped region from color to grayscale to simplify the subsequent edge detection process,
        # as grayscale images reduce complexity and enhance contrast for edge detection algorithms
        gray_below = cv2.cvtColor(below_red_line, cv2.COLOR_BGR2GRAY)
        # Applying the Canny edge detection algorithm to the grayscale region to identify significant changes in intensity,
        # which typically correspond to physical boundaries and are critical for extracting shape information
        edges_below = cv2.Canny(gray_below, 50, 150)
        # Extracting the contours from the edge-detected image to capture the outlines of distinct objects within the cropped region,
        # which will later be analyzed to determine if the feature represents a droplet or a flowing stream
        contours_edges, _ = cv2.findContours(edges_below, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # If no contours are found in the edge-detected region, it implies that neither a droplet nor a stream is present,
        # and the function returns an appropriate classification to indicate this outcome
        if len(contours_edges) == 0:
            return 'Droplet or stream not detected', image_resized, lowest_y
        # Initializing a flag to monitor whether a line segment with sufficient length has been detected in the edge region,
        # which is used to distinguish between a droplet (typically round) and a stream or gel (which exhibits linear characteristics)
        line_detected = False
        # Iterating over each contour obtained from the edge-detection process to analyze its structure,
        # which enables the function to decide if a long continuous line exists that would indicate a different physical phenomenon
        for contour in contours_edges:
            # Compensating for the vertical offset introduced earlier by adding a constant to the contour coordinates,
            # so that the detected contour aligns correctly with the original image coordinate system for visualization
            contour_shifted = contour + np.array([[0, adjusted_y + 1]])
            # Drawing the shifted contour on the image using a distinct color to provide visual feedback on the detected edge structure,
            # which assists in validating the correctness of the contour extraction step during debugging
            cv2.drawContours(image_resized, [contour_shifted], -1, (255, 0, 0), 1)
            # Looping over consecutive point pairs in the contour to evaluate the length of each segment,
            # which is important to determine if any part of the contour forms a long, continuous line that might represent a stream
            for i in range(len(contour) - 1):
                # Extracting two successive points that define a small segment of the contour,
                # where each point represents a coordinate in the two-dimensional image space
                pt1, pt2 = tuple(contour[i][0]), tuple(contour[i + 1][0])
                # Calculating the Euclidean distance between the two points to measure the length of the segment,
                # which involves applying the Pythagorean theorem to compute the direct distance between the coordinates
                if np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2) > self.line_length_threshold:
                    # If the computed distance exceeds the predefined threshold, it is considered a significant linear feature,
                    # indicating that the observed structure is more likely a stream or gel rather than a simple droplet
                    line_detected = True
                    # Redrawing the entire shifted contour in a different color to highlight the presence of this long segment,
                    # which visually confirms the detection of a linear structure for further interpretation
                    cv2.drawContours(image_resized, [contour_shifted], -1, (0, 255, 255), 1)
                    # Exiting the inner loop early because the presence of one sufficiently long segment is enough to classify the feature,
                    # thereby optimizing processing time by not analyzing all segments unnecessarily
                    break
        # Based on whether a long line segment was detected, returning the corresponding classification along with the annotated image,
        # which informs downstream processing whether to treat the feature as a droplet or as a stream/gel with different properties
        if line_detected:
            return 'Stream/Gel', image_resized, lowest_y
        else:
            return 'Droplet Detected', image_resized, lowest_y

# Defining a class that applies various image preprocessing techniques to enhance the quality and extract key features
class ImagePreprocessor:
    # Declaring constants for configuring noise reduction and smoothing filters that help in cleaning the image data
    MEDIAN_BLUR_KERNEL_SIZE = 5  # Specifying the size of the kernel for median filtering to remove impulsive noise while preserving edges
    BILATERAL_FILTER_D = 9  # Defining the diameter for the bilateral filter, which smooths the image while maintaining sharp boundaries
    BILATERAL_FILTER_SIGMA_COLOR = 75  # Setting the color space standard deviation in the bilateral filter to control how colors are mixed during smoothing
    BILATERAL_FILTER_SIGMA_SPACE = 75  # Setting the coordinate space standard deviation to determine the extent of spatial smoothing in the bilateral filter
    NON_LOCAL_MEANS_H = 10  # Specifying the filter strength parameter for the non-local means algorithm to reduce noise based on patch similarity

    def __init__(self, image):
        # Storing the input image that is to be processed, which may be either a color or grayscale image
        self.image = image
        # Initializing a placeholder for the processed image that will be generated after applying denoising and thresholding operations
        self.processed = None
        # Setting up variables to later hold the positions of the highest and lowest significant features (typically bright pixels)
        self.highest = None
        self.lowest = None
        # Reserving storage for the horizontal extremes (leftmost and rightmost positions) of the feature region,
        # which are used to define the boundaries of the droplet or shape under analysis
        self.leftmost = None
        self.rightmost = None
        # Initializing a variable to eventually hold the computed vertical axis of symmetry, which aids in splitting the shape into mirrored halves for analysis
        self.axis_of_symmetry = None

    @staticmethod
    def apply_maximum_denoising(image):
        # Converting a color image to grayscale if necessary, since many denoising algorithms are more effective on single-channel images
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        # Applying a median blur filter to the grayscale image to remove isolated noise spikes while keeping edges sharp,
        # which is particularly useful when the noise is non-Gaussian in nature
        denoised = cv2.medianBlur(gray, ImagePreprocessor.MEDIAN_BLUR_KERNEL_SIZE)
        # Further smoothing the image using a bilateral filter that preserves edges by considering both spatial proximity and color similarity,
        # ensuring that important structural details are not lost during the noise reduction process
        denoised = cv2.bilateralFilter(denoised,
                                       d=ImagePreprocessor.BILATERAL_FILTER_D,
                                       sigmaColor=ImagePreprocessor.BILATERAL_FILTER_SIGMA_COLOR,
                                       sigmaSpace=ImagePreprocessor.BILATERAL_FILTER_SIGMA_SPACE)
        # Applying the fast non-local means denoising method to exploit redundancy in image patches,
        # which enhances the overall clarity of the image while effectively reducing noise artifacts
        denoised = cv2.fastNlMeansDenoising(denoised, h=ImagePreprocessor.NON_LOCAL_MEANS_H)
        # Returning the fully denoised image, which is now better suited for subsequent thresholding and feature extraction operations
        return denoised

    @staticmethod
    def apply_morphological_closing(image):
        # Creating an elliptical structuring element with specified dimensions that is used to perform morphological operations,
        # which helps in filling small holes or gaps in the image regions representing the object of interest
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Applying the morphological closing operation (dilation followed by erosion) to the image,
        # which consolidates nearby bright regions and eliminates small dark spots, thereby refining the object boundaries
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def apply_adaptive_threshold(image):
        # Applying adaptive thresholding using a Gaussian weighted sum of neighborhood pixels,
        # which allows the algorithm to account for varying illumination across the image and produces a robust binary image
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)

    @staticmethod
    def apply_scharr(image):
        # Calculating the gradient of the image intensity in the horizontal direction using the Scharr operator,
        # which provides a more accurate estimation of derivatives compared to the Sobel operator, especially for small details
        scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        # Similarly, calculating the gradient in the vertical direction to capture edge information along that axis,
        # ensuring that both horizontal and vertical features are detected effectively
        scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
        # Converting the computed gradients to an absolute 8-bit scale so that they can be visualized and further processed,
        # as this conversion normalizes the gradient values into a standard image format
        return cv2.convertScaleAbs(scharr_x), cv2.convertScaleAbs(scharr_y)

    @staticmethod
    def threshold_non_white_to_black(image):
        # Applying a binary threshold that converts all pixel values below a high threshold into black,
        # which effectively isolates only the pixels that are nearly white and suppresses all other information
        _, binary_image = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)
        # Returning the resulting binary image, where the areas of interest (originally white) are preserved while all other regions are set to black
        return binary_image

    @staticmethod
    def find_vertical_symmetry_axis(image):
        # Normalizing the binary image so that the pixel intensity becomes either 0 or 1,
        # which simplifies the computation of the symmetry axis by enabling the use of arithmetic mean on the positions of bright pixels
        binary_image = image / 255
        # Locating the positions (coordinates) in the image where the normalized pixel value is 1 (indicating a white pixel),
        # which represent the locations of significant features that are used to compute the center of mass horizontally
        y_coords, x_coords = np.where(binary_image == 1)
        # If no bright pixels are found, it implies that the image does not contain any features to analyze,
        # so the function returns None to indicate that the symmetry axis cannot be computed
        if len(x_coords) == 0:
            return None
        # Computing the mean of the x-coordinates of all white pixels, which yields the vertical line that best splits the image into two balanced halves,
        # serving as an approximation of the axis of symmetry for the detected object
        return np.mean(x_coords)

    @staticmethod
    def find_highest_and_lowest_white_pixels(image, axis_of_symmetry):
        # Rounding the computed symmetry axis to the nearest integer to use it as a column index in the image matrix,
        # which is necessary since pixel indices must be integers
        axis_of_symmetry = int(round(axis_of_symmetry))
        # Validating that the computed column index lies within the valid horizontal range of the image,
        # ensuring that subsequent operations on this column will not cause an index out-of-bound error
        if axis_of_symmetry < 0 or axis_of_symmetry >= image.shape[1]:
            return None, None
        # Extracting all pixel intensity values from the specific column corresponding to the symmetry axis,
        # since these values will be analyzed to determine the vertical extent of the object in that column
        column_pixels = image[:, axis_of_symmetry]
        # Finding the row indices where the pixel value is 255 (white), which identifies the vertical positions of the object’s boundary in that column,
        # and these indices are used to determine the topmost and bottommost extents of the feature
        white_pixel_indices = np.where(column_pixels == 255)[0]
        # If no white pixels are found in this column, then it is impossible to define the vertical boundaries,
        # so the function returns None values to indicate the failure of this step
        if len(white_pixel_indices) == 0:
            return None, None
        # Returning the first (topmost) and last (bottommost) indices from the array of white pixel positions,
        # which represent the highest and lowest points of the object along the column, respectively
        return white_pixel_indices[0], white_pixel_indices[-1]

    @staticmethod
    def find_largest_white_shape(image, highest_y, min_area=500):
        # Creating a blank mask with the same dimensions as the input image, which will be used to isolate a specific region of interest
        mask_upper = np.zeros_like(image)
        # Defining a buffer size to exclude a small margin below the highest detected feature,
        # ensuring that only the relevant part of the image is considered for shape extraction
        buffer = 15
        # Determining the row index where the mask should start based on the highest feature position and buffer,
        # and ensuring that this index does not exceed the image boundaries
        start_y = highest_y + buffer if highest_y + buffer < image.shape[0] else image.shape[0] - 1
        # Filling the mask from the top of the image down to the computed row index with a constant value (white),
        # which effectively marks the upper portion of the image as the region where the object of interest is expected to be found
        mask_upper[:start_y, :] = 255
        # Applying a bitwise AND operation between the original image and the mask to isolate the region of interest,
        # thereby suppressing features outside the expected area and simplifying subsequent contour detection
        masked_image = cv2.bitwise_and(image, mask_upper)
        # Extracting contours from the masked image using an algorithm that retrieves only the external boundaries,
        # which is suitable for identifying the overall shape of the object while ignoring internal details
        contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Initializing variables to keep track of the largest detected contour and its area,
        # which will be used to select the most prominent white shape that likely corresponds to the droplet or object of interest
        largest_contour = None
        max_area = min_area
        # Iterating over each contour found in the masked image to evaluate its area,
        # which helps in filtering out insignificant shapes that do not meet the minimum size criteria
        for contour in contours:
            # Calculating the area enclosed by the current contour, which quantifies the size of the detected shape
            area = cv2.contourArea(contour)
            # If the area of the current shape is larger than the minimum required area and larger than any previously found shape,
            # update the record to mark this contour as the current best candidate for the object of interest
            if area > max_area:
                largest_contour = contour
                max_area = area
        # If a valid large contour has been found, proceed to create a mask that highlights only this shape,
        # which will later be used to extract further measurements from the object
        if largest_contour is not None:
            mask = np.zeros_like(image)
            # Drawing the largest contour onto the blank mask and filling it completely,
            # so that the region corresponding to the object is clearly isolated from the rest of the image
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            # Returning both the mask (which now contains only the object) and the contour itself for further analysis
            return mask, largest_contour
        # If no contour satisfies the area condition, return None values to indicate that the object could not be isolated
        return None, None

    @staticmethod
    def find_widest_row_in_band(image_x, adjusted_highest_y, lowest_y):
        # Initializing a variable to record the maximum horizontal span found among rows within the specified vertical band,
        # which will help in identifying the row that best represents the overall width of the object
        max_width = 0
        # Reserving a placeholder to record the vertical position (row index) where the maximum width is observed,
        # so that this information can be used for aligning measurements and further processing
        best_row_y = None
        # Initializing placeholders to record the leftmost and rightmost pixel positions in the identified row,
        # which are critical for calculating the horizontal extent of the object under analysis
        leftmost_pixel_band = None
        rightmost_pixel_band = None
        # Iterating over each row in the image segment defined by the adjusted top and bottom boundaries,
        # which focuses the search on the region where the object is expected to be located
        for row_y in range(adjusted_highest_y, lowest_y):
            # Extracting the row of pixel values from the processed image,
            # where each value indicates whether the pixel is part of the object (white) or the background (black)
            row = image_x[row_y, :]
            # Identifying the column indices in the row where the pixel value indicates presence of the object,
            # thus highlighting the horizontal extent of the object in that particular row
            white_pixel_indices = np.where(row == 255)[0]
            # If there are at least two white pixels in the row, then a horizontal span can be computed,
            # which is necessary to quantify the object's width in that row
            if len(white_pixel_indices) > 1:
                # Computing the difference between the farthest right and left white pixel positions,
                # which yields the width of the object as observed in the current row
                width = white_pixel_indices[-1] - white_pixel_indices[0]
                # If the computed width exceeds the maximum width recorded so far,
                # update the maximum width and record the current row as the one with the best representation of the object’s horizontal extent
                if width > max_width:
                    max_width = width
                    best_row_y = row_y
                    left_x = white_pixel_indices[0]
                    right_x = white_pixel_indices[-1]
                    # Storing the row index and corresponding column positions for the left and right edges,
                    # which will be used later to define the lateral boundaries of the object for further analysis
                    leftmost_pixel_band = (row_y, left_x)
                    rightmost_pixel_band = (row_y, right_x)
        # Returning the coordinates of the leftmost and rightmost points in the row with maximum width,
        # along with the row index itself, to be used for subsequent measurements and visualizations
        return leftmost_pixel_band, rightmost_pixel_band, best_row_y

    @staticmethod
    def process_image_with_marking_in_band(image_with_lines):
        # Applying a morphological closing operation to the input image to consolidate nearby regions and eliminate small gaps,
        # which produces a smoother image that is easier to threshold and analyze for continuous features
        morphed_image = ImagePreprocessor.apply_morphological_closing(image_with_lines)
        # Performing adaptive thresholding on the morphologically processed image to convert it into a binary format,
        # which simplifies the identification of features by representing them as high-contrast regions against a dark background
        adaptive_thresh = ImagePreprocessor.apply_adaptive_threshold(morphed_image)
        # Calculating the horizontal and vertical gradients using the Scharr operator,
        # which provides enhanced edge detection capabilities critical for accurately delineating object boundaries
        scharr_x, scharr_y = ImagePreprocessor.apply_scharr(morphed_image)
        # Converting the vertical gradient image to a binary format to isolate prominent edge features,
        # thereby facilitating the determination of the object’s vertical extents
        scharr_y_black = ImagePreprocessor.threshold_non_white_to_black(scharr_y)
        # Similarly, thresholding the horizontal gradient image to focus on significant features along that axis,
        # which assists in later stages of feature extraction and measurement
        scharr_x_black = ImagePreprocessor.threshold_non_white_to_black(scharr_x)
        # Computing the vertical axis of symmetry by analyzing the thresholded vertical gradient image,
        # which provides a reference line to partition the object into symmetric halves and aids in alignment of measurements
        axis_of_symmetry = ImagePreprocessor.find_vertical_symmetry_axis(scharr_y_black)
        # If no symmetry axis can be determined (due to insufficient bright features), the function terminates early,
        # since subsequent analysis depends on having a well-defined centerline
        if axis_of_symmetry is None:
            return None
        # Determining the highest and lowest vertical positions of the object along the computed symmetry column,
        # which are used to define the vertical bounds of the region that contains the droplet or shape of interest
        highest_y, lowest_y = ImagePreprocessor.find_highest_and_lowest_white_pixels(scharr_y_black, axis_of_symmetry)
        # If either the top or bottom boundary cannot be determined, terminate the process to avoid erroneous measurements
        if highest_y is None or lowest_y is None:
            return None
        # Adjusting the highest boundary by a fixed offset to ensure that the analysis region fully encompasses the object,
        # while also remaining within the image dimensions to avoid index errors
        adjusted_highest_y = highest_y + 25 if highest_y + 25 < scharr_y_black.shape[0] else scharr_y_black.shape[0] - 1
        # Identifying the row within the specified band that exhibits the greatest horizontal spread,
        # which is assumed to correspond to the widest part of the object and is critical for determining its lateral dimensions
        leftmost_pixel_band, rightmost_pixel_band, best_row_y = ImagePreprocessor.find_widest_row_in_band(scharr_x_black, adjusted_highest_y, lowest_y)
        # Returning a comprehensive tuple that includes various processed images and key measurement data,
        # such as the thresholded gradient images, adjusted boundaries, symmetry axis, and the binary image produced by adaptive thresholding;
        # this data is used in later stages of analysis and visualization
        return (scharr_x_black, adjusted_highest_y, lowest_y, leftmost_pixel_band,
                rightmost_pixel_band, highest_y, best_row_y, axis_of_symmetry, adaptive_thresh)

    def process(self):
        # Checking whether the input image is in color (with multiple channels) and converting it to grayscale if necessary,
        # because many of the subsequent processing algorithms are designed to operate on single-channel images
        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            # If the image is already in grayscale, create a duplicate to preserve the original data while processing
            gray = self.image.copy()

        # Applying a sequence of denoising and smoothing operations to reduce noise and enhance the features of interest,
        # thereby improving the reliability of subsequent thresholding and edge detection steps
        denoised = ImagePreprocessor.apply_maximum_denoising(self.image)
        # Using morphological closing to eliminate small holes and connect disjoint regions in the denoised image,
        # which leads to a more coherent representation of the object boundaries
        morphed = ImagePreprocessor.apply_morphological_closing(denoised)
        # Converting the smoothed image into a binary format using adaptive thresholding,
        # which separates the object from the background even under non-uniform lighting conditions
        adaptive_thresh = ImagePreprocessor.apply_adaptive_threshold(morphed)
        # Computing horizontal and vertical gradients using the Scharr operator to detect edges,
        # since precise gradient computation is crucial for accurately locating object boundaries
        scharr_x, scharr_y = ImagePreprocessor.apply_scharr(morphed)
        # Converting the vertical gradient image to binary to isolate regions of significant change,
        # which are indicative of the object’s upper and lower boundaries
        scharr_y_black = ImagePreprocessor.threshold_non_white_to_black(scharr_y)

        # Determining the vertical axis of symmetry from the binary gradient image,
        # which is used to partition the object and align measurements consistently across different images
        self.axis_of_symmetry = ImagePreprocessor.find_vertical_symmetry_axis(scharr_y_black)
        # If the symmetry axis cannot be determined, an error is raised to signal that essential information is missing,
        # thus preventing further processing that would yield unreliable results
        if self.axis_of_symmetry is None:
            raise ValueError("Failing to determine symmetry axis")
        # Extracting the highest and lowest vertical positions of the object along the symmetry column,
        # which provides the vertical extent necessary for computing the object’s height and other dimensional properties
        self.highest, self.lowest = ImagePreprocessor.find_highest_and_lowest_white_pixels(scharr_y_black, self.axis_of_symmetry)
        # Validating that both boundaries have been successfully identified; if not, an error is raised to halt processing
        if self.highest is None or self.lowest is None:
            raise ValueError("Failing to find highest or lowest white pixel")

        # If the lowest boundary is within the valid range of the image dimensions,
        # the corresponding row is examined to identify the extreme horizontal positions (left and right) of the object
        if self.lowest < scharr_y_black.shape[0]:
            row = scharr_y_black[self.lowest, :]
            # Extracting positions in the row where the pixel intensity indicates part of the object,
            # which allows the algorithm to determine the horizontal span of the object at its base
            white_pixels = np.where(row == 255)[0]
            # If the row contains white pixels, the first and last positions in the row are taken as the lateral boundaries;
            # if not, the horizontal boundaries remain undefined
            if white_pixels.size > 0:
                self.leftmost = int(white_pixels[0])
                self.rightmost = int(white_pixels[-1])
            else:
                self.leftmost, self.rightmost = None, None
        else:
            self.leftmost, self.rightmost = None, None

        # Storing the adaptive thresholded image as the final processed output,
        # which will be used in subsequent analysis and measurement extraction routines
        self.processed = adaptive_thresh

        # Returning a dictionary that aggregates all the key measurement data extracted from the image,
        # providing a structured interface for downstream processing tasks such as droplet dimension analysis
        return {
            'highest': self.highest,
            'lowest': self.lowest,
            'leftmost': self.leftmost,
            'rightmost': self.rightmost,
            'axis_of_symmetry': self.axis_of_symmetry,
            'processed_image': self.processed
        }

# Defining a class that measures the diameter of a needle by applying edge detection and the Hough transform,
# which involves detecting lines in the image and using their intersections to compute a distance measurement
class NeedleDiameter:
    # Specifying thresholds and parameters for the Canny edge detector, which is used to identify sharp transitions in intensity,
    # ensuring that the algorithm captures the edges of the needle accurately
    CANNY_THRESHOLD1 = 50
    CANNY_THRESHOLD2 = 150
    CANNY_APERTURE_SIZE = 3

    # Setting parameters for the Hough transform, a technique that converts points in the image space into a parameter space,
    # where lines can be identified by their distance and angle, enabling robust detection of straight-line features
    HOUGH_RHO = 2
    HOUGH_THETA = np.pi / 180
    HOUGH_THRESHOLD = 100

    # Specifying tolerances to group similar line orientations and to filter out nearly vertical lines,
    # which aids in selecting the most consistent set of lines that represent the needle edges
    ANGLE_TOLERANCE = 5e-2
    COSINE_THRESHOLD = 1e-2
    MAX_LINES = 10

    def __init__(self,
                 canny_threshold1=None,
                 canny_threshold2=None,
                 aperture_size=None,
                 hough_rho=None,
                 hough_theta=None,
                 hough_threshold=None,
                 max_lines=None,
                 angle_tolerance=None,
                 cosine_threshold=None):
        # Configuring the edge detection thresholds, using either the provided values or default constants,
        # to ensure that the detector responds appropriately to the specific contrast and noise levels in the images
        self.canny_threshold1 = canny_threshold1 if canny_threshold1 is not None else NeedleDiameter.CANNY_THRESHOLD1
        self.canny_threshold2 = canny_threshold2 if canny_threshold2 is not None else NeedleDiameter.CANNY_THRESHOLD2
        self.aperture_size = aperture_size if aperture_size is not None else NeedleDiameter.CANNY_APERTURE_SIZE
        # Setting the parameters for the Hough transform to control the resolution of the parameter space and the detection sensitivity,
        # which directly affect how well the algorithm can detect the lines that form the needle’s boundaries
        self.hough_rho = hough_rho if hough_rho is not None else NeedleDiameter.HOUGH_RHO
        self.hough_theta = hough_theta if hough_theta is not None else NeedleDiameter.HOUGH_THETA
        self.hough_threshold = hough_threshold if hough_threshold is not None else NeedleDiameter.HOUGH_THRESHOLD
        # Configuring limits on the number of lines to consider and tolerances for grouping similar line angles,
        # which help in reducing computational load and improving the consistency of the diameter measurement
        self.max_lines = max_lines if max_lines is not None else NeedleDiameter.MAX_LINES
        self.angle_tolerance = angle_tolerance if angle_tolerance is not None else NeedleDiameter.ANGLE_TOLERANCE
        self.cosine_threshold = cosine_threshold if cosine_threshold is not None else NeedleDiameter.COSINE_THRESHOLD

    def measure(self, original_image, highest_y, axis_of_symmetry):
        # Converting the input image to grayscale if it is in color,
        # since edge detection algorithms typically operate on single-channel images to simplify intensity gradient computations
        if len(original_image.shape) == 3:
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = original_image.copy()
        # Applying the Canny edge detection algorithm with the configured thresholds and aperture size,
        # which extracts the edges that are likely to correspond to the physical boundaries of the needle
        edges = cv2.Canny(gray_image, self.canny_threshold1, self.canny_threshold2, apertureSize=self.aperture_size)
        # Using the Hough transform to detect lines in the edge-detected image,
        # which transforms edge points into a parameter space and identifies lines by finding peaks in the accumulator array
        lines = cv2.HoughLines(edges, self.hough_rho, self.hough_theta, threshold=self.hough_threshold)
        # If a sufficient number of lines are detected, proceed with the analysis; otherwise, return failure indicators
        if lines is not None and len(lines) >= 2:
            # Limiting the number of lines to be processed to avoid excessive computation and to focus on the most prominent ones
            lines = lines[:self.max_lines] if len(lines) > self.max_lines else lines
            # Initializing a dictionary to group detected lines by their angle, which facilitates the identification of the dominant orientation,
            # assumed to correspond to the needle’s edges
            angle_groups = {}
            # Iterating over each detected line in the parameter space,
            # where each line is represented by its distance from the origin and its angle relative to the horizontal axis
            for line in lines:
                rho, theta = line[0]
                # Setting a flag to indicate whether the current line's orientation is similar to an already identified group,
                # which is necessary to cluster lines that represent the same physical edge
                grouped = False
                # Iterating through the groups of similar angles to check if the current line fits into any existing cluster
                for key in angle_groups:
                    # Comparing the angle of the current line with the group’s representative angle,
                    # using a specified tolerance to account for minor variations due to noise and discretization
                    if np.isclose(theta, key, atol=self.angle_tolerance):
                        # Adding the line to the existing group since its orientation is sufficiently similar,
                        # which reinforces the statistical significance of that particular orientation
                        angle_groups[key].append((rho, theta))
                        grouped = True
                        break
                # If the line does not match any existing group, create a new group with this line as the initial member,
                # ensuring that all distinct orientations are considered in the analysis
                if not grouped:
                    angle_groups[theta] = [(rho, theta)]
            # If, after grouping, no valid clusters are found (an unlikely scenario), return None to signal a detection failure
            if not angle_groups:
                return None, None
            # Identifying the dominant angle group by selecting the group with the highest number of lines,
            # which is assumed to best represent the true orientation of the needle's edges in the image
            dominant_angle = max(angle_groups.keys(), key=lambda k: len(angle_groups[k]))
            # Retrieving all the lines that belong to the dominant orientation group,
            # as these are the candidates that will be used to compute the needle diameter
            dominant_lines = angle_groups[dominant_angle]
            # If the dominant group contains at least two lines, the algorithm can proceed to calculate the distance between them,
            # which is interpreted as the diameter of the needle in pixel units
            if len(dominant_lines) >= 2:
                # Preparing a copy of the original image (or converting it to a color image if necessary) for drawing the detected lines,
                # which facilitates visual verification of the line detection and the subsequent diameter measurement
                diameter_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR) if len(original_image.shape) == 2 else original_image.copy()
                # Iterating over pairs of lines within the dominant group to find a pair that straddles the symmetry axis,
                # ensuring that the measured distance is a valid representation of the needle’s width
                for i in range(len(dominant_lines)):
                    for j in range(i + 1, len(dominant_lines)):
                        rho1, theta1 = dominant_lines[i]
                        rho2, theta2 = dominant_lines[j]
                        # Skipping line pairs whose angles differ significantly, as they are unlikely to represent parallel edges of the same needle
                        if abs(theta1 - theta2) > self.angle_tolerance:
                            continue
                        # Filtering out lines that are nearly vertical by examining the cosine of their angles,
                        # since the measurement relies on sufficiently horizontal intercepts to be accurate
                        if np.abs(np.cos(theta1)) < self.cosine_threshold or np.abs(np.cos(theta2)) < self.cosine_threshold:
                            continue
                        # Calculating the x-intercepts of both lines at the vertical coordinate corresponding to the highest feature in the image,
                        # using the line equation in polar form, which transforms the polar parameters into Cartesian coordinates
                        x1 = (rho1 - highest_y * np.sin(theta1)) / np.cos(theta1)
                        x2 = (rho2 - highest_y * np.sin(theta2)) / np.cos(theta2)
                        # Checking if one intercept lies to the left and the other to the right of the vertical symmetry line,
                        # which confirms that the two lines bound the needle from opposite sides
                        if (x1 < axis_of_symmetry and x2 > axis_of_symmetry) or (x1 > axis_of_symmetry and x2 < axis_of_symmetry):
                            # Calculating the absolute distance between the two x-intercepts,
                            # which corresponds to the measured diameter of the needle in the image
                            needle_diameter_px_distance = abs(x2 - x1)
                            # For visualization, drawing the first two dominant lines over a long range,
                            # so that their orientations and positions can be inspected in relation to the needle
                            for (rho_val, theta_val) in dominant_lines[:2]:
                                a = np.cos(theta_val)
                                b = np.sin(theta_val)
                                x0 = a * rho_val
                                y0 = b * rho_val
                                # Computing two far apart points along the line to ensure that the line appears extended when drawn
                                x1_line = int(x0 + 1000 * (-b))
                                y1_line = int(y0 + 1000 * a)
                                x2_line = int(x0 - 1000 * (-b))
                                y2_line = int(y0 - 1000 * a)
                                cv2.line(diameter_image, (x1_line, y1_line), (x2_line, y2_line), (255, 0, 0), thickness=1)
                            # Drawing a thick line between the two computed intercepts on the image,
                            # which visually represents the measured diameter for confirmation and later reporting
                            pt1 = (int(x1), int(highest_y))
                            pt2 = (int(x2), int(highest_y))
                            cv2.line(diameter_image, pt1, pt2, (0, 255, 0), thickness=2)
                            # Returning the computed needle diameter along with the annotated image that shows the measurement,
                            # so that both numerical and visual outputs are available for further analysis
                            return needle_diameter_px_distance, diameter_image
                # If no pair of lines satisfying the required conditions is found, return failure indicators
                return None, None
            else:
                # If fewer than two lines exist in the dominant group, it is impossible to compute the diameter reliably,
                # so the function signals this by returning None values
                return None, None
        else:
            # If the initial edge detection did not yield enough lines, the function returns None to indicate that the measurement cannot be performed
            return None, None

# Defining a class for interpolating the contour data to reconstruct smooth curves,
# and for computing geometric properties such as curvature and circle fitting from these interpolated curves
class Interpolator:
    """
    Providing a suite of interpolation tools to reconstruct continuous curves from discrete data points,
    which is essential for accurately modeling the shape of droplets or other features and for computing derived quantities such as curvature and fitted circle parameters
    """
    @staticmethod
    def fit_hermite(x, y):
        # Generating a parameter that ranges from 0 to 1 with as many points as there are data values,
        # which is used to parameterize the curve for interpolation purposes
        t = np.linspace(0, 1, len(x))
        try:
            # Attempting to fit a Hermite spline that smoothly passes through the data points,
            # using an internal helper function that approximates derivatives for the spline construction
            spline = Interpolator._create_hermite_spline(t, x, y)
        except Exception as e:
            # If the spline fitting fails, raising an error that provides insight into what went wrong,
            # ensuring that the failure is clearly communicated for debugging purposes
            raise ValueError(f"Hermite spline fitting failed: {e}")
        # Returning the fitted Hermite spline representation for further evaluation or plotting
        return spline

    @staticmethod
    def _create_hermite_spline(t, x, y):
        # Converting the parameter and data arrays into numpy arrays to ensure compatibility with numerical operations,
        # which is necessary for efficient computation of derivatives and spline coefficients
        t = np.array(t)
        x = np.array(x)
        y = np.array(y)
        # Determining the number of data points available for interpolation
        n = len(t)
        # Validating that there are enough points to perform interpolation,
        # since at least two points are required to define a line or curve
        if n < 2:
            raise ValueError("Need at least two points for spline interpolation")
        # Initializing arrays to hold the derivatives of the data points,
        # which are critical for constructing a Hermite spline that not only passes through the points but also respects the local slopes
        dx = np.zeros(n)
        dy = np.zeros(n)
        if n > 2:
            # Estimating the derivative at the first data point using a forward difference,
            # which approximates the slope based on the change between the first two points
            dx[0] = (x[1] - x[0]) / (t[1] - t[0])
            # Estimating the derivative at the last data point using a backward difference,
            # which approximates the slope based on the change between the last two points
            dx[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2])
            dy[0] = (y[1] - y[0]) / (t[1] - t[0])
            dy[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
            # For interior points, using the central difference method to estimate the derivative,
            # which provides a balanced approximation by considering the change over an interval centered at the point of interest
            for i in range(1, n - 1):
                dt = t[i + 1] - t[i - 1]
                dx[i] = (x[i + 1] - x[i - 1]) / dt
                dy[i] = (y[i + 1] - y[i - 1]) / dt
        else:
            # In the edge case where exactly two points exist, using a simple forward difference for both points,
            # ensuring that the spline has a consistent slope even with minimal data
            dx[0] = (x[1] - x[0]) / (t[1] - t[0])
            dx[1] = dx[0]
            dy[0] = (y[1] - y[0]) / (t[1] - t[0])
            dy[1] = dy[0]
        # Creating monotonic piecewise cubic interpolators (PCHIP) for the x and y data separately,
        # which are preferred because they preserve the shape of the data and avoid oscillations between points
        spline_x = interpolate.PchipInterpolator(t, x)
        spline_y = interpolate.PchipInterpolator(t, y)
        # Returning the pair of spline functions that represent the smooth interpolated curves for the x and y coordinates
        return (spline_x, spline_y)

    @staticmethod
    def fit_pchip(x, y):
        # Generating a normalized parameter that spans from 0 to 1 with a length equal to the number of input data points,
        # which standardizes the input for the PCHIP interpolator and facilitates evaluation at arbitrary points
        t = np.linspace(0, 1, len(x))
        # Fitting a PCHIP interpolator for the horizontal component of the data,
        # which produces a smooth curve that preserves the monotonicity of the input values
        spline_x = interpolate.PchipInterpolator(t, x)
        # Fitting a similar interpolator for the vertical component of the data,
        # ensuring that the full two-dimensional shape of the object is accurately reconstructed
        spline_y = interpolate.PchipInterpolator(t, y)
        # Returning the two spline functions along with the parameter array,
        # which allows for the evaluation of the interpolated curve at any point in the [0,1] interval
        return spline_x, spline_y, t

    @staticmethod
    def compute_curvature(spline_x, spline_y, t_dense):
        # Evaluating the first derivatives of the spline functions at a dense set of parameter values,
        # which provides the instantaneous rate of change of the curve in both horizontal and vertical directions
        dx_dt = spline_x.derivative(1)(t_dense)
        dy_dt = spline_y.derivative(1)(t_dense)
        # Evaluating the second derivatives to capture the acceleration or the rate at which the slope changes,
        # which is essential for computing the curvature of the curve at each point
        d2x_dt2 = spline_x.derivative(2)(t_dense)
        d2y_dt2 = spline_y.derivative(2)(t_dense)
        # Using the standard formula for curvature of a parametric curve, which involves both the first and second derivatives,
        # and handling potential division by zero by ignoring any invalid numerical operations
        with np.errstate(divide='ignore', invalid='ignore'):
            curvature = np.abs((dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / np.power(dx_dt**2 + dy_dt**2, 1.5))
        # Returning the computed curvature values, which describe how sharply the curve bends at each point
        return curvature

    @staticmethod
    def fit_circle(points):
        # Separating the two-dimensional point data into horizontal and vertical components,
        # which are needed to set up the system of equations for circle fitting
        x = points[:, 0]
        y = points[:, 1]
        # Constructing a matrix based on the general circle equation in expanded form,
        # where each row corresponds to the transformation of a point's coordinates into the linear system
        A = np.column_stack((2*x, 2*y, np.ones(len(x))))
        # Forming a vector from the squared distances of each point from the origin,
        # which represents the right-hand side of the least squares problem for circle fitting
        B = x**2 + y**2
        # Solving the linear least squares problem to estimate the circle parameters,
        # which finds the best-fitting circle in the sense of minimizing the sum of squared residuals
        sol, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        # Extracting the estimated parameters that correspond to the circle's center coordinates and an auxiliary constant,
        # which are then used to compute the radius of the circle
        a, b, c = sol
        # Calculating the circle's radius using the relationship between the parameters,
        # which completes the circle fitting process by providing both location and size of the best-fit circle
        R = sqrt(a**2 + b**2 + c)
        # Returning the computed center coordinates and radius as a tuple,
        # enabling further use in measurements such as droplet size estimation
        return a, b, R

# Defining a class that models the droplet shape by solving the Young–Laplace differential equation,
# which governs the balance between surface tension and pressure differences in a fluid interface
class YoungLaplaceShape:
    def __init__(self, delta_rho, initial_beta=1.0, max_arclength=10.0, rtol=1e-7, atol=1e-10, max_iter=1000):
        # Storing the difference in density between the two fluids involved,
        # which is a critical physical parameter influencing the shape and stability of the droplet
        self.delta_rho = delta_rho
        # Initializing the parameter beta, which relates to the curvature of the droplet's interface,
        # and serves as a tuning parameter in the differential equation modeling the shape
        self.beta = initial_beta
        # Defining the maximum arclength over which the differential equation will be solved,
        # ensuring that the integration covers the full extent of the droplet profile
        self.max_s = max_arclength
        # Storing the relative tolerance for the numerical ODE solver to control the precision of the solution,
        # which is necessary for obtaining accurate and stable results from the integration process
        self.rtol = rtol
        # Storing the absolute tolerance for the ODE solver to further refine the solution accuracy,
        # particularly important when dealing with small numerical values in the physical model
        self.atol = atol
        # Recording the maximum number of iterations allowed for any iterative procedures,
        # which prevents infinite loops and ensures that the solver terminates within a reasonable timeframe
        self.max_iter = max_iter
        # Setting the gravitational acceleration constant, which influences the pressure distribution in the droplet,
        # and is a standard physical constant required in the computation of surface tension forces
        self.g = 9.81
        # Defining the initial conditions for the state variables in the ODE system,
        # representing the starting geometry of the droplet profile (e.g., initial radius, height, and their derivatives)
        self.y0 = [0.0, 0.0, 1.0, 0.0]
        # Initializing a placeholder to store the solution of the ODE once it has been computed,
        # so that the computed droplet profile can be accessed and analyzed later
        self.solution = None
        # Setting a threshold value for the parameter beta, above which the ODE system is considered stiff,
        # and a different numerical integration method may be required to obtain a stable solution
        self.beta_threshold = 10.0

    def phi(self, r, z, dr_ds, dz_ds):
        # Defining a small constant to avoid division by zero in subsequent calculations,
        # which ensures numerical stability during the evaluation of the differential equation
        epsilon = 1e-12
        # Computing the auxiliary function that appears in the differential equation,
        # which combines contributions from the curvature of the interface and the pressure term,
        # thereby capturing the balance of forces at the fluid interface
        return 2.0 - self.beta * z - (dz_ds + epsilon) / (r + epsilon)

    def ode_system(self, s, y):
        # Unpacking the state vector into its constituent variables:
        # a radial coordinate, a vertical coordinate, and their respective derivatives with respect to arclength,
        # which describe the current configuration of the droplet's profile
        r, z, dr_ds, dz_ds = y
        # Computing the auxiliary function using the current state variables,
        # which modulates the curvature and is central to the Young–Laplace equation formulation
        Phi = self.phi(r, z, dr_ds, dz_ds)
        # Calculating the second derivative of the radial coordinate with respect to arclength,
        # based on the balance of forces described by the ODE system
        d2r_ds2 = -dz_ds * Phi
        # Calculating the second derivative of the vertical coordinate with respect to arclength,
        # which together with the radial second derivative, fully describes the curvature of the droplet profile
        d2z_ds2 = dr_ds * Phi
        # Returning the complete set of first derivatives (including the original first derivatives and the computed second derivatives),
        # which constitutes the system of ODEs to be integrated over the droplet’s arclength
        return [dr_ds, dz_ds, d2r_ds2, d2z_ds2]

    def solve_ode(self):
        # Selecting an appropriate numerical method for integrating the ODE system based on the stiffness of the problem,
        # which is determined by comparing the current value of the curvature parameter to a predefined threshold
        if self.beta > self.beta_threshold:
            # Choosing an implicit solver suited for stiff problems to ensure stability and accuracy,
            # and simultaneously reducing the tolerances to achieve a more precise solution when the system is challenging
            method_used = 'BDF'
            current_rtol = self.rtol / 10
            current_atol = self.atol / 10
        else:
            # For non-stiff problems, using a high-order explicit solver that is efficient and accurate for smooth ODE systems,
            # while maintaining the originally specified tolerance levels for precision
            method_used = 'DOP853'
            current_rtol = self.rtol
            current_atol = self.atol
        # Invoking the numerical ODE solver over the interval from zero to the maximum arclength,
        # providing the initial conditions and the chosen solver options to compute the droplet profile
        sol = solve_ivp(self.ode_system, [0, self.max_s], self.y0,
                        method=method_used, rtol=current_rtol, atol=current_atol, max_step=0.05)
        # If the solver fails to converge to a solution, raise an error to signal that the ODE integration did not succeed,
        # which prompts further investigation or adjustment of parameters
        if not sol.success:
            raise RuntimeError("ODE solution failed")
        # Storing the successful solution for future access by other methods in the class,
        # enabling the generation and analysis of the droplet profile without recomputing the ODE
        self.solution = sol

    def generate_profile(self, beta):
        # Updating the curvature parameter to the new value provided,
        # which modifies the balance of forces in the model and thus the shape of the droplet
        self.beta = beta
        # Solving the ODE system with the updated parameter to generate a new droplet profile,
        # ensuring that the computed profile reflects the current physical conditions
        self.solve_ode()
        # Extracting the arclength values from the computed solution, which serve as the independent variable for the profile
        s_values = self.solution.t
        # Extracting the radial positions from the solution, which represent one dimension of the droplet’s geometry
        r_values = self.solution.y[0]
        # Extracting the vertical positions from the solution, which represent the other dimension of the droplet’s profile
        z_values = self.solution.y[1]
        # Returning the complete set of computed profile data (arclength, radial, and vertical coordinates),
        # which can be compared with experimental measurements to evaluate the model’s accuracy
        return s_values, r_values, z_values

    def compute_cost(self, beta, experimental_profile):
        try:
            # Generating the theoretical droplet profile for the current value of the curvature parameter,
            # which involves solving the ODE and obtaining the corresponding geometric coordinates
            s_model, r_model, z_model = self.generate_profile(beta)
        except Exception as e:
            # In case the profile generation fails, output an error message to inform about the issue,
            # and return an infinite cost to indicate that the current parameter value is unacceptable
            print(f"Error generating profile for beta={beta}: {e}")
            return np.inf
        # Checking whether the experimental data extends beyond the range of the model,
        # which would make a direct comparison impossible and require rejecting the current model configuration
        if experimental_profile[0][-1] > s_model[-1]:
            return np.inf
        # Interpolating the theoretical profile’s radial coordinates to the arclength points provided by the experimental data,
        # thereby aligning the model with the experimental measurement domain for a meaningful comparison
        r_model_interp = np.interp(experimental_profile[0], s_model, r_model)
        # Interpolating the vertical coordinates in a similar manner to ensure that both dimensions of the profile are compared consistently
        z_model_interp = np.interp(experimental_profile[0], s_model, z_model)
        # Computing the cost as the sum of the squared differences between the theoretical and experimental profiles in both dimensions,
        # which quantifies the discrepancy and serves as the objective function for the optimization process
        cost = np.sum((r_model_interp - experimental_profile[1])**2 +
                      (z_model_interp - experimental_profile[2])**2)
        # Returning the computed cost, which the optimization algorithm will attempt to minimize by adjusting the parameter
        return cost

    def optimize_beta(self, experimental_profile, initial_beta=1.0):
        # Defining an inner function that returns the cost associated with a given parameter value,
        # which is used by the optimization routine to search for the parameter that minimizes the discrepancy with experimental data
        def cost_function(beta):
            return self.compute_cost(beta, experimental_profile)
        # Using a bounded minimization routine to find the optimal parameter within a physically reasonable range,
        # ensuring that the solution is both mathematically sound and physically meaningful
        res = minimize_scalar(cost_function, bounds=(0.01, 100), method='bounded', options={'xatol': 1e-7})
        # If the optimization routine converges successfully, return the optimal parameter value,
        # which will then be used to compute the interfacial tension and other derived quantities
        if res.success:
            return res.x
        else:
            # If the optimization fails, raise an error to signal that the parameter tuning did not converge,
            # prompting further investigation into the model or the experimental data
            raise RuntimeError("Optimization failed")

    def compute_ift(self, experimental_profile, r0):
        try:
            # Attempting to find the optimal curvature parameter by minimizing the discrepancy between the model and experimental profiles,
            # which is a crucial step for accurately predicting the physical properties of the droplet
            beta_opt = self.optimize_beta(experimental_profile)
        except Exception as e:
            # If the optimization process fails, print an error message to notify of the failure and return None values,
            # indicating that the interfacial tension cannot be computed under the current conditions
            print(f"Beta optimization failed: {e}")
            return None, None
        # Calculating the interfacial tension using the physical relationship that involves the density difference, gravitational acceleration, the square of the characteristic radius, and the optimized parameter,
        # which yields the final physical quantity of interest expressed in SI units
        sigma = self.delta_rho * self.g * (r0**2) * beta_opt
        # Returning both the computed interfacial tension and the optimized curvature parameter for further analysis or reporting
        return sigma, beta_opt

# Defining a helper function that processes an image by performing edge detection, filtering out points based on spatial constraints,
# and applying interpolation to reconstruct a smooth contour that represents the droplet boundary
def interpolate_and_filter(input_image_path, highest_y, lowest_y, left_x, right_x):
    # Reading the image from disk in grayscale mode to ensure consistent processing,
    # as grayscale images simplify the task of edge detection and reduce computational complexity
    original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    # If the image cannot be loaded (e.g., due to a missing file), an error is raised to alert the user of the problem,
    # preventing further processing on invalid data
    if original_image is None:
        raise FileNotFoundError(f"Cannot load image: {input_image_path}")
    # Applying the Canny edge detector with predefined thresholds to extract the edges,
    # which represent the boundaries of objects in the image and are crucial for subsequent contour extraction
    edges = cv2.Canny(original_image, NeedleDiameter.CANNY_THRESHOLD1, NeedleDiameter.CANNY_THRESHOLD2)
    # Identifying the coordinates of all pixels in the edge-detected image that are part of an edge,
    # and organizing these coordinates into a two-dimensional array where each row represents a point
    white_pixels = np.column_stack(np.where(edges > 0))
    # If no edge pixels are detected, an error is raised to indicate that the image lacks the necessary features for interpolation,
    # which may be due to poor contrast or other imaging issues
    if white_pixels.size == 0:
        raise ValueError("No white pixels found after applying Canny filter")
    # Separating the horizontal and vertical coordinate data from the array of edge points,
    # which will be used to filter and parameterize the detected contour
    x_values = white_pixels[:, 1]
    y_values = white_pixels[:, 0]
    # Expanding the left boundary slightly by subtracting a fixed margin while ensuring it does not go below zero,
    # which helps to include a small buffer zone around the detected feature
    left_x_expanded = max(left_x - 5, 0)
    # Expanding the right boundary similarly by adding a margin while ensuring it stays within the image dimensions,
    # thereby accounting for potential measurement uncertainty at the edges
    right_x_expanded = min(right_x + 5, original_image.shape[1] - 1)
    # Creating a boolean mask that selects only those edge points whose vertical and horizontal coordinates fall within the specified boundaries,
    # which focuses the interpolation on the region most likely to contain the droplet’s contour
    mask = ((y_values >= highest_y) & (y_values <= lowest_y) &
            (x_values >= left_x_expanded) & (x_values <= right_x_expanded))
    # Filtering the horizontal coordinate data using the mask to retain only the relevant points,
    # which reduces noise and irrelevant features from the dataset
    x_values_filtered = x_values[mask]
    # Filtering the vertical coordinate data similarly to ensure that the points used for interpolation accurately represent the object’s boundary
    y_values_filtered = y_values[mask]
    # If the filtering process results in an empty set of points, an error is raised to indicate that the region of interest does not contain sufficient data,
    # which may occur if the provided boundaries are too restrictive or if the image quality is poor
    if len(x_values_filtered) == 0 or len(y_values_filtered) == 0:
        raise ValueError("No points for interpolation after applying area constraints")
    # Computing the Euclidean distance between consecutive filtered points,
    # which is used to parameterize the curve based on the physical distances along the contour
    diffs = np.sqrt(np.diff(x_values_filtered)**2 + np.diff(y_values_filtered)**2)
    # Initializing a parameter array that will represent the cumulative distance along the contour,
    # which is essential for creating a smooth mapping from the discrete points to a continuous curve
    t = np.zeros(len(x_values_filtered))
    # Setting the parameter values based on the cumulative sum of the distances, which effectively assigns a relative position along the contour to each point
    t[1:] = np.cumsum(diffs)
    # Normalizing the parameter so that it spans from 0 to 1,
    # which standardizes the input for the interpolation functions and facilitates comparisons across different images
    t = t / t[-1] if t[-1] != 0 else np.linspace(0, 1, len(x_values_filtered))
    try:
        # Creating PCHIP interpolators for both the horizontal and vertical coordinate data,
        # which ensures that the resulting interpolated curves are smooth and maintain the original shape characteristics of the data
        pchip_interp_x = interpolate.PchipInterpolator(t, x_values_filtered)
        pchip_interp_y = interpolate.PchipInterpolator(t, y_values_filtered)
    except Exception as e:
        # If the interpolation function fails, an error is raised with a detailed message,
        # enabling easier diagnosis of the problem by indicating that the interpolation step could not be completed
        raise ValueError(f"PCHIP interpolation failed: {e}")
    # Generating a dense array of parameter values between 0 and 1 to evaluate the interpolators,
    # which results in a smooth and finely-sampled representation of the contour
    t_dense = np.linspace(0, 1, 20000)
    try:
        # Evaluating the horizontal interpolator on the dense parameter array to obtain a smooth curve of x-coordinates,
        # which represents a refined version of the object's boundary along the horizontal direction
        x_dense_pchip = pchip_interp_x(t_dense)
        # Evaluating the vertical interpolator similarly to obtain the corresponding smooth curve of y-coordinates,
        # which completes the reconstruction of the continuous contour of the object
        y_dense_pchip = pchip_interp_y(t_dense)
    except Exception as e:
        # If the evaluation of the interpolators fails, an error is raised to indicate that the smooth curve could not be generated,
        # which prevents further processing and ensures that only valid data is used for measurements
        raise ValueError(f"Failed to evaluate PCHIP interpolator: {e}")
    # Determining the symmetry axis of the contour by computing the median of the horizontal coordinates of the dense interpolated points,
    # which provides a robust estimate of the center of the object without being overly influenced by outliers
    symmetry_axis_x = np.median(x_dense_pchip)
    # Adjusting the highest vertical boundary by adding a fixed offset to ensure that the region for further processing captures the upper part of the object,
    # while also ensuring that the index remains within the image dimensions
    adjusted_highest_y = highest_y + 10 if (highest_y + 10 < original_image.shape[0]) else original_image.shape[0] - 1
    # Combining the dense interpolated horizontal and vertical data into a two-dimensional array where each row represents a point on the contour,
    # which facilitates further geometric analysis such as convex hull computation and curvature estimation
    combined_interpolated_points = np.column_stack((x_dense_pchip, y_dense_pchip))
    # Filtering the combined points to retain only those that are below the adjusted highest boundary,
    # which focuses the analysis on the portion of the contour that is relevant for droplet shape determination
    combined_interpolated_points = combined_interpolated_points[combined_interpolated_points[:, 1] >= adjusted_highest_y]
    # Converting the floating-point coordinates to integers,
    # as many geometric operations (like convex hull calculation) require integer input for correct functioning
    points = combined_interpolated_points.astype(np.int32)
    # Computing the convex hull of the set of points to determine the outer boundary that encloses the contour,
    # which is useful for obtaining a simplified shape representation and for dividing the points into left and right segments
    hull = cv2.convexHull(points).squeeze()
    # If the convex hull is returned as a one-dimensional array due to a minimal number of points,
    # reshaping it to ensure that it has the proper two-dimensional format required for further processing
    if hull.ndim == 1:
        hull = np.expand_dims(hull, axis=0)
    # Identifying the subset of hull points that lie to the left of the computed symmetry axis,
    # which helps in splitting the object into two halves for independent analysis of each side
    left_indices = hull[:, 0] < symmetry_axis_x
    # Similarly, identifying the subset of hull points that lie to the right of the symmetry axis,
    # which completes the division of the object’s contour into two regions
    right_indices = hull[:, 0] >= symmetry_axis_x
    # If one side of the contour has no points, it indicates an issue with the convex hull computation or the symmetry estimation,
    # so an error is raised to notify that the shape cannot be properly divided for analysis
    if not np.any(left_indices) or not np.any(right_indices):
        raise ValueError("No points found on one side of the symmetry axis in the convex hull")
    # Extracting the horizontal and vertical coordinates for the left side of the hull,
    # which will be used to generate a smooth curve representing the left boundary of the object
    x_left, y_left = hull[left_indices, 0], hull[left_indices, 1]
    # Extracting the corresponding coordinates for the right side,
    # so that both sides of the object can be analyzed separately and compared
    x_right, y_right = hull[right_indices, 0], hull[right_indices, 1]
    # Sorting the left side points in order of increasing vertical coordinate,
    # which ensures that the interpolated curve will be generated in a consistent direction (from top to bottom)
    sorted_indices_left = np.argsort(y_left)
    x_left_sorted = x_left[sorted_indices_left]
    y_left_sorted = y_left[sorted_indices_left]
    # Sorting the right side points similarly to prepare them for independent interpolation,
    # which is necessary to accurately capture the contour on both sides of the symmetry axis
    sorted_indices_right = np.argsort(y_right)
    x_right_sorted = x_right[sorted_indices_right]
    y_right_sorted = y_right[sorted_indices_right]
    # If the ordering of the left side points is reversed (i.e., descending instead of ascending),
    # reversing the arrays to ensure that the sequence progresses from top to bottom, which is standard for interpolation
    if y_left_sorted[0] > y_left_sorted[-1]:
        x_left_sorted = x_left_sorted[::-1]
        y_left_sorted = y_left_sorted[::-1]
    # Repeating the reordering for the right side to maintain consistency in the representation of the contour
    if y_right_sorted[0] > y_right_sorted[-1]:
        x_right_sorted = x_right_sorted[::-1]
        y_right_sorted = y_right_sorted[::-1]
    # Determining the lowest vertical coordinate from both sides, which is used as a reference point for merging the left and right contours
    y_lowest = max(y_left_sorted.max(), y_right_sorted.max())
    # Defining a common horizontal coordinate based on the computed symmetry axis,
    # which will be used to ensure that the left and right curves join smoothly at the bottom of the object
    x_common = symmetry_axis_x
    # If the endpoints of the left side curve do not match the common reference point,
    # appending the reference point to both the left and right curves to enforce a smooth connection
    if (x_left_sorted[-1] != x_common) or (y_left_sorted[-1] != y_lowest):
        x_left_sorted = np.append(x_left_sorted, x_common)
        y_left_sorted = np.append(y_left_sorted, y_lowest)
        x_right_sorted = np.append(x_right_sorted, x_common)
        y_right_sorted = np.append(y_right_sorted, y_lowest)
    # Fitting a PCHIP interpolator to the left side sorted data to obtain a smooth curve representation of the left boundary,
    # which is essential for accurate curvature and shape analysis of the droplet
    spline_x, spline_y, t_left = Interpolator.fit_pchip(x_left_sorted, y_left_sorted)
    # Fitting a similar interpolator to the right side data to reconstruct the right boundary curve,
    # ensuring that both sides of the object are represented by continuous and differentiable functions
    spline_x_r, spline_y_r, t_right = Interpolator.fit_pchip(x_right_sorted, y_right_sorted)
    # Computing the curvature of the left boundary by evaluating the derivatives of the interpolated curve,
    # which provides information about the bending of the object and is related to physical properties like surface tension
    curvature_left = Interpolator.compute_curvature(spline_x, spline_y, t_left)
    # Similarly, computing the curvature for the right boundary to complete the geometric analysis,
    # allowing for the estimation of an average radius of curvature for the entire object
    curvature_right = Interpolator.compute_curvature(spline_x_r, spline_y_r, t_right)
    # Inverting the mean curvature values to obtain an estimate of the radius of curvature for each side,
    # while suppressing any numerical warnings that may arise from division by very small numbers
    with np.errstate(divide='ignore', invalid='ignore'):
        R_left = 1 / np.nanmean(curvature_left) if np.nanmean(curvature_left) != 0 else None
        R_right = 1 / np.nanmean(curvature_right) if np.nanmean(curvature_right) != 0 else None
    # Converting the original grayscale image to a color image so that colored annotations can be overlaid,
    # which aids in the visual verification of the interpolation, curvature, and boundary detection steps
    original_image_colored = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    # Drawing the left boundary curve on the colored image by connecting consecutive points with a colored line,
    # which visually represents the reconstructed contour of the object’s left side
    for i in range(len(x_left_sorted) - 1):
        pt1 = (int(x_left_sorted[i]), int(y_left_sorted[i]))
        pt2 = (int(x_left_sorted[i + 1]), int(y_left_sorted[i + 1]))
        cv2.line(original_image_colored, pt1, pt2, (255, 0, 0), 2)
    # Drawing the right boundary curve similarly to visually represent the object's right side,
    # which completes the annotation of the droplet's contour on the image for verification purposes
    for i in range(len(x_right_sorted) - 1):
        pt1 = (int(x_right_sorted[i]), int(y_right_sorted[i]))
        pt2 = (int(x_right_sorted[i + 1]), int(y_right_sorted[i + 1]))
        cv2.line(original_image_colored, pt1, pt2, (0, 255, 0), 2)
    # Sampling a subset of points from the combined dense interpolation to reduce clutter in the visualization,
    # ensuring that only a manageable number of points are plotted while still representing the overall contour accurately
    sampled_indices = np.linspace(0, len(combined_interpolated_points) - 1, min(1000, len(combined_interpolated_points)), dtype=int)
    # Iterating over the sampled points to draw small markers on the colored image,
    # which provides a visual cue of the density and distribution of the interpolated contour points
    for point in combined_interpolated_points[sampled_indices]:
        x_p, y_p = int(point[0]), int(point[1])
        # Ensuring that the marker is drawn only if the point lies within the bounds of the image,
        # thereby avoiding drawing errors due to invalid coordinates
        if 0 <= x_p < original_image_colored.shape[1] and 0 <= y_p < original_image_colored.shape[0]:
            cv2.circle(original_image_colored, (x_p, y_p), 1, (0, 255, 255), -1)
    # Creating a boolean mask to select points on the left side of the symmetry axis from the dense interpolation,
    # which is necessary to identify the apex (the highest point) on that side of the droplet
    left_mask_pchip = x_dense_pchip < symmetry_axis_x
    if np.any(left_mask_pchip):
        # Extracting the vertical coordinates of points on the left side and finding the one with the maximum value,
        # which corresponds to the apex (highest point) of the left boundary
        y_left_pchip = y_dense_pchip[left_mask_pchip]
        idx_left_pchip = np.argmax(y_left_pchip)
        left_apex = (x_dense_pchip[left_mask_pchip][idx_left_pchip], y_left_pchip[idx_left_pchip])
    else:
        left_apex = (None, None)
    # Similarly, creating a mask for the right side of the symmetry axis to locate the highest point on that side,
    # which is used to determine the overall apex of the droplet by comparing both sides
    right_mask_pchip = x_dense_pchip > symmetry_axis_x
    if np.any(right_mask_pchip):
        y_right_pchip = y_dense_pchip[right_mask_pchip]
        idx_right_pchip = np.argmax(y_right_pchip)
        right_apex = (x_dense_pchip[right_mask_pchip][idx_right_pchip], y_right_pchip[idx_right_pchip])
    else:
        right_apex = (None, None)
    # Determining the overall apex of the droplet by comparing the highest points from both the left and right boundaries,
    # and selecting the one with the greater vertical coordinate as the apex, since it represents the topmost point of the droplet
    if left_apex[1] is not None and right_apex[1] is not None:
        apex = left_apex if left_apex[1] >= right_apex[1] else right_apex
    elif left_apex[1] is not None:
        apex = left_apex
    elif right_apex[1] is not None:
        apex = right_apex
    else:
        apex = (None, None)
    # If a valid apex is found, marking it on the colored image with a distinct symbol and label,
    # which facilitates visual confirmation of the detected droplet’s highest point
    if apex[0] is not None and apex[1] is not None:
        cv2.circle(original_image_colored, (int(apex[0]), int(apex[1])), 6, (0, 0, 255), -1)
        cv2.putText(original_image_colored, "Apex", (int(apex[0]) + 10, int(apex[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # Returning a comprehensive set of outputs including the annotated image, estimated radii of curvature,
    # the spline functions for both boundaries, the symmetry axis, apex coordinates, dense parameterization,
    # and computed curvature arrays, all of which will be used for further physical analysis and visualization
    return (original_image_colored, R_left, R_right, spline_x, spline_y,
            spline_x_r, spline_y_r, symmetry_axis_x, apex[0], apex[1],
            t_dense, x_dense_pchip, y_dense_pchip, curvature_left, curvature_right)

# Defining a function that leverages B-spline interpolation to further process an image and perform circle fitting on a region around the droplet apex,
# which is used to estimate a characteristic radius that is then converted into a physical length scale using a calibration factor
def interpolate_and_filter_bspline(input_image_path, highest_y, lowest_y, left_x, right_x, scale):
    try:
        # Calling the previously defined interpolation function to obtain a detailed reconstruction of the droplet contour,
        # which returns various measurement data and an annotated image that are essential for further analysis
        (interpolated_image_colored, R_left, R_right, spline_x, spline_y,
         spline_x_r, spline_y_r, symmetry_axis_x, apex_x, apex_y,
         u_dense, x_dense_pchip, y_dense_pchip, curvature_left, curvature_right) = interpolate_and_filter(
            input_image_path, highest_y, lowest_y, left_x, right_x)
    except Exception as e:
        # If the interpolation process fails (due to image quality issues or parameter mismatches), printing an error message to notify the user,
        # and returning a set of None values to indicate that the subsequent steps cannot be performed
        print(f"Interpolation and filtering failed for {input_image_path}: {e}")
        return None, None, None, None, None, None, None, None, None, None, None
    # Defining a fraction of the total parameter range to determine a local window around the apex for circle fitting,
    # which ensures that the circle is fitted only to the region where the curvature is most representative of the droplet's characteristic size
    window_fraction = 0.05
    # Locating the index corresponding to the apex (highest vertical position) in the dense parameterized data,
    # which serves as the center point of the window for circle fitting
    apex_idx = np.argmax(y_dense_pchip)
    # Calculating the window size in terms of the number of data points, based on the specified fraction,
    # so that the circle fitting is performed on a subset of points that capture the essential curvature information
    window_size = int(window_fraction * len(u_dense))
    # Determining the lower bound of the window, ensuring that it does not go below the start of the data array,
    # which is important to avoid index errors during the selection of points for circle fitting
    low_idx = max(0, apex_idx - window_size)
    # Determining the upper bound of the window, ensuring that it does not exceed the length of the data,
    # so that the subset of points remains valid and representative of the local region around the apex
    high_idx = min(len(u_dense), apex_idx + window_size)
    # Extracting the subset of interpolated points within the determined window,
    # which are then used as input for fitting a circle that approximates the local curvature of the droplet
    pts_window = np.column_stack((x_dense_pchip[low_idx:high_idx], y_dense_pchip[low_idx:high_idx]))
    try:
        # Fitting a circle to the points in the local window using a least squares approach,
        # which estimates the center and radius of the circle that best approximates the local droplet curvature
        a_center, b_center, r0_pixels = Interpolator.fit_circle(pts_window)
    except Exception as e:
        # If circle fitting fails, printing an error message to indicate the problem and setting the radius to None,
        # which ensures that the failure is recorded and that subsequent computations that depend on the circle fit do not proceed erroneously
        print(f"Circle fitting failed: {e}")
        r0_pixels = None
    # Returning a comprehensive tuple containing the annotated image, estimated curvature radii from both sides,
    # the dense parameter values, the evaluated spline functions for both boundaries, the apex coordinates,
    # and the fitted circle radius (in pixel units), which together provide the necessary data for physical interpretation
    return (interpolated_image_colored, R_left, R_right, u_dense,
            spline_x(u_dense), spline_y(u_dense),
            spline_x_r(u_dense), spline_y_r(u_dense),
            apex_x, apex_y, r0_pixels)

# Main execution block: if this script is executed as the main program, process all images in the specified directory,
# perform the necessary image analyses, compute physical measurements, and save the results to organized folders and a CSV file
if __name__ == '__main__':
    # Specifying the directory where input images are located; these images are expected to contain the droplets or features to be analyzed
    input_directory = r"input"
    # Specifying the directory where output files, such as processed images and results CSV, will be saved,
    # ensuring that results are organized and easily accessible
    output_directory = r"output"

    def process_images(input_directory, output_directory):
        # Gathering a list of file paths for images that have supported file extensions,
        # which filters the directory contents to include only relevant image files for analysis
        input_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory)
                       if f.lower().endswith(('.png', '.jpg', '.bmp'))]
        # If no images are found in the input directory, printing a message to inform the user and terminating the processing,
        # as there is no data to analyze
        if not input_files:
            print("No input files found in the specified directory.")
            return
        # Defining physical properties for the fluids involved: the densities of water and oil,
        # which are used to compute the density difference that drives the droplet formation
        rho_water = 999
        rho_oil = 773
        # Calculating the density difference, a key parameter in the Young–Laplace equation,
        # which influences the droplet shape and is used in subsequent physical computations
        delta_rho = rho_water - rho_oil
        # Creating a folder based on the current date within the output directory to store results,
        # which organizes the outputs chronologically for easy retrieval and comparison
        date_folder = FolderManager.create_date_folder(output_directory)
        # Obtaining a timestamp (current hour, minute, second) to create a subfolder for the current measurement session,
        # ensuring that data from different sessions are kept separate
        current_hour = datetime.now().strftime('%H-%M-%S')
        measurement_folder = FolderManager.create_measurement_folder(date_folder, current_hour)
        # Initializing an empty list to collect rows of data that will be written to a CSV file at the end of processing,
        # which provides a structured summary of the analysis results for each image
        csv_data = []
        # Instantiating the classifier object that will be used to detect features such as the needle and droplet in each image,
        # which encapsulates the contour analysis and classification logic
        classifier = DetectionClassifier()
        # Instantiating the needle diameter measurement object,
        # which encapsulates the logic for detecting edges and measuring the physical width of the needle in the images
        needle_diameter_calculator = NeedleDiameter()
        # Iterating over each image file, with an index to allow periodic updates and folder changes,
        # ensuring that processing remains organized over large datasets
        for i, input_file in enumerate(input_files):
            # Every 100 images, updating the measurement folder based on the current time to avoid overcrowding a single folder,
            # which aids in keeping the output files well-organized and timestamped
            if i > 0 and i % 100 == 0:
                current_hour = datetime.now().strftime('%H-%M-%S')
                measurement_folder = FolderManager.create_measurement_folder(date_folder, current_hour)
            # Reading the current image from file using OpenCV,
            # which loads the image into memory for processing; if the image cannot be loaded, a warning is printed and the file is skipped
            image_original = cv2.imread(input_file)
            if image_original is None:
                print(f"Warning: Cannot read image {input_file}. Skipping.")
                continue
            # Extracting the dimensions (height and width) of the image,
            # which are used to determine scaling factors and to resize the image for more efficient processing
            original_height, original_width = image_original.shape[:2]
            # Reducing the image size by a factor of 3 in each dimension,
            # which speeds up processing while retaining sufficient detail for detection and measurement tasks
            new_size = (original_width // 3, original_height // 3)
            image_resized = cv2.resize(image_original, new_size, interpolation=cv2.INTER_AREA)
            # Converting the resized image to grayscale,
            # since many image processing algorithms operate on single-channel intensity data, simplifying the analysis
            gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            # Applying Gaussian blur with a 5x5 kernel to the grayscale image to reduce noise and smooth variations,
            # which improves the performance of thresholding and contour detection in subsequent steps
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Thresholding the blurred image using an inverted binary threshold to highlight the droplet or needle,
            # which results in a binary image where features of interest appear as white regions on a dark background
            _, binary = cv2.threshold(blurred, 35, 255, cv2.THRESH_BINARY_INV)
            # Extracting contours from the binary image using an external retrieval method and a simplified chain approximation,
            # which captures the boundaries of significant white regions corresponding to physical objects in the image
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Using the classifier to analyze the extracted contours and determine the type of feature present in the image,
            # which returns a classification string, an annotated image with drawn contours, and the lowest vertical coordinate among the detected features
            classification, image_with_contours, lowest_y = classifier.classify(contours, image_resized)
            # Handling the case where no needle is detected by annotating the image, saving it, and recording the result in the CSV data,
            # which prevents further processing on an image that does not meet the necessary criteria
            if classification == 'Needle not detected':
                cv2.putText(image_with_contours, 'Needle not detected', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.imwrite(os.path.join(measurement_folder, f'{os.path.basename(input_file)}_needle_not_detected.png'), image_with_contours)
                csv_data.append([os.path.basename(input_file), 'Needle not detected', f"{delta_rho:.2f} kg/m^3", '-', '-', '-', '-', '-'])
                continue
            # Handling the case where a stream or gel is detected instead of a droplet by annotating, saving, and recording the outcome,
            # which ensures that the analysis distinguishes between different physical phenomena
            if classification == 'Stream/Gel':
                cv2.putText(image_with_contours, 'Stream/Gel', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                csv_data.append([os.path.basename(input_file), 'Stream/Gel', f"{delta_rho:.2f} kg/m^3", '-', '-', '-', '-', '-'])
                cv2.imwrite(os.path.join(measurement_folder, f'{os.path.basename(input_file)}_detection.png'), image_with_contours)
                continue
            # Handling the case where neither a droplet nor a stream is detected by annotating, saving, and recording the result,
            # which prevents further processing on images that do not contain relevant features
            elif classification == 'Droplet or stream not detected':
                cv2.putText(image_with_contours, 'Droplet or stream not detected', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.imwrite(os.path.join(measurement_folder, f'{os.path.basename(input_file)}_detection.png'), image_with_contours)
                csv_data.append([os.path.basename(input_file), 'Droplet or stream not detected', f"{delta_rho:.2f} kg/m^3", '-', '-', '-', '-', '-'])
                continue
            # If a droplet is detected, annotating the image accordingly and saving the detection result,
            # which provides visual confirmation of the successful identification of the droplet in the image
            elif classification == 'Droplet Detected':
                cv2.putText(image_with_contours, 'Droplet Detected', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.imwrite(os.path.join(measurement_folder, f'{os.path.basename(input_file)}_detection.png'), image_with_contours)
            try:
                # Reading the full-resolution original image again for high-accuracy processing,
                # as certain measurements such as needle diameter require the full detail of the unscaled image
                image_original_full = cv2.imread(input_file)
                if image_original_full is None:
                    print(f"Warning: Cannot read image {input_file} for additional processing. Skipping.")
                    csv_data.append([os.path.basename(input_file), 'Issue loading image for additional processing', f"{delta_rho:.2f} kg/m^3", '-', '-', '-', '-', '-'])
                    continue
                # Applying the maximum denoising procedure to the full-resolution image to enhance feature clarity,
                # which is critical for accurate edge and curvature detection in subsequent processing steps
                denoised_image = ImagePreprocessor.apply_maximum_denoising(image_original)
                # Computing the Scharr gradients on the denoised image to extract fine edge details,
                # which are used to identify the precise boundaries of the droplet for measurement
                scharr_x, scharr_y = ImagePreprocessor.apply_scharr(denoised_image)
                # Thresholding the vertical gradient image to obtain a binary image that highlights significant edge features,
                # thereby focusing the analysis on regions with strong intensity changes corresponding to the droplet boundary
                scharr_y_black = ImagePreprocessor.threshold_non_white_to_black(scharr_y)
                # Processing the denoised image to extract key measurement points within a specific vertical band,
                # which yields several parameters such as the adjusted highest and lowest boundaries, symmetry axis, and a binary image for further analysis
                result_band = ImagePreprocessor.process_image_with_marking_in_band(denoised_image)
                if result_band and len(result_band) >= 9 and result_band[5] is not None and result_band[7] is not None:
                    scharr_x_black = result_band[0]
                    adjusted_highest_y = result_band[1]
                    lowest_y = result_band[2]
                    leftmost_pixel_band = result_band[3]
                    rightmost_pixel_band = result_band[4]
                    highest_y = result_band[5]
                    best_row_y = result_band[6]
                    axis_of_symmetry = result_band[7]
                    adaptive_thresh = result_band[8]
                    # Defining a lower boundary for masking based on the highest detected feature,
                    # ensuring that regions far from the droplet are excluded from further processing
                    end_y = highest_y - 50 if highest_y - 50 >= 0 else 0
                    buffer = 15
                    # Determining the starting point for applying a mask to the thresholded image,
                    # which helps in isolating the region containing the droplet by excluding irrelevant areas
                    mask_start_y = highest_y + buffer if highest_y + buffer < scharr_y_black.shape[0] else scharr_y_black.shape[0] - 1
                    adaptive_thresh_masked = adaptive_thresh.copy()
                    # Applying the mask by zeroing out regions above and below specified boundaries,
                    # which isolates the droplet for further analysis such as contour extraction and shape measurement
                    adaptive_thresh_masked[mask_start_y:, :] = 0
                    adaptive_thresh_masked[:end_y, :] = 0
                    # Attempting to find the largest contiguous white shape in the masked binary image,
                    # which is assumed to correspond to the droplet; if no such shape is found, processing for this image is aborted
                    largest_shape_mask, largest_contour = ImagePreprocessor.find_largest_white_shape(adaptive_thresh_masked, highest_y)
                else:
                    axis_of_symmetry = None
                    highest_y = None
                    lowest_y = None
                    left_x = None
                    right_x = None
                    largest_shape_mask = None
                needle_diameter_px_distance, diameter_image = None, None
                # If the full-resolution image, along with the required geometric boundaries, are available,
                # measure the needle diameter using the previously instantiated measurement object to compute the distance between parallel lines
                if image_original_full is not None and highest_y is not None and axis_of_symmetry is not None:
                    needle_diameter_px_distance, diameter_image = needle_diameter_calculator.measure(image_original_full, highest_y, axis_of_symmetry)
                # If the largest shape could not be isolated, skip further processing for the image and log the issue
                if largest_shape_mask is None:
                    print(f"Warning: Largest shape not found in image {input_file}. Skipping additional processing.")
                    csv_data.append([os.path.basename(input_file), 'Largest shape not found', f"{delta_rho:.2f} kg/m^3", '-', '-', '-', '-', '-'])
                    continue
                # If the vertical boundaries are available, compute the droplet's height in pixels
                if result_band and result_band[2] is not None and result_band[5] is not None:
                    drop_height_px = abs(result_band[2] - result_band[5])
                else:
                    drop_height_px = None
                # If the horizontal boundaries are available, compute the horizontal extent (de_px) of the droplet in pixels
                if result_band and result_band[3] is not None and result_band[4] is not None:
                    left_x = result_band[3][1]
                    right_x = result_band[4][1]
                    de_px = abs(right_x - left_x)
                else:
                    left_x = None
                    right_x = None
                    de_px = None
                # Using the measured needle diameter in pixels and a known physical needle diameter (from calibration) to compute a scale factor,
                # which converts pixel measurements into real-world units (meters)
                if needle_diameter_px_distance is not None and needle_diameter_px_distance != 0:
                    needle_diameter_mm = 0.312
                    scale = (needle_diameter_mm / needle_diameter_px_distance) * 1e-3
                else:
                    scale = None
                # Converting the droplet height from pixels to meters using the computed scale factor,
                # which provides a physically meaningful measurement of the droplet size
                if drop_height_px is not None and scale is not None:
                    drop_height_m = drop_height_px * scale
                else:
                    drop_height_m = None
                beta = None
                # If both the vertical and horizontal extents of the droplet are available,
                # proceed to further analyze the droplet shape; otherwise, skip the detailed computation of interfacial tension
                if drop_height_px is not None and de_px is not None:
                    # Checking if there is a disproportion between the vertical and horizontal dimensions,
                    # which may indicate an irregular shape and preclude reliable physical measurements
                    if drop_height_px > 2 * de_px or de_px > 2 * drop_height_px:
                        ift_result = 'g/s'
                        R1_mean_m = '-'
                        R2_mean_m = '-'
                        r0 = None
                    else:
                        try:
                            # If all necessary boundaries and scale factors are valid, perform a B-spline based interpolation and circle fitting,
                            # which refines the contour and extracts a characteristic radius for the droplet
                            (interpolated_image, R_left, R_right, u_dense, x_dense_left, y_dense_left,
                             x_dense_right, y_dense_right, apex_x, apex_y, r0_pixels) = interpolate_and_filter_bspline(input_file, highest_y, lowest_y, left_x, right_x, scale)
                            if interpolated_image is None:
                                raise ValueError("Interpolated image is None.")
                            # Converting the circle's radius from pixels to meters using the computed scale factor,
                            # which yields a physically meaningful characteristic dimension of the droplet
                            r0 = r0_pixels * scale if r0_pixels is not None else None
                            # Assembling an experimental profile consisting of the parameterized contour and the left boundary coordinates,
                            # which will be used to optimize the physical model and compute interfacial tension
                            experimental_profile = (u_dense, x_dense_left, y_dense_left)
                            # Instantiating the Young–Laplace shape model using the computed density difference,
                            # which encapsulates the physics governing the droplet shape
                            yls = YoungLaplaceShape(delta_rho=delta_rho)
                            # If the characteristic radius is successfully computed, use the model to compute the interfacial tension and optimize beta,
                            # which provides a quantitative measure of the surface forces acting on the droplet
                            if r0 is not None:
                                sigma, beta_val = yls.compute_ift(experimental_profile, r0)
                            else:
                                sigma, beta_val = None, None
                            beta = beta_val
                            # If the interfacial tension is computed successfully, convert it to mN/m (milliNewtons per meter)
                            # and record the mean radii from both the left and right curvature analyses in physical units
                            if sigma is not None:
                                ift_value_mn_m = sigma * 1e2
                                ift_result = f"{ift_value_mn_m:.6f} mN/m"
                                R1_mean_m = R_left * scale if R_left is not None else 'Error'
                                R2_mean_m = R_right * scale if R_right is not None else 'Error'
                            else:
                                ift_result = 'Calculation error'
                                R1_mean_m = 'Error'
                                R2_mean_m = 'Error'
                        except Exception as e:
                            print(f"Interpolation and filtering failed for {input_file}: {e}")
                            interpolated_image = None
                            R1_mean_m = None
                            R2_mean_m = None
                            ift_result = 'Computation error'
                            r0 = None
                else:
                    ift_result = 'g/k'
                    R1_mean_m = '-'
                    R2_mean_m = '-'
                    r0 = None
                # Formatting the beta parameter for recording, or using a placeholder if it is not available
                beta_str = f"{beta:.6f}" if beta is not None else '-'
                # Appending a row of data to the CSV data list with relevant measurements and computed values,
                # which will later be written to a CSV file for record-keeping and further analysis
                csv_data.append([
                    os.path.basename(input_file),
                    ift_result,
                    f"{delta_rho:.2f} kg/m^3",
                    f"{R1_mean_m:.6f}" if isinstance(R1_mean_m, float) else R1_mean_m,
                    f"{R2_mean_m:.6f}" if isinstance(R2_mean_m, float) else R2_mean_m,
                    beta_str,
                    f"{apex_x:.2f}" if apex_x is not None else '-',
                    f"{apex_y:.2f}" if apex_y is not None else '-',
                    f"{r0:.6f}" if r0 is not None else '-'
                ])
                # Saving the interpolated image if it was successfully generated,
                # which provides a visual record of the contour refinement process
                if interpolated_image is not None:
                    try:
                        cv2.imwrite(os.path.join(measurement_folder, f'{os.path.basename(input_file)}_interpolated.png'), interpolated_image)
                    except Exception as e:
                        print(f"Saving interpolated image failed for {input_file}: {e}")
                else:
                    print(f"Interpolated image is empty for {input_file}. Skipping save.")
                # Creating a figure with two subplots using matplotlib to visualize key aspects of the analysis,
                # such as the marked droplet dimensions and the needle diameter measurement
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                # If the thresholded horizontal gradient image is available, display it with overlaid measurement points,
                # which provides insight into the spatial distribution of the droplet dimensions
                if 'scharr_x_black' in locals() and scharr_x_black is not None:
                    ax1.imshow(scharr_x_black, cmap='gray')
                    ax1.set_title("Droplet Dimension Points")
                    if leftmost_pixel_band is not None and rightmost_pixel_band is not None:
                        ax1.scatter(leftmost_pixel_band[1], leftmost_pixel_band[0], color='teal', s=20, marker='o', label='Leftmost')
                        ax1.scatter(rightmost_pixel_band[1], rightmost_pixel_band[0], color='cyan', s=20, marker='o', label='Rightmost')
                    if axis_of_symmetry is not None and highest_y is not None and lowest_y is not None:
                        highest_point = (int(axis_of_symmetry), int(highest_y))
                        lowest_point = (int(axis_of_symmetry), int(lowest_y))
                        ax1.scatter(highest_point[0], highest_point[1], color='pink', s=20, marker='^', label='Highest')
                        ax1.scatter(lowest_point[0], lowest_point[1], color='magenta', s=20, marker='v', label='Lowest')
                        ax1.axvline(x=axis_of_symmetry, color='gray', linestyle='--', label='Symmetry Axis')
                    ax1.legend()
                else:
                    ax1.set_title("Marked points for measurements in the band - Processing Error")
                # Displaying the image with the needle diameter drawn on it in the second subplot,
                # which allows for direct visual verification of the diameter measurement
                if diameter_image is not None:
                    ax2.imshow(cv2.cvtColor(diameter_image, cv2.COLOR_BGR2RGB))
                    ax2.set_title(f"Needle Diameter: {needle_diameter_px_distance:.2f} px" if needle_diameter_px_distance is not None else "Needle Diameter Visualization - Unavailable")
                else:
                    ax2.set_title("Needle Diameter Visualization - Unavailable")
                # Saving the figure as an image file in the measurement folder to preserve a record of the analysis visuals,
                # which can be useful for documentation, troubleshooting, or presentation purposes
                plt.savefig(os.path.join(measurement_folder, f'{os.path.basename(input_file)}_result.png'), bbox_inches='tight')
                # Closing the figure to free system resources and prevent memory overload during batch processing
                plt.close(fig)
            except Exception as e:
                # If any exception occurs during the additional processing steps (e.g., denoising, interpolation, or measurement),
                # print an error message and record a failure entry in the CSV data, then continue with the next image
                print(f"Additional processing failed for {input_file}: {e}")
                csv_data.append([os.path.basename(input_file), 'Additional processing error', f"{delta_rho:.2f} kg/m^3", '-', '-', '-', '-', '-'])
                continue
        # After processing all images, constructing the path for the CSV file where results will be stored
        csv_file_path = os.path.join(measurement_folder, 'results.csv')
        # Opening the CSV file for writing in UTF-8 encoding to ensure proper character handling,
        # and writing the header row followed by all collected data rows to provide a complete summary of the analysis
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Input File', 'IFT', 'Delta Rho', 'R1 Mean [m]', 'R2 Mean [m]', 'Beta', 'Apex X [px]', 'Apex Y [px]', 'r0 [m]'])
            writer.writerows(csv_data)
        # Printing a final message to indicate that the analysis has been completed and the results have been successfully saved,
        # providing closure to the processing routine and notifying the user of successful execution
        print('Analysis completed and results saved.')
    # Calling the function to process all images with the specified input and output directories,
    # thereby initiating the full batch analysis workflow described above
    process_images(input_directory, output_directory)
