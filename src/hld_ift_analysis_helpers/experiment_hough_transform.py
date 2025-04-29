#python "D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/scripts/experiment_hough_transform.py"
#
import sys
sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/scripts")
#import pandas as pd
import re
import os
import time
import math

#import copy
import numpy as np

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data
from skimage.io import imread,imsave
from skimage.color import rgb2gray

from skimage.filters import scharr

import matplotlib.pyplot as plt
from matplotlib import cm


from collect_files_folders import collect_files
from droplet_stats import needle_image, region_below_needle

from skimage.morphology import skeletonize as skeletonize_skim
from skimage.util import invert

start = time.time()

def needle_only(im, height = 0):
    return im[:200, :]

def needle_edges(im, px = 2, max_theta = 90, angles = 360, transpose = False):
    image = scharr(needle_only(im))
    x = np.quantile(image, 1 - px / image.shape[1] )
    image[image < x] = 0.0
    image[image > 0] = 1.0
    image = image.transpose() if transpose else image

    theta_val = np.pi * max_theta / 2 / 90
    tested_angles = np.linspace(-theta_val, theta_val, angles, endpoint=False)
    h, theta, d = hough_line(image, theta=tested_angles)

    #for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    #    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    #    ax[2].axline((x0, y0), slope=np.tan(angle + np.pi / 2))
    #    print((x0,y0))
    #    print(dist)
    #    print(np.cos(angle))
    #    print(np.sin(angle))
    #    print(np.tan(angle + np.pi / 2))
    return zip(*hough_line_peaks(h, theta, d))

#>def plot_image(img1, lbl1, img2, lbl2):
#>    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
#>    ax = axes.ravel()
#>    
#>    ax[0].imshow(img1, cmap=cm.gray)
#>    ax[0].set_title(lbl1)
#>    
#>    ax[1].imshow(img2, cmap=cm.gray)
#>    ax[1].set_title(lbl2)
#>    
#>    for a in ax:
#>        a.set_axis_off()
#>    
#>    plt.tight_layout()
#>    plt.show()


    
root = "D:/temp_data/exp_2025-02-24_001_20g_BrijL4_C7-C16"
root_out = "D:/temp_data/output/exp_2025-02-24_001_20g_BrijL4_C7-C16"

root = "D:/temp_data/exp_2025-03-26_06g_BrijL4_C7C16_001"
root_out = "D:/temp_data/output/exp_2025-03-26_06g_BrijL4_C7C16_001"

jpgs = collect_files(root, "[0-9]{5}.jpg$") #"conc_[0-9.]{7}[\/].*[0-9]{5}.jpg$")


for i, im in enumerate(jpgs):
    if i < 5:
        print(f'{i: >5d}: {im}')




print(f'opening file:\n   {jpgs[0]}')
image_2 = needle_image(imread(jpgs[0], as_gray = True), 100)
image_org = np.copy(image_2)

# plot_image(image_org, "", image_2, "")

print("============================")

def print_out(out):
    for _, angle, dist in out:
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        #ax[2].axline((x0, y0), slope=np.tan(angle + np.pi / 2))
        print(".")
        print(f'point: {(x0,y0)}')
        print(f'distance: {dist}')
        print(f'x_proj: {np.cos(angle)}')
        print(f'y_proj: {np.sin(angle)}')
        print(f'angle: {np.tan(angle + np.pi / 2)}')

print("============================ no transpose")
out = needle_edges(image_2, px = 2)
print_out(out)

print("============================ with transpose, 360")
out = needle_edges(image_2, px = 2, transpose = True)
print_out(out)

print("============================ with transpose 5,20")
out = needle_edges(image_2, px = 2, max_theta = 5, angles = 20, transpose = True)
print_out(out)

print("============================")
#dims = image.shape
#print(f'shape: {dims}')
#x = np.quantile(image, 1 - 2 / dims[1] )
#print(f'fraction: {x: 0.5f}')
#
#image[image < x] = 0.0
#image[image > 0] = 1.0


#--example_1->|#============================================================
#--example_1->|# Constructing test image
#--example_1->|image = np.zeros((200, 200))
#--example_1->|idx = np.arange(25, 175)
#--example_1->|image[idx, idx] = 255
#--example_1->|image[draw_line(45, 25, 25, 175)] = 255
#--example_1->|image[draw_line(25, 135, 175, 155)] = 255
#--example_1->|
#--example_1->|# Classic straight-line Hough transform
#--example_1->|# Set a precision of 0.5 degree.
#--example_1->|tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
#--example_1->|h, theta, d = hough_line(image, theta=tested_angles)
#--example_1->|
#--example_1->|# Generating figure 1
#--example_1->|fig, axes = plt.subplots(1, 3, figsize=(15, 6))
#--example_1->|ax = axes.ravel()
#--example_1->|
#--example_1->|ax[0].imshow(image, cmap=cm.gray)
#--example_1->|ax[0].set_title('Input image')
#--example_1->|ax[0].set_axis_off()
#--example_1->|
#--example_1->|angle_step = 0.5 * np.diff(theta).mean()
#--example_1->|d_step = 0.5 * np.diff(d).mean()
#--example_1->|bounds = [
#--example_1->|    np.rad2deg(theta[0] - angle_step),
#--example_1->|    np.rad2deg(theta[-1] + angle_step),
#--example_1->|    d[-1] + d_step,
#--example_1->|    d[0] - d_step,
#--example_1->|]
#--example_1->|ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
#--example_1->|ax[1].set_title('Hough transform')
#--example_1->|ax[1].set_xlabel('Angles (degrees)')
#--example_1->|ax[1].set_ylabel('Distance (pixels)')
#--example_1->|ax[1].axis('image')
#--example_1->|
#--example_1->|ax[2].imshow(image, cmap=cm.gray)
#--example_1->|ax[2].set_ylim((image.shape[0], 0))
#--example_1->|ax[2].set_axis_off()
#--example_1->|ax[2].set_title('Detected lines')
#--example_1->|
#--example_1->|for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
#--example_1->|    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
#--example_1->|    ax[2].axline((x0, y0), slope=np.tan(angle + np.pi / 2))
#--example_1->|    print((x0,y0))
#--example_1->|    print(dist)
#--example_1->|    print(np.cos(angle))
#--example_1->|    print(np.sin(angle))
#--example_1->|    print(np.tan(angle + np.pi / 2))
#--example_1->|
#--example_1->|plt.tight_layout()
#--example_1->|plt.show()


#-example--->|#======================================
#-example--->|from skimage.transform import probabilistic_hough_line
#-example--->|
#-example--->|# Line finding using the Probabilistic Hough Transform
#-example--->|image = data.camera()
#-example--->|edges = canny(image, 2, 1, 25)
#-example--->|lines = probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)
#-example--->|
#-example--->|# Generating figure 2
#-example--->|fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
#-example--->|ax = axes.ravel()
#-example--->|
#-example--->|ax[0].imshow(image, cmap=cm.gray)
#-example--->|ax[0].set_title('Input image')
#-example--->|
#-example--->|ax[1].imshow(edges, cmap=cm.gray)
#-example--->|ax[1].set_title('Canny edges')
#-example--->|
#-example--->|ax[2].imshow(edges * 0)
#-example--->|for line in lines:
#-example--->|    p0, p1 = line
#-example--->|    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
#-example--->|ax[2].set_xlim((0, image.shape[1]))
#-example--->|ax[2].set_ylim((image.shape[0], 0))
#-example--->|ax[2].set_title('Probabilistic Hough')
#-example--->|
#-example--->|for a in ax:
#-example--->|    a.set_axis_off()
#-example--->|
#-example--->|plt.tight_layout()
#-example--->|plt.show()
#-example--->|
#=======================================================
# using gradient find outer edges of everything!!!

if False:
    needle_width = 70
    nmb_w = 5
    plot_image(image_org, "original", image_2, "before looking for stuff under needle")
    im_outc = region_below_needle(image_2)[:nmb_w * needle_width, :]
    im_outc = scharr(im_outc)
    px = 2
    x = np.quantile(im_outc, 1 - px / im_outc.shape[1] )
    im_outc[im_outc < x] = 0.0
    im_outc[im_outc > 0] = 1.0
    
    
    image_2 = needle_image(imread(jpgs[0], as_gray = True), 100)
    plot_image(image_2, "original", im_outc, "edges close to tip of needle")



#============================================================
#       experiment with segmentation based on watershed

#.................................................. example from
# 

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def sobel_general(im, type = "scharr"):
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ksize = -1 if type == "scharr" else 3
    gX = cv.Sobel(gray, ddepth=cv.CV_32F, dx=1, dy=0, ksize=ksize)
    gY = cv.Sobel(gray, ddepth=cv.CV_32F, dx=0, dy=1, ksize=ksize)
    gX = cv.convertScaleAbs(gX)
    gY = cv.convertScaleAbs(gY)
    combined = cv.addWeighted(gX, 0.5, gY, 0.5, 0)
    return combined


def scharr(im):
    return sobel_general(im, type = "scharr")
def sobel(im):
    return sobel_general(im, type = "sobel")

def skeletonize_4(img):
    # Invert the horse image
    #image = invert(img)
    # perform skeletonization
    return skeletonize_skim(img) #image)
    
# ===========
def save_edges_needle_and_droplet(img_path, img_path_out):
    img_raw = cv.imread(img_path)
    img = needle_image(img_raw, 100)
    edges = edges_needle_and_droplet(img) 
    os.makedirs(os.path.split(img_path_out)[0], exist_ok = True)
    cv.imwrite(img_path_out, edges)    

def edges_needle_and_droplet(img):
    edges = sobel(img)
    ret, sthr = cv.threshold(edges,32, 255, cv.THRESH_BINARY) #cv.THRESH_BINARY_INV)
    
    kernel = np.ones((3,3),np.uint8) #np.ones((3,3),np.uint8)
    im_dil = cv.dilate(sthr,kernel,iterations=1)
    im_er = cv.erode(im_dil, kernel, iterations = 1)
    im_skelet_4 = skeletonize_4(im_er)
    im_skelet_4 = 255 * im_skelet_4
    x = im_skelet_4.astype(np.uint8)
    return x

def new_image_name(path, root_in, root_out):
    return re.sub(root_in, root_out, path)    
def derive_from_new(path, root_in, root_out, prefix):
    new_path = new_image_name(path, root_in, root_out)
    path_parts = os.path.split(new_path)
    return os.path.join(path_parts[0], prefix + path_parts[1])

def needle_trace_distribution(im):
    """
    fn finds y coordinates range for tip of the needle

    """
    im_gray = im if len(im.shape) == 2 else rgb2gray(im) 
    im_gray2 = im_gray.astype(int)
    im_std = im_gray2.sum(axis = 1)
    im_max = im_std.max()
    im_min = im_std.min()
    return im_std, im_max, [int(x[0]) for x in zip(*np.where(im_std == im_max))]

def needle_trace_distribution_plot(path_in, path_out):
    img_raw = cv.imread(path_in)
    #img = needle_image(img_raw, 100)

    x, max_val, max_index = needle_trace_distribution(img_raw)
    print(x)
    print(f'max_index: {max_index}, max_val: {max_val}')
    #x = [1,2,3,4,5,4,3,2,1]
    
    plt.plot(x)
    plt.ylabel("count")
    plt.savefig(path_out, bbox_inches='tight')
    #plt.show()
    plt.close()

# how to find a drop right  under the needle
#   find where needle ends
#   take a biggest connected region that has bits at top edge
#   take only bits below needle
#   check if that remainder is close to contour

def connected_bits_below(bin_im, needle_tip_y_coord):
    bin_im = 255 * (bin_im == 255) 
    contour = largest_connected_bits_at_needle(bin_im, needle_tip_y_coord)
    #>plt.imshow(contour)
    #>plt.show()
    pixel_index = np.where(contour)
    print("connected_bits_below")
    print(pixel_index)
    #print(pixel_index.shape)
    pixel_index_select = pixel_index[0] > needle_tip_y_coord
    pixel_index_use = (pixel_index[0][pixel_index_select], pixel_index[1][pixel_index_select])
    print(pixel_index_select)
    print(pixel_index_use)
    out = np.zeros(contour.shape)
    out[pixel_index_use] = 255
    return out.astype(np.uint8)
    #print(pixel_index_select.shape)



def largest_connected_bits_at_needle(bin_im, needle_tip_y_coord):
    value_fg = 255
    bin_im_new = bin_im.astype(int)
    total_fg_pixels = bin_im_new.sum() / value_fg 
    bin_im_copy = bin_im.astype(np.uint8)
    num_labels, labels_im = cv.connectedComponents(bin_im_copy)

    #>plt.imshow(labels_im) # == 0)
    #>plt.title("labels im")
    #>plt.show()
    #>print("num_labels")
    #>print(num_labels)
    #>print(type(num_labels))
    #>print("labels_im")
    #>print(labels_im)
    #>print(type(labels_im))
    #>print(labels_im.max())

    #find contour size and if it overlaps with needle region
    contour_size = []
    part_coincides_w_needle = []
    vals = [x for x in range(1, num_labels + 1)]
    for val in vals:
        temp = np.where(labels_im == val)
        contour_size.append(len(temp[0]))
        part_coincides_w_needle.append(any(temp[0] < needle_tip_y_coord))

    #return largest contour that overlaps with needle region
    contour_sizes_sorted = sorted(contour_size)
    N = len(contour_sizes_sorted)
    for i in range(N):
        sorted_value = contour_sizes_sorted[N - 1 - i]
        index = contour_size.index(sorted_value)
        if part_coincides_w_needle[index]:
            out = labels_im == vals[index]
            #>plt.imshow(out)
            #>plt.title("largest contour")
            #>plt.show()
            return out 
    return None

def drop_contour(im):
    #,,x = needle_trace_distribution
    return 1

# how to find a drop right  under the needle
#   find where needle ends
#   take a biggest connected region that has bits at top edge
#   take only bits below needle
#   check if that remainder is close to contour

# ===========================================

def seperate_points_by_line(coefs, xs, ys, x_ref = 0, y_ref = 0):
    """
    separate given set of points by given line in two sets

    coefs: an array containing line coefficients [a, b, c] where line specified by ax + by + c = 0
    xs: an x coordinate of given points
    ys: a y coordinate of given points
    x_ref, y_reg: a point specifying which side is considered this (True) side
    """
    def delta_fn(x, y):
        return coefs[0] * x + coefs[1] * y + coefs[2] 
    
    delta_ref = delta_fn(x_ref, y_ref)

    return list(map(lambda x,y: delta_fn(x,y) * delta_ref >= 0, xs, ys))

# =================================
#find edges of needle with hanging drop
#find where tip of needle is likely --> y_needle_tip
#find all strong lines
#    select 2 strongest vertical edges --> edge1, edge2
#find line that is perpendicular to 2 selected lines AND is approximately at y_needle_tip --> line_bottom
#select all points below line_bottom


#------------------------------------------------------
# how to deal with r = x cos(theta) + y sin(theta)
# if abs(cos(theta)

def add_line_to_image(im, r, theta, value, offset = (0,0)):
    if abs(math.sin(theta)) < 0.1:
        index = 0
        fn = lambda y: (y + offset[0], round(r - y * math.sin(theta) / math.cos(theta)) + offset[1])
    else:
        index = 1 
        fn = lambda x: (round(r - x * math.cos(theta) / math.sin(theta)) + offset[0], x + offset[1])

    pts = list(map(fn, range(0, im.shape[index])))

    for p in pts:
        #print(p)
        im[p] = value
    return im


        
def needle_tip_boundary(bin_im, needle_tip_y_coord, max_theta = 90, angles = 20, delta_px = 10):
    # do hough transform
    # pick 2 strongest and vertical lines
    # find strongest horizontal line, should coincide with needle_tip_y_coord
    theta_val = np.pi * max_theta / 2 / 90
    tested_angles = np.linspace(-theta_val, theta_val, angles, endpoint=False)
    h, theta, d = hough_line(bin_im, theta=tested_angles)

    #for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    #    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    #    ax[2].axline((x0, y0), slope=np.tan(angle + np.pi / 2))
    #    print((x0,y0))
    #    print(dist)
    #    print(np.cos(angle))
    #    print(np.sin(angle))
    #    print(np.tan(angle + np.pi / 2))
    all_lines = zip(*hough_line_peaks(h, theta, d))
    print("needle_tip_boundary()")
    print(" -------------- all_lines ------------------")
    print(type(all_lines))
    print(f'items: {len(list(all_lines))}')
    plt.imshow(bin_im)
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        print(f'point: {(x0,y0)}')
        print(f'dist: {dist}')
        print(f'angle: {angle}, {angle + np.pi/2}, {angle - np.pi/2}')
        print(f'cos(angle): {np.cos(angle)}')
        print(f'sin(angle): {np.sin(angle)}')
        print(np.tan(angle + np.pi / 2))
        print(f'eq: ({x0}) * x + ({y0}) * y + ({-(x0*x0 + y0*y0)}) = 0')
        #add_line_to_image(bin_im, dist, angle, 127, (0,0))
        plt.axline((x0,y0), slope = np.tan(angle + np.pi/2))
    yst = needle_tip_y_coord - delta_px
    yend = needle_tip_y_coord + delta_px
    print(f'{yst}:{yend}')
    bin_im_2 = bin_im[yst:yend, :]
    h, theta, d = hough_line(bin_im_2, theta=tested_angles)
    print(" -------------- all_lines 2 ------------------")
    print(type(all_lines))
    print(f'items: {len(list(all_lines))}')
    first = True
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        if first:
            first = False
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            print(f'point: {(x0,y0)}')
            print(f'dist: {dist}')
            print(f'angle: {angle}, {angle + np.pi/2}, {angle - np.pi/2}')
            print(f'cos(angle): {np.cos(angle)}')
            print(f'sin(angle): {np.sin(angle)}')
            print(f'slope1: {np.tan(angle + np.pi / 2)}')
            print(f'slope2: {-x0 / y0}')
            print(f'eq: ({x0}) * x + ({y0}) * y + ({-(x0*x0 + y0*y0)}) = 0')
            #add_line_to_image(bin_im_2, dist, angle, 127, (0, 0))
            #>plt.imshow(bin_im_2)
            #>plt.title("___22___")
            #>plt.show()
            #add_line_to_image(bin_im, dist, angle, 127, (yst, 0))
            #>plt.imshow(bin_im)
            #>plt.title("________")
            #>plt.show()
            plt.axline((x0,y0 + yst), slope = np.tan(angle + np.pi/2))
            line_coefs = hough_to_AXpBYpCis0(dist, angle, offset = (0, yst))
            comp = lambda x,y: seperate_points_by_line(line_coefs, x, y)
            # have to get line_coefs for whole image!!!! not just cutout

def seperate_points_by_line(coefs, xs, ys, x_ref = 0, y_ref = 0):
    plt.show()
    if val > 0 and seperate_points_by_line(line_coefs

def hough_to_AXpBYpCis0(dist, angle, offset = (0,0)):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    x0_mod = x0 + offset[0]
    y0_mod = y0 + offset[1]
    return [x0, y0, -(x0 * x0_mod + y0 * y0_mod)]

 
# Create an image with text on it
img = np.zeros((100,400),dtype='uint8')
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'TheAILearner',(5,70), font, 2,(255),5,cv.LINE_AA)
img1 = img.copy()

img_path = 'D:/temp_data/exp_2025-03-26_06g_BrijL4_C7C16_001/scan_001/conc_0.87500/00008.jpg'
img_path = 'D:/temp_data/exp_2025-03-26_06g_BrijL4_C7C16_001/scan_001/conc_0.87500/00009.jpg'
img_path = 'D:/temp_data/exp_2025-03-26_06g_BrijL4_C7C16_001/scan_001/conc_0.87500/00066.jpg'
img_path = 'D:/temp_data/exp_2025-03-26_06g_BrijL4_C7C16_001/scan_005/conc_0.87500/00066.jpg'
img_path = jpgs[0]
img_path = 'D:/temp_data/exp_2025-03-26_06g_BrijL4_C7C16_001/scan_006/conc_0.87500/00066.jpg'

#--->| aaa = np.array([x for x in range(1000)])
#--->| print(aaa < 500)
#--->| xxx = np.array([ x % 3 for x in range(1000)])
#--->| print(xxx)
#--->| print(xxx.shape)
#--->| yyy = np.where(xxx == 0)
#--->| print(yyy)
#--->| print(yyy[0])
#--->| print(len(yyy[0]))

for i,p in enumerate(jpgs):
    if i < 1450 and i > 1447: #10: #2500: # and i >= 1500:
        output_img_path = new_image_name(p, root, root_out)
        print(f'{i: >3d}: {output_img_path}')
        save_edges_needle_and_droplet(p, output_img_path)

        #>edge_distribution_path = derive_from_new(p, root, root_out, "a_trace_distr_")
        #>print(f'{i: >3d}: {edge_distribution_path}')
        #>needle_trace_distribution_plot(output_img_path, edge_distribution_path)        

        im = imread(output_img_path)
        im = 255 * (im > 127) #saving to jpg creates artifacts that can be removed by thresholding image!!!
        #>plt.imshow(im)
        #>plt.title("1")
        #>plt.show()
        x_sec,max_val,max_val_indexes = needle_trace_distribution(im)
        plt.imshow(im)
        plt.title("2")
        plt.show()

        #__>llcb = connected_bits_below(im, max_val_indexes[0])
        #__>contour_path = derive_from_new(p, root, root_out, "a_drop_contour_")
        #__>#>plt.imshow(llcb)
        #__>#>plt.title("3")
        #__>#>plt.show()
        #__>imsave(contour_path, llcb)
        needle_tip_boundary(im, max_val_indexes[0], angles = 720)


#OLDO_bits>|#============
#OLDO_bits>|img_raw = cv.imread(img_path)
#OLDO_bits>|print(img_raw)
#OLDO_bits>|print(type(img_raw))
#OLDO_bits>|print(img_raw.shape)
#OLDO_bits>|print(len(img_raw.shape))
#OLDO_bits>|img = needle_image(img_raw, 100)
#OLDO_bits>|img_org = img.copy()
#OLDO_bits>|#img = needle_image(imread(jpgs[0], as_gray = True), 100)
#OLDO_bits>|
#OLDO_bits>|edges = sobel(img)
#OLDO_bits>|if False: 
#OLDO_bits>|    plot_image(img, "img", edges, "edges")
#OLDO_bits>|
#OLDO_bits>|ret, sthr = cv.threshold(edges,32, 255, cv.THRESH_BINARY) #cv.THRESH_BINARY_INV)
#OLDO_bits>|if False: 
#OLDO_bits>|    plot_image(edges, "img", sthr, "sthr")
#OLDO_bits>|print("sthr=================================")
#OLDO_bits>|print(sthr)
#OLDO_bits>|print("sthr=================================")
#OLDO_bits>|
#OLDO_bits>|kernel = np.ones((3,3),np.uint8) #np.ones((3,3),np.uint8)
#OLDO_bits>|im_dil = cv.dilate(sthr,kernel,iterations=1)
#OLDO_bits>|if False: 
#OLDO_bits>|    plot_image(edges, "img", im_dil, "dilated image")
#OLDO_bits>|
#OLDO_bits>|im_er = cv.erode(im_dil, kernel, iterations = 1)
#OLDO_bits>|if False: 
#OLDO_bits>|    plot_image(edges, "img", im_er, "eroded image")
#OLDO_bits>|
#OLDO_bits>|
#OLDO_bits>|im_skelet_4 = skeletonize_4(im_er)
#OLDO_bits>|if False:
#OLDO_bits>|    plot_image(edges, "img", im_skelet_4, "skeletonized image 4")
#OLDO_bits>|
#OLDO_bits>|im_skelet_4 = 255 * im_skelet_4
#OLDO_bits>|x = im_skelet_4.astype(np.uint8)
#OLDO_bits>|
#OLDO_bits>|added_image = cv.addWeighted(img,0.7,cv.cvtColor(x, cv.COLOR_GRAY2RGB),0.2,0)
#OLDO_bits>|if False: 
#OLDO_bits>|    plot_image(img_org, "", added_image, "")
#OLDO_bits>|
#OLDO_bits>|assert img is not None, "file could not be read, check with os.path.exists()"
#OLDO_bits>|#gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#OLDO_bits>|gray = scharr(img)
#OLDO_bits>|ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
#OLDO_bits>|
#OLDO_bits>|if False: 
#OLDO_bits>|    plot_image(gray, "gray", thresh, "thresh")
#OLDO_bits>|
#OLDO_bits>|# noise removal
#OLDO_bits>|kernel = np.ones((2,2),np.uint8) #np.ones((3,3),np.uint8)
#OLDO_bits>|opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 1)
#OLDO_bits>|
#OLDO_bits>|if False: 
#OLDO_bits>|    plot_image(thresh, "original", opening, "opening")
#OLDO_bits>| 
#OLDO_bits>|# sure background area
#OLDO_bits>|sure_bg = cv.dilate(opening,kernel,iterations=3)
#OLDO_bits>|
#OLDO_bits>|if False: 
#OLDO_bits>|    plot_image(opening, "opening", sure_bg, "sure_bg")
#OLDO_bits>|
#OLDO_bits>|# Finding sure foreground area
#OLDO_bits>|#-->dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
#OLDO_bits>|#-->ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
#OLDO_bits>|sure_fg = cv.erode(sure_bg, kernel, iterations = 6)
#OLDO_bits>|
#OLDO_bits>|#plot_image(opening, "opening", dist_transform, "dist_tr")
#OLDO_bits>|if False: 
#OLDO_bits>|    plot_image(opening, "opening", sure_fg, "sureg_fg")
#OLDO_bits>| 
#OLDO_bits>|# Finding unknown region
#OLDO_bits>|sure_fg = np.uint8(sure_fg)
#OLDO_bits>|unknown = cv.subtract(sure_bg,sure_fg)
#OLDO_bits>|
#OLDO_bits>|if False: 
#OLDO_bits>|    plot_image(opening, "opening", unknown, "unknown")
#OLDO_bits>|
#OLDO_bits>|# Marker labelling
#OLDO_bits>|ret, markers = cv.connectedComponents(sure_fg)
#OLDO_bits>|
#OLDO_bits>|## Add one to all labels so that sure background is not 0, but 1
#OLDO_bits>|#markers = markers+1
#OLDO_bits>|#
#OLDO_bits>|## Now, mark the region of unknown with zero
#OLDO_bits>|#markers[unknown==255] = 0
#OLDO_bits>|
#OLDO_bits>|if False: 
#OLDO_bits>|    plot_image(img, "img", markers, "markers")
#OLDO_bits>|
#OLDO_bits>|markers2 = cv.watershed(img,markers)
#OLDO_bits>|img[markers2 == -1] = [255,0,0]
#OLDO_bits>|
#OLDO_bits>|if False: 
#OLDO_bits>|    plot_image(img, "1", gray, "2")
#OLDO_bits>|
#OLDO_bits>|print(img.shape)
#OLDO_bits>|print(img.dtype)
#OLDO_bits>|print(markers.shape)
#OLDO_bits>|print(markers.dtype)
#OLDO_bits>|temp =  cv.normalize(markers,  None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
#OLDO_bits>|added_image = cv.addWeighted(img,0.4,cv.cvtColor(temp, cv.COLOR_GRAY2RGB),0.4,0)
#OLDO_bits>|if False: 
#OLDO_bits>|    plot_image(img, "", added_image, "")



print(f'duration: {time.time() - start}')


