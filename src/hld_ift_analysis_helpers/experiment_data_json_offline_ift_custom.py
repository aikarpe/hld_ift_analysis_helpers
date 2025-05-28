"""
script to calculate ift values for acquired images after experiment
run:
    python experiment_data_json_offline_ift.py <source>
    
    <source>: full path to `data.json` (<experiment_root>/data.json) to be processed
    
    script output: new json file like `data.json` with updated ift values
            output location: <experiment_root>/processed/data_processed.json


"""

import sys
sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_calc/src")
sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/src")

import re
import json
import os
import cv2
import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt

from hld_ift_calc.image_to_contour import (
                                        ImageFiltering,
                                        DropletExtremePoints,
                                        Contours,
                                        ImagePreprocessor,
                                        NeedleDiameter
                                        )
from hld_ift_calc.contour_to_ift import (
                                        Interpolator,
                                        YoungLaplaceShape,
                                        IFTCalculator
                                        )

from hld_ift_analysis_helpers.collect_files_folders import collect_data_jsons
from hld_ift_analysis_helpers.locations import data_json_path_to_processed_data_json_path

import argparse

# ============================================================ Detection and classification
# __TYPE_CONTOUR_LIST__ list[numpy.ndarray]
# __TYPE_IMAGE__ numpy.ndarray

class DetectionClassifier:
    def __init__(self, min_contour_area=300, line_length_threshold=20):
        """
        f-n initializes detection classifier with thresholds

        :param int min_contour_area: min pixel area to consider a contour
        :param int line_length_threshold: min length to detect continuous stream
        """
        self.min_contour_area = min_contour_area
        self.line_length_threshold = line_length_threshold

    def classify(self, contours, image_resized):
        """
        f-n classifies presence of needle, droplet, or stream in image

        :param __TYPE_CONTOUR_LIST__ contours: list of found contours
        :param __TYPE_IMAGE__ image_resized: downsampled BGR image for drawing
        :return: label, annotated image, y-coordinate of lowest needle point
        :rtype: (str, numpy.ndarray, int)
        """
        lowest_y, needle_detected = 0, False

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                needle_detected = True
                cv2.drawContours(image_resized, [contour], -1, (0, 255, 0), 1)
                lowest_y = max(lowest_y,
                               max(contour, key=lambda p: p[0][1])[0][1])

        if not needle_detected:
            return 'Needle not detected', image_resized, lowest_y

        adjusted_y = lowest_y + 4
        cv2.line(
            image_resized,
            (0, adjusted_y),
            (image_resized.shape[1], adjusted_y),
            (0, 0, 255), 1
        )

        # apply edge detection just below needle line
        below = cv2.Canny(
            cv2.cvtColor(image_resized[adjusted_y+1:], cv2.COLOR_BGR2GRAY),
            50, 150
        )
        contours_below, _ = cv2.findContours(
            below, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours_below:
            return 'Droplet or stream not detected', image_resized, lowest_y

        # detect any long line segments indicating stream/gel
        line_detected = False
        for cnt in contours_below:
            cnt_shift = cnt + np.array([[0, adjusted_y+1]])
            for i in range(len(cnt)-1):
                p1 = tuple(cnt[i][0])
                p2 = tuple(cnt[i+1][0])
                if np.hypot(p2[0]-p1[0], p2[1]-p1[1]) > self.line_length_threshold:
                    line_detected = True
                    cv2.drawContours(image_resized, [cnt_shift], -1, (0, 255, 255), 1)
                    break

        return ('Stream/Gel' if line_detected else 'Droplet Detected',
                image_resized, lowest_y)


# ============================================================ Main processing pipeline

def do_classification_A(img):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w//3, h//3), interpolation=cv2.INTER_AREA)
    gray = ImageFiltering.to_gray(small)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, binary = cv2.threshold(blur, 35, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(binary,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

    cls_label, vis, ly = classifier.classify(cnts, small)

    return cls_label


def process_an_image(fpath, rho_w, rho_o): #, classifier: DetectionClassifier, ndc: NeedleDiameter):
    """
        :returns (drop_status_str, error_str, ift_val) (str, str, float): 
    """
    classifier = DetectionClassifier()
    ndc = NeedleDiameter()
    
    delta = rho_w - rho_o
    #ift_calc = IFTCalculator(delta, ndc)

    base = os.path.basename(fpath)
    img = cv2.imread(fpath)
    if img is None:
        return ('no_image', 'no_image', np.nan)

    # Stage 1: needle and droplet detection
    h, w = img.shape[:2]
    small = cv2.resize(img, (w//3, h//3), interpolation=cv2.INTER_AREA)
    gray = ImageFiltering.to_gray(small)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, binary = cv2.threshold(blur, 35, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(binary,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

    cls_label, vis, ly = classifier.classify(cnts, small)
    #cv2.putText(vis, cls_label, (10,30),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    #cv2.imwrite(os.path.join(meas_f, f"{base}_detection.png"), vis)

    if cls_label != 'Droplet Detected':
        return (cls_label, "", np.nan)

    # Stage 2: convert image to contour for IFT
    den = ImageFiltering.apply_maximum_denoising(img)
    band = ImagePreprocessor.process_image_with_marking_in_band(den)
    if band is None:
        return (cls_label, 'Band processing failed', np.nan) 

    sx_b, top, low, left, right, high, axis, ath = band

    mask = ath.copy()
    buf = min(high + 15, ath.shape[0] - 1)
    mask[buf:] = 0
    mask[:high-50] = 0
    shape_mask, _ = ImagePreprocessor.find_largest_white_shape(mask, high)
    if shape_mask is None:
        return (cls_label, 'Largest shape not found', np.nan) 

    # Stage 3: compute interfacial tension from contour
    drop_h = abs(low - high)
    if left and right:
        lx, rx = left[1], right[1]
        de = abs(rx - lx)
    else:
        lx = rx = de = None

    nd_px, markers = ndc.measure(img, high, axis)
    scale = (0.312 / nd_px)*1e-3 if nd_px and nd_px > 0 else None

    if drop_h and de:
        if drop_h > 2*de or de > 2*drop_h:
            return (cls_label, 'g/s', np.nan) 
        if scale and lx is not None and rx is not None:
            ann, td, xl, yl, xr, yr, ax, ay, r0px = Interpolator.interpolate_and_filter_pchip(
                img, high, low, lx, rx
            )
            if ann is None:
                return (cls_label, 'Interpolation failed', np.nan) 
            r0 = r0px * scale if r0px else None
            sigma = YoungLaplaceShape(delta).compute_ift((td, xl, yl), r0) if r0 else None
            if sigma:
                ift_val = sigma * 1e6
                res = f"{ift_val:.6f} mN/m"
                return (cls_label, '', ift_val) 
            else:
                return (cls_label, 'Calculation error', np.nan) 
        else:
            res, r0 = 'Invalid scale or boundaries', None
            return (cls_label, 'Calculation error', np.nan) 
    else:
        return (cls_label, 'h/n', np.nan) 


# ---------- translation template ----------------------------------------
def translate_experiment(a_dict, **kwargs):
    dict_out = a_dict.copy()
    dict_out["scans"] = list(map(lambda x: translate_scan(x), a_dict["scans"]))
    return dict_out

def translate_scan(a_dict, **kwargs):
    dict_out = a_dict.copy()
    dict_out["measurements"] = list(map(lambda x: translate_measurement(x), a_dict["measurements"]))
    return dict_out

def translate_measurement(a_dict, **kwargs):
    dict_out = a_dict.copy()
    dict_out["ift_images"] = list(map(lambda x: translate_image(x), a_dict["ift_images"]))
    return dict_out

def translate_image(a_dict, **kwargs):
    dict_out = a_dict.copy()
    return dict_out

# --------------------------------------------------------------------------------

# ---------- offline ift detection ----------------------------------------

def replace_path_start(a_path, path_starts_with, path_replace_with):
    if a_path == path_starts_with:
        return path_replace_with
    else:
        cmpnnts = os.path.split(a_path)
        return os.path.join(replace_path_start(cmpnnts[0], path_starts_with, path_replace_with), cmpnnts[1])

def translate_experiment(a_dict, **kwargs):
    print("will update ift values")
    dict_out = a_dict.copy()
    dict_out["scans"] = list(map(lambda x: translate_scan(x, 
                                                            path_starts_with = os.path.split(a_dict["root"])[0],
                                                            path_replace_with = kwargs["path_replace_with"]
                                                            ), 
                                a_dict["scans"]))
    return dict_out

def translate_scan(a_dict, **kwargs):
    dict_out = a_dict.copy()
    print(f'processing scan:\n{dict_out["root"]}')
    if not re.search("scan_[0-9]{3}$",
                # .\AOT_IB-45_C7C16_NaCl\exp_2025-04-09_10.00g_AOT_C7C16_002
                #dict_out["root"]) or dict_out["label"] in ["scan_002", "scan_004"]:                       ### edit condition
                # .\AOT_IB-45_C7C16_NaCl\exp_2025-05-06_10.00g_AOT_C7_07.50g_nacl_001
                #dict_out["root"]) or dict_out["label"] in ["scan_002", "scan_003"]:                       ### edit condition
                # .\AOT_IB-45_C7C16_NaCl\exp_2025-05-07_05.00g_AOT_C7_07.50g_nacl_001
                #dict_out["root"]) or dict_out["label"] in ["scan_003"]:         #["scan_001", "scan_003"]:                       ### edit condition
                # .\AOT_IB-45_C7C16_NaCl\exp_2025-05-08_10.00g_AOT_C7_07.50g_nacl_001
                #dict_out["root"]) or dict_out["label"] in ["scan_002", "scan_003"]:       #["scan_001", "scan_002", "scan_003"]                ### edit condition
                # .\AOT_IB-45_C7C16_NaCl\exp_2025-05-13_10.00g_AOT_C7_07.50g_nacl_001
                dict_out["root"]) or dict_out["label"] in ["scan_003"]:         
        print(f'will copy as is scan: {dict_out["label"]}')
        k = input("press <ENTER>")
        return dict_out
    dict_out["measurements"] = list(map(lambda x: translate_measurement(x,
                                                                        path_starts_with = kwargs["path_starts_with"],
                                                                        path_replace_with = kwargs["path_replace_with"]
                                                                        ), 
                                    a_dict["measurements"]))
    return dict_out

def translate_measurement(a_dict, **kwargs):

    dict_out = a_dict.copy()
    print(f'processing scan:\n{dict_out["measurement_folder"]}')
    if re.search("autofocus",
                os.path.split(dict_out["measurement_folder"])[1]):
        return dict_out
    
    dict_out["ift_images"] = list(map(lambda x: translate_image(x,
                                                                ro_inner = a_dict["solution_inner"]["ro"], 
                                                                ro_outer = a_dict["solution_outer"]["ro"],
                                                                path_starts_with = kwargs["path_starts_with"],
                                                                path_replace_with = kwargs["path_replace_with"]
                                                                ),
                                    a_dict["ift_images"]))
    return dict_out

def translate_image(a_dict, **kwargs):

    dict_out = a_dict.copy()
    # open image at modified path
    # process the image
    # report error in status, if needed
    # report ift value 
    #process_an_image(fpath, rho_w, rho_o): #, classifier: DetectionClassifier, ndc: NeedleDiameter):
    st, err_st, val = process_an_image(
                                    replace_path_start(
                                                    dict_out["path"],
                                                    kwargs["path_starts_with"],
                                                    kwargs["path_replace_with"]),
                                    kwargs["ro_inner"],
                                    kwargs["ro_outer"])
    dict_out["analyzed"]  = True
    dict_out["status"] = st
    dict_out["ift"] = val
    if st == "Droplet Detected": 
        dict_out["failed_analysis"] = False
    else:
        dict_out["failed_analysis"] = True
        dict_out["failure"] = err_st
   
    return dict_out


#=========================================================================================
#                                                   arguments

parser = argparse.ArgumentParser()
parser.add_argument("source", help = "source of data.json file(s); can be a path to file, a path to file containing list of pathes or a folder")
parser.add_argument("-e", "--extraction_options", help = "a path to file that specifies extraction options: default value, value path in a data.json file, target variable name", default = "")
parser.add_argument("-v", "--view", help = "view option analyzes input files and summarizes unique variable pathes in data.json file(s)", action = "store_true")
args = parser.parse_args()



# select source files
file_path = []
#extraction_options = args.extraction_options if os.path.isfile(args.extraction_options) else ""

def process_string_pointing_to_data_json_file(astr):
    if os.path.split(astr)[1] == "data.json":
        file_path.append(astr)

if os.path.isfile(args.source):
    if os.path.split(args.source)[1] == "data.json":
        # a single source file
        process_string_pointing_to_data_json_file(args.source) 
    else:
        # a file containing list of source files
        with open(args.source, 'r') as file:
            for line in file:
                process_string_pointing_to_data_json_file(line) 

if os.path.isdir(args.source):
    for f in collect_data_jsons(args.source):
        process_string_pointing_to_data_json_file(f)

if len(file_path) == 0:
    print(f'not sure what to do with given source:\n `{args.source}`\n exiting')
    exit()



#x = r'\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\AOT_IB-45_C7C16_NaCl\exp_2025-04-07_05g_AOT_C7C16_001\scan_001\conc_0.03125\00001.jpg'
#
#imtemp = cv2.imread(x)
#plt.imshow(imtemp)
#plt.title(f'imtemp example!!!')
#plt.show()

#data_json_path = r'\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\AOT_IB-45_C7C16_NaCl\exp_2025-04-07_05g_AOT_C7C16_001\data.json'

#def exp_root_fldr_to_processed_fldr_path(a_path):
#    return os.path.join(a_path, "processed")

#def data_json_path_to_processed_data_json_path(a_path):
#    fldr, fnm = os.path.split(a_path)
#    if fnm == "data.json":
#        processed_fldr_path = exp_root_fldr_to_processed_fldr_path(fldr) 
#        os.path.makedirs(processed_fldr_path, exist_ok = True)
#        return os.path.join(processed_fldr_path, "data_processed.json")
#    else:
#        raise Exception("Not a data.json file path")
#

def process_offline_ift_estimate(data_json_path, data_json_path_out):
    data_json_dict = None
    with open(data_json_path, "r") as f:
        data_json_dict = json.load(f)
    
    if data_json_dict is None:
        print(f'failed to open:\n {data_json_path}\n .... terminating now ......')
        return
    
    out = translate_experiment(
                data_json_dict,
                path_replace_with = os.path.split(os.path.split(data_json_path)[0])[0]
                )
    
    
    with open(data_json_path_out, "w") as ff:
       json.dump(out, ff, indent = 4) 
    
for apath in file_path:
    process_offline_ift_estimate(
                                apath,
                                data_json_path_to_processed_data_json_path(apath)
                                )
