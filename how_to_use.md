---
title: 'How to use bits'
date: 2025-04-29
---

```{mermaid; OLD, delete once NEW is updated}
flowchart TB

    Main(experiment) --> RawData(data.json files w/ metadata)
    Main --> Images(raw PDT images)
    Images -->|scriptA| DropStats(droplet /regions under needle/ stats, table, record per region)
    Images -->|create_experiment_copy_w_reduced_images.py| ImagesSmall(only relevant, needle, bits, optional, convenience)
    ImagesSmall -->|scriptA| DropStats
    RawData -->|extract_experimental_params_from_data_json.py| RawDataTable(composition to images, table, record per image)
    Images -->|IFT_app_v12_2025_02_06_ap.py| IFT_val(IFT values)
    ImagesToDropStats(scriptA = extract_droplet_stats.py, scriptB = IFT_app_v12_2025-02-06.py)
```

```{mermaid; NEW folder content}
<experiment_root>
    data.json
    log.log
    scan_001/
        conc_0.0000/
            00000.jpg
            ...
            0000n.jpg
        ...
        conc_1.0000/
    ...
    scan_00n/
    processed/
        data_processed.json
        log_digest/
            _LOG_CMDS_
            _LOG_URLS_
        montages/
```

```{explanation of content}
data.json: contains all experimental metadata, solution content, ift values, image  location, etc.
log.log: a raw log of opentron commands passed from PC
scan_001/ (and similarly named folders): contain individual scans performed during experiment
conc_0.0000/ (and similarly named folders): contain images for an individual measuement
processed/: contains processed data
data_processed.json: this is a processed data.json file 
log_digest/: folder containing human readable extracts from raw log file
_LOG_CMDS_: human readable list of commands sent to opentron
_LOG_URLS_: human readable list of urls for all requests to opentron
montages/: folder for experimental and measurement montages
```


```{modifications}
data.json --> |hld_ift_analysis_helpers/experiment_data_json_offline_ift.py| processed/data_processed.json
data_processed.json --> |hld_ift_analysis_helpers/utils/data_json_extraction/extract_experimental_params_from_data_json.py| processed/????extract_data.csv????

<Measurement images>: all images in folders ending with `conc_x.xxxxx`; other folders contain autofocusing images that are not useful

<Measurement images> --> |hld_ift_analysis_helpers/utils/extract_droplet_stats.py| processed/drop_stats.csv

<Measurement images> --> |hld_ift_analysis_helpers/utils/visualize/make_montages_experiment.py| processed/montages/exp_montage.jpg
<Measurement images> --> |hld_ift_analysis_helpers/utils/visualize/make_montages_measurements.py| processed/montages/measurements/

log.log --> |hld_ift_analysis_helpers/utils/extract__bits_from_logs.py| <digested_logs>
```

```{script explanations}
|hld_ift_analysis_helpers/experiment_data_json_offline_ift.py|: offline evaluation of ift values
|hld_ift_analysis_helpers/utils/extract_droplet_stats.py|: extract stats of regions dripping under needle
|hld_ift_analysis_helpers/utils/data_json_extraction/extract_experimental_params_from_data_json.py|: extracts experimental data and metadata paramters into csv table from data.json (or data_processed.json)
|hld_ift_analysis_helpers/utils/visualize/make_montages_experiment.py|: creates montage of a whole experiment
|hld_ift_analysis_helpers/utils/visualize/make_montages_measurements.py|: creates montage of each individual measuement
|hld_ift_analysis_helpers/utils/extract__bits_from_logs.py|: extracts various bits from logs in human readable form
```


scripts

./scripts/ift_data_extraction/extract_ift_data_to_csv.py
./scripts/ift_data_extraction/IFT_app_v12_2025-02-06.py
./src/hld_ift_analysis_helpers/collect_files_folders.py
./src/hld_ift_analysis_helpers/droplet_stats.py
./src/hld_ift_analysis_helpers/experiment_hough_transform.py
./src/hld_ift_analysis_helpers/experiment_hough_transform_v2.py
./src/hld_ift_analysis_helpers/IFT_app_v12_2025_02_06_ap.py
./src/hld_ift_analysis_helpers/json_data_extraction.py
./src/hld_ift_analysis_helpers/json_data_extraction__example.py
./src/hld_ift_analysis_helpers/montage_bits.py
./src/hld_ift_analysis_helpers/utils/data_json_extraction/extract_experimental_params_from_data_json.py
./src/hld_ift_analysis_helpers/utils/data_json_extraction/save_conversion_dict.py


```{command prompt}
cd /D C:\Users\Aigar\miniconda3\
set LOC="D:\temp_data"
set SCR_SRC="D:\projects\HLD_parameter_determination\hld_ift_http\test"
set SRC_ANF="D:\projects\HLD_parameter_determination\hld_ift_analysis_helpers\src\hld_ift_analysis_helpers"
#python "%SCR_SRC%\IFT_app_v12_2025-02-06.py"   #generate ift values offline
#python %LOC%\create_experiment_copy_w_reduced_images.py #reduces large experiments to smaller
#python %LOC%\extract_experimental_params_from_data_json.py
#python "%LOC%\extract_droplet_stats.py"        #calculate droplet stats
#python %LOC%\make_montages_experiment.py
#python %LOC%\make_montages_measurements.py
#python %SRC_ANF%\utils\visualize\make_montages_experiment.py \\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\AOT_IB-45_C7C16_NaCl
```

