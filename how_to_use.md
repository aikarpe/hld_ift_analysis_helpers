---
title: 'How to use bits'
date: 2025-04-29
---

```
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

scripts
:
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

