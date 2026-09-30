---
title: "data analysis notes"
date: 2026-06-29
---

Steps neede for data analysis

1. ift (offline) calculation (not really used anymore)
2. data.json convertion into data table (<name???>)
3. raw image classification using adam's model
4. image stat extraction
5. log extraction (optional)



### general setup

```{anaconda shell}
cd /D C:\Users\agaiosa\miniconda3\
conda activate hld_ift1
REM set LOC="D:\temp_data"
REM set AN_SRC="D:\projects\HLD_parameter_determination\hld_ift_analysis_helpers\src\hld_ift_analysis_helpers"
set AN_SRC=C:\Users\agaiosa\code\hld_ift_analysis_helpers\src\hld_ift_analysis_helpers
set ROBOLAB=\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\HLD_scan
set LOCAL_LOC=C:\Users\agaiosa\delme\hld_analysis\processing_logs

set YEAR=%DATE:~6,4%
set MONTH=%DATE:~3,2%
set DAY=%DATE:~0,2%
set DATE_PRETTY=%YEAR%-%MONTH%-%DAY%

set PROJECT_NAME=VLCI_site_Ecosurf_EH-3

set SURF_PRJ_PATH=%ROBOLAB%\%PROJECT_NAME%
set SURF_EXT_PATH=%SURF_PRJ_PATH%\.processing_config\_data.json_extraction_options.json
REM mkdir %SURF_PRJ_PATH%\.processing_config
REM echo "" > %SURF_EXT_PATH%

set PROC_LOG=%LOCAL_LOC%\%PROJECT_NAME%.processing_log.%DATE_PRETTY%.log
set PROC_LOG_REMOTE=%SURF_PRJ_PATH%\processing_log.%DATE_PRETTY%.log

REM #--------------------------------------------------------------------------------
REM # experiment names for (re-)using

set EXPERIMENT_NAME=exp_2026-05-11_5g100mL_Ecosurf_EH3_C7C16_test_run1
set EXPERIMENT_NAME=exp_2026-05-11_5g100mL_Ecosurf_EH3_C7C16_test_run2
set EXPERIMENT_NAME=exp_2026-05-11_5g100mL_Ecosurf_EH3_C7C16_test_run3
set EXPERIMENT_NAME=exp_2026-05-21_15g100mL_Ecosurf_EH3_C7C16_test_run1
set EXPERIMENT_NAME=exp_2026-05-21_15g100mL_Ecosurf_EH3_C7C16_test_run2
set EXPERIMENT_NAME=exp_2026-05-21_15g100mL_Ecosurf_EH3_C7C16_test_run3

set EXPERIMENTS=exp_2026-05-11_5g100mL_Ecosurf_EH3_C7C16_test_run1 exp_2026-05-11_5g100mL_Ecosurf_EH3_C7C16_test_run2 exp_2026-05-11_5g100mL_Ecosurf_EH3_C7C16_test_run3 exp_2026-05-21_15g100mL_Ecosurf_EH3_C7C16_test_run1 exp_2026-05-21_15g100mL_Ecosurf_EH3_C7C16_test_run2 exp_2026-05-21_15g100mL_Ecosurf_EH3_C7C16_test_run3

echo %SURF_PRJ_PATH%
echo %PROC_LOG_REMOTE%
echo %PROC_LOG%

```

### generic commands

```{anaconda-prompt commands}
REM used once for setting up extraction path from json to csv table
for %i in (%EXPERIMENTS%) do (
    python %AN_SRC%\utils\data_json_extraction\extract_experimental_params_from_data_json.py  "%SURF_PRJ_PATH%\%i\data.json" -v >> %PROC_LOG%
)


REM commands to make raw montages
for %i in (%EXPERIMENTS%) do (
    python %AN_SRC%\utils\visualize\make_montages_experiment.py  "%SURF_PRJ_PATH%\%i\data.json" -i 100 -n 10 >> %PROC_LOG%
    python %AN_SRC%\utils\visualize\make_montages_measurements.py  "%SURF_PRJ_PATH%\%i\data.json" >> %PROC_LOG%
)


REM specific extraction commands
for %i in (%EXPERIMENTS%) do (
    echo =========================================================== >> %PROC_LOG%
    echo "             %i" >> %PROC_LOG%
    echo =========================================================== >> %PROC_LOG%
    REM python %AN_SRC%\experiment_data_json_offline_ift.py  "%SURF_PRJ_PATH%\%i\data.json" >> %PROC_LOG%
    python %AN_SRC%\utils\data_json_extraction\extract_experimental_params_from_data_json.py  "%SURF_PRJ_PATH%\%i\data.json" -o -e %SURF_EXT_PATH% >> %PROC_LOG%
    python %AN_SRC%\utils\extract__bits_from_logs.py  "%SURF_PRJ_PATH%\%i\data.json" >> %PROC_LOG%
    python %AN_SRC%\utils\extract_droplet_stats.py "%SURF_PRJ_PATH%\%i\data.json" -a -m 0.8 -w 150 >> %PROC_LOG%
)


```

### raw image classification

```{anaconda-prompt raw-image-classification}
conda activate hld_ift0
set MODEL=\\huckdfs-srv.science.ru.nl\huckrobotlab\WOOW\WOOW_HLD
set SRC=\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\HLD_scan

set PROJECT=Ecosurf_EH_3_VCLI_C7_C16
set EXPERIMENTS=exp_2025-11-12_10.00g_Ecosurf_EH_3_VCLI_C7_C16_NaCl_001 exp_2025-11-13_10.00g_Ecosurf_EH_3_VCLI_C7_C16_NaCl_001 exp_2025-11-13_20.00g_Ecosurf_EH_3_VCLI_C7_C16_NaCl_001 exp_2025-11-14_15.00g_Ecosurf_EH_3_VCLI_C7_C16_NaCl_001

for %i in (%EXPERIMENTS%) do (
    python %MODEL%\image_classification.py --checkpoint %MODEL%\best.pth --phase-dir %SRC%\%PROJECT%\%i
)
```


### general assumptions

- several experiments are combined into single project folder that contains related experiments (e.g. differenct surfactant concentrations)
- analysis is done for experiments in a project folder together
- all analysis share single extraction template

### general workflow

- copy `status.md` and edit relevant details (path to project, experiment names)
- create `...\.processing_config\_data.json_extraction_options.json` that contains how experimental data is extracted
- prepare `data_set_inputs.json` file for analysis
    - edit data manipulation lines
    - add experiments




### typical analysis sequence

