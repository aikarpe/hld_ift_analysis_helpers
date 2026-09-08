conda activate hld_ift0
cd C:\Users\admin\Documents\Data\aikars\opentron\TergitolS_15_S_3n9_C7C16_NaCl_v1
REM set SCRIPT="\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\scripts"
set SCRIPT="C:\Users\admin\code\hld_ift_analysis_helpers\src\hld_ift_analysis_helpers\utils\acquire"
set SETTINGS=scan_settings.json
echo %SCRIPT%

REM ================================================== create new experiment set 

python %SCRIPT%\new_profile_wizard.py
python %SCRIPT%\solution_repository_editor.py %SETTINGS%


REM ========================================== prepare and execute an experiment

python %SCRIPT%\2D_HLD_scan_v3__prepare_solutions.py %SETTINGS%
python %SCRIPT%\2D_HLD_scan_v3__setup_configuration.py %SETTINGS%
python %SCRIPT%\2D_HLD_scan_v3__execute.py %SETTINGS%


REM ==================================================== simple measurement test

python %SCRIPT%\2D_HLD_scan_v2__measurement_only_OIW.py %SETTINGS%
python %SCRIPT%\2D_HLD_scan_v2__measurement_only_OIW_2.py %SETTINGS%



================================================================================

