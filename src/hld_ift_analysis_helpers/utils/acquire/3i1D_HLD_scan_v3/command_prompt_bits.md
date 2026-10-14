conda activate hld_ift0
cd C:\Users\admin\Documents\Data\aikars\opentron\TergitolS_15_S_3n15_C7C16_NaCl_v1
set SCRIPT="C:\Users\admin\code\hld_ift_analysis_helpers\src\hld_ift_analysis_helpers\utils\acquire"
set SETTINGS=scan_settings.json
echo %SCRIPT%

python %SCRIPT%\new_profile_wizard.py
python %SCRIPT%\solution_repository_editor.py %SETTINGS%

python %SCRIPT%\3i1D_HLD_scan_v3__setup_configuration.py %SETTINGS%
python %SCRIPT%\3i1D_HLD_scan_v3__execute.py %SETTINGS%


