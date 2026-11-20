conda activate hld_ift0
cd __path_data_folder___
set SCRIPT="__path_rep__\hld_ift_analysis_helpers\src\hld_ift_analysis_helpers\utils\acquire"
set SCRIPT_EXTR="__path_rep__\hld_ift_analysis_helpers\src\hld_ift_analysis_helpers\utils\data_json_extraction"
set SETTINGS=scan_settings.json
echo %SCRIPT%
echo %SETTINGS%


python %SCRIPT%\new_profile_wizard.py
python %SCRIPT%\solution_repository_editor.py %SETTINGS%
python %SCRIPT%\2D_HLD_scan_v3__prepare_solutions.py %SETTINGS%
python %SCRIPT%\2D_HLD_scan_v3__setup_configuration.py %SETTINGS%
python %SCRIPT%\2D_HLD_scan_v3__execute.py %SETTINGS%
python ..\test_move_hold_meas_loc.py

python %SCRIPT%\2D_HLD_scan_v2__measurement_only_OIW.py %SETTINGS%
python %SCRIPT%\2D_HLD_scan_v2__measurement_only_OIW_2.py %SETTINGS%


python %SCRIPT%\3i1D_HLD_scan_v3__setup_configuration.py %SETTINGS%
python %SCRIPT%\3i1D_HLD_scan_v3__execute.py %SETTINGS%

================================================================================
================================================================================

REM ................................................................. 2D_HLD_v4:
================================================================================

python %SCRIPT%\2D_HLD_scan_v4__prepare_solutions.py %SETTINGS%
python %SCRIPT%\2D_HLD_scan_v4__setup_configuration.py %SETTINGS%
python %SCRIPT%\2D_HLD_scan_v4__execute.py %SETTINGS%




REM .................................... 2D_HLD_OIW_v4 and 2D_HLD_OIW_logout_v4:
================================================================================

python %SCRIPT%\2D_HLD_scan_v4__prepare_solutions.py %SETTINGS%
python %SCRIPT%\2D_HLD_OIW_scan_v4__distribute_solutions.py %SETTINGS%
python %SCRIPT%\2D_HLD_OIW_scan_v4__setup_configuration.py %SETTINGS%
python %SCRIPT%\2D_HLD_OIW_scan_v4__execute.py %SETTINGS%




REM .......................................................... 3i1D_HLD_scan_v4:
================================================================================

python %SCRIPT%\3i1D_HLD_scan_v4__setup_configuration.py %SETTINGS%
python %SCRIPT%\3i1D_HLD_scan_v4__execute.py %SETTINGS%

================================================================================
================================================================================


python %SCRIPT%\pick_up_pipette_tips_bunch.py %SETTINGS%
python %SCRIPT%\update_labware_well_positions.py %SETTINGS%
python %SCRIPT%\change_cuvette_test.py %SETTINGS%

python %SCRIPT_EXTR%\extract_experimental_conditions_from_data_json.py -s %SETTINGS% ""

python %SCRIPT%\test_well_address_in_operation.py 



