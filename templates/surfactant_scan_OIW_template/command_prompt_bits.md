REM ######################################################################
REM #                                                                    # 
REM # Instructions to start executing hld_ift scan(s) and their helper   #
REM #          scripts                                                   #
REM #                                                                    # 
REM ######################################################################

REM ######################################################################
REM #                                                                    # 
REM #  1. turn on LED light source                                       # 
REM #                                                                    # 
REM #  2. start pylon viewer and initialize camera                       #
REM #  3. configure image format options                                 # 
REM #     - width:  1696                                                 # 
REM #     - height: 1500                                                 # 
REM #     - centered X: yes                                              # 
REM #     - centered Y: yes                                              # 
REM #  4. start acquisition (in pylon viewer) make sure that observed    #
REM #       image has:                                                   #
REM #     - uniform light intensity                                      # 
REM #     - counts are below 255 (230 - 240)                             # 
REM #  5. position cuvette in the center of an image!!! (no walls visible#
REM #  6. stop acquisition and free camera                               #
REM #                                                                    # 
REM #  7. start anaconda prompt and run commands below in it !!!         # 
REM #                                                                    # 
REM ######################################################################

REM ######################################################################
REM #                                                                    # 
REM #  IF YOU WANT TO UPDATE TO NEW HLD_IFT_HTTP VERSION:                # 
REM #                                                                    # 
REM #  1. open git bash terminal                                         # 
REM #  2. navigate to hld_ift_http source folder                         # 
REM #     cd ~/code/hld_ift_http                                         # 
REM #     git pull origin reworked_measurements                          #
REM #                                                                    # 
REM #  3. in anaconda prompt                                             # 
REM #     pip install C:\Users\Admin\code\hld_ift_http                   # 
REM #                                                                    # 
REM ######################################################################


REM ######################################################################
REM #                                                                    # 
REM # commands to execute only once for each session                     # 
REM #                                                                    # 
REM ######################################################################

conda activate hld_ift0
cd C:\Users\admin\Documents\Data\aikars\opentron\surfactant_scan_OIW_template
set SCRIPT="\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\scripts"
set SETTINGS=scan_settings.json
echo %SCRIPT%
echo %SETTINGS%

REM ######################################################################
REM #                                                                    # 
REM # commands to accomplish certain task(s)                             # 
REM #                                                                    # 
REM ######################################################################


#------------------------------
python %SCRIPT%\generate_recepies_for_solutions_v2.py  recipes_solutions.json
python %SCRIPT%\solution_repository__edit.py recipes_solutions.json

#------------------------------
echo %SCRIPT%
python %SCRIPT%\2D_HLD_scan_v2__prepare_solutions_OIW.py %SETTINGS%

python %SCRIPT%\2D_HLD_scan_v2__setup_configuration_OIW.py  %SETTINGS%


python %SCRIPT%\2D_HLD_scan_v2__execute_OIW.py  %SETTINGS%

#python %SCRIPT%\2D_HLD_scan_v2__measurement_only_OIW.py  %SETTINGS%
