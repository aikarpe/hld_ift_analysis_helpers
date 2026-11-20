% HLD scan v4 explainer
% aigars piruska
% November 03, 2026

# scans

- 3i1D_HLD_scan_v4: quick scan in 1D changing 3 dimensions simultaneously
- 2D_HLD_scan_v4: a general 2D scan, workflow for 2 surfactants, can be used for a single surfactant
- 2D_HLD_OIW_scan_v4: a specialized 2D scan for volatile inner phase

---

# Workflow in brief

- prepare a setting files
- prepare stock and dilution solutions and add their definitions to the solution repository
- prepare run stocks 
- create run configuration files 
- execute a scan 

---

# scan_settings

```
{
  "note": "a json representation of inputs for solution preparation for 2D scan, surfactant in oil; this is dummy test scan",
  "note1": "must remembet to....",
  "DATA_PATH": "C:/Users/admin/Documents/Data/aikars/opentron\\TergitolS_15_S_3n9_C7C16_NaCl_v1",
  "LOG_PATH": "C:/Users/admin/Documents/Data/aikars/opentron\\TergitolS_15_S_3n9_C7C16_NaCl_v1\\log.log",
  "CONFIG_PATH": "C:/Users/admin/Documents/Data/aikars/opentron\\TergitolS_15_S_3n9_C7C16_NaCl_v1\\config",
  "SOLUTION_REPOSITORY_PATH": "C:/Users/admin/Documents/Data/aikars/opentron\\TergitolS_15_S_3n9_C7C16_NaCl_v1\\config\\solution_repository.json",
  "c_surfactant_outer_run_solution": 5.0,
  "c_surfactant_outer_stock": 20.0,
  "c_surfactant_inner_run_solution": 15.0,
  "c_surfactant_inner_stock": 20.0,
  "stocks": {
    "inner_stock_1": "20g/100ml_Tergitol_15-S-9_in_wt_20260902",
    "inner_stock_2": "20g/100ml_Tergitol_15-S-9_15g/100ml_nacl_in_wt_20260902",
    "inner_diluter_1": "water",
    "inner_diluter_2": "15g/100ml_nacl_in_wt_20260902",
    "inner_run_solution_1": "RS15g/100ml_Tergitol_15-S-9_in_wt_20260904",
    "inner_run_solution_2": "RS15g/100ml_Tergitol_15-S-9_15g/100ml_nacl_in_wt_20260904",
    "outer_stock_1": "20g/100mL Tergitol 15-S-3 in heptane 20260824",
    "outer_stock_2": "20g/100mL Tergitol 15-S-3 in hexadecane 20260824",
    "outer_diluter_1": "heptane",
    "outer_diluter_2": "hexadecane",
    "outer_run_solution_1": "RS05g/100mL_Tergitol_15-S-3_C7_20260904",
    "outer_run_solution_2": "RS05g/100mL_Tergitol_15-S-3_C16_20260904",
    "washing_solution": "water"
  },
  "configurations": {
    "blank": "hld_ift_experiment__blank",
    "blank_logout_OIW": "",
    "blank_OIW": "",
    "start": "2026-09-04_5-15-20g_tot_Tergtls_15_S_3n9_in_C7C16",
    "end": "2026-09-04_5-15-20g_tot_Tergtls_15_S_3n9_in_C7C16_run",
    "to_reuse": {
      "name": "hld_ift_experiment__blank",
      "content": []
    }
  },
  "scan": {
    "experiment_metadata": {
      "description": "Tergitol_15_S_3 15g/100ml in oil (C7/C16), NaCl sol as aq phase",
      "needle_dia": 0.312,
      "oil": "C7 to C16",
      "measurement": "pulsed, 20 steps, 6 s pause",
      "scan_type": "2D scan: salinity, oil"
    },
    "n_expansions": 3,
    "n_approximation": 0,
    "number_of_outer_solution_points": 6,
    "volume_of_outer_solution": 3200,
    "scan_type": "linear",
    "restart_index": 0
  }
}

```

# settings parts: pathes

- `DATA_PATH`: an absolute path to this set of experiment(s)
- `LOG_PATH`: an absolute path to log file generated during a scan
- `CONFIG_PATH`: an absolut path to folder that contains configuration files (opentron, mixing graph, measurement)
- `SOLUTION_REPOSITORY_PATH`: an absolute path to a solution repository

---

# settings parts: stocks

- `c_surfactant_outer_run_solution`: a concentration of a surfactant in an outer run solution used in an experiment; float value that is less or eaqul to `c_surfactant_outer_stock`
- `c_surfactant_outer_stock`: a concentration of a surfactant in main oil stock, it is expected that both oil stocks contain the same concentration. 
- `c_surfactant_inner_run_solution`: a concentration of a surfactant in aqueous phase used in an experiment; float value that is less or eaqul to `c_surfactant_inner_stock`
- `c_surfactant_inner_stock`: a concentration of a surfactant in main aqueous stock, it is expected that both aqueous stocks contain the same concentration. 

- subsection `stocks` in settings file contains names of various solutions used in experiment. The solutions are stored in the solution repository associated with the experiment(s). The names given in this section have to match exactly names of solutions availabel in the solution repository. E.g. `heptane` and `Heptane` are two different names!
- running solutions are created from main stocks and diluter fluids. Their names can be arbitrarily chosen and definition of these solutions are created during the solution prep step (__add_script_name__).

- `outer_stock_1`: a name of main stock of an oil phase 1,
- `outer_stock_2`: a name of main stock of an oil phase 2,
- `outer_diluter_1`: a name of a diluter of an oil phase 1,
- `outer_diluter_2`: a name of a diluter of an oil phase 2,
- `outer_run_solution_1`: a name of run stock of an oil phase 1,
- `outer_run_solution_2`: a name of run stock of an oil phase 2,
- `inner_stock_1`: a name of main stock of an aqueous phase 1,
- `inner_stock_2`: a name of main stock of an aqueous phase 2,
- `inner_diluter_1`: a name of diluter fluid of an aqueous phase 1,
- `inner_diluter_2`: a name of diluter fluid of an aqueous phase 2,
- `inner_run_solution_1`: a name of run stock of an aqueous phase 1,
- `inner_run_solution_2`: a name of run stock of an aqueous phase 2,
- `washing_solution`: a name of washing and waste solution

---

# abbreviations: opentrons configuration notes

|abbreviation| notes |
|----------|--------------------------------------------------|
| `SMPL WST`| Sample waste; here remaining sample is discarded after measurement|
| `WASH1 WST`| Wash 1 waste; here a washing fluid is discarded after first set of washing steps|
| `WASH2 WST`| Wash 2 waste; here a sample fluid is discarded after washing steps with a sample fluid|
| `WASH1`| A source vial of a washing fluid|
| `IRS1`| An inner run solution 1 |
| `IRS2`| An inner run solution 2|
| `ORS1`| An outer run solution 1|
| `ORS2`| An outer run solution 2|
| `IS1`| A main inner stock 1 |
| `IS2`| A main inner stock 2|
| `OS1`| A main outer stock 1|
| `OS2`| A main outer stock 2|
| `ID1`| An inner diluter 1|
| `ID2`| An inner diluter 2|
| `OD1`| An outer diluter 1|
| `OD1`| An outer diluter 2|
| `WASH`| A location of washing fluid for distribution|


---

# Workflow in brief: 1

- prepare a setting files
- prepare stock and dilution solutions and add their definitions to the solution repository
- prepare run stocks 
- create run configuration files 
- execute a scan 

```
conda activate hld_ift0
cd C:\Users\admin\Documents\Data\aikars\opentron\TergitolS_15-S-5n15_C7C16_NaCl_v2
REM set SCRIPT="\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\scripts"
set SCRIPT="C:\Users\admin\code\hld_ift_analysis_helpers\src\hld_ift_analysis_helpers\utils\acquire"
set SETTINGS=scan_settings.json
echo %SCRIPT%

python %SCRIPT%\new_profile_wizard.py
python %SCRIPT%\solution_repository_editor.py %SETTINGS%

```

---

# Workflow in brief: 2

```

REM ................................................................. 2D_HLD_v4:
================================================================================

python %SCRIPT%\2D_HLD_scan_v4__prepare_solutions.py %SETTINGS%
python %SCRIPT%\2D_HLD_scan_v4__setup_configuration.py %SETTINGS%
python %SCRIPT%\2D_HLD_scan_v4__execute.py %SETTINGS%


```

---

# Workflow in brief: 3

```

REM .................................... 2D_HLD_OIW_v4 and 2D_HLD_OIW_logout_v4:
================================================================================

python %SCRIPT%\2D_HLD_scan_v4__prepare_solutions.py %SETTINGS%
python %SCRIPT%\2D_HLD_OIW_scan_v4__distribute_solutions.py %SETTINGS%
python %SCRIPT%\2D_HLD_OIW_scan_v4__setup_configuration.py %SETTINGS%
python %SCRIPT%\2D_HLD_OIW_scan_v4__execute.py %SETTINGS%



```

---

# Workflow in brief: 4

```

REM .......................................................... 3i1D_HLD_scan_v4:
================================================================================

python %SCRIPT%\3i1D_HLD_scan_v4__setup_configuration.py %SETTINGS%
python %SCRIPT%\3i1D_HLD_scan_v4__execute.py %SETTINGS%

```

---


### execution (3i1D_HLD_scan_v4__execute.py)


```
python %SCRIPT%\3i1D_HLD_scan_v4__execute.py %SETTINGS%

slot 2:

          A1$        A2         A3         A4         A5         A6
          SMPL WST   WASH1 WST  WASH2 WST  WASH1      empty      empty
          0.5        0.5        0.5        1.4        0          0        

          B1         B2         B3         B4         B5         B6
          empty      empty      empty      empty      empty      empty       
          0          0          0          0          0          0          
          
          C1         C2         C3         C4         C5         C6
          empty      empty      empty      empty      empty      empty       
          0          0          0          0          0          0          
          
          D1         D2         D3         D4         D5         D6
          empty      empty      empty      empty      empty      empty       
          0          0          0          0          0          0          
          
          
slot 9:
          A1         A2         A3         A4         A5
          -          -          -          -          -
          -          -          -          -          -

          B1         B2         B3         B4         B5
          OS1        OS2        IS1        IS2        -
          12         12         12         12         -
          
          C1         C2         C3         C4         C5
          OD1        OD2        ID1        ID2        -
          12         12         12         12         -

```

---

# Workflow: 2D_HLD_scan_v4, prepare solutions 

```
python %SCRIPT%\2D_HLD_scan_v4__prepare_solutions.py %SETTINGS%

slot 2:

          A1$        A2         A3         A4         A5         A6
          SMPL WST   WASH1 WST  WASH2 WST  WASH1      -          -  
          0->0.5     0->0.5     0->0.5     0->1.4     -          -        

          B1         B2         B3         B4         B5         B6
          empty      empty      empty      empty      empty      empty       
          0          0          0          0          0          0          
          
          C1         C2         C3         C4         C5         C6
          empty      empty      empty      empty      empty      empty       
          0          0          0          0          0          0          
          
          D1         D2         D3         D4         D5         D6
          empty      empty      empty      empty      empty      empty       
          0          0          0          0          0          0          
          
          
slot 9:
          A1         A2         A3         A4         A5
          ORS1       ORS2       IRS1       IRS2       -
          ->12       ->12       ->12       ->12       -

          B1         B2         B3         B4         B5
          OS1        OS2        IS1        IS2        -
          12->       12->       12->       12->       -
          
          C1         C2         C3         C4         C5
          OD1        OD2        ID1        ID2        -
          12->       12->       12->       12->       -

```

---

# Workflow: 2D_HLD_OIW_scan_v4: distribute solutions: (2D_HLD_OIW_scan_v4__distribute_solutions.py)


```
    python %SCRIPT%\2D_HLD_scan_v4__prepare_solutions.py %SETTINGS%
>>> python %SCRIPT%\2D_HLD_OIW_scan_v4__distribute_solutions.py %SETTINGS%
    python %SCRIPT%\2D_HLD_OIW_scan_v4__setup_configuration.py %SETTINGS%
    python %SCRIPT%\2D_HLD_OIW_scan_v4__execute.py %SETTINGS%

slot 9:
          A1         A2         A3         A4         A5
          -          -          IRS1       IRS2       -
          -          -          12         12         -

          B1         B2         B3         B4         B5
          -          I80/20     I60/40     I40/60     I20/80
          -          ->4        ->4        ->4        ->4
          
          C1         C2         C3         C4         C5
          -          -          -          -          -
          -          -          -          -          -

```


---


# Workflow: 2D_HLD_OIW_scan_v4: execute: (2D_HLD_OIW_scan_v4__execute.py) 

```
    python %SCRIPT%\2D_HLD_scan_v4__prepare_solutions.py %SETTINGS%
    python %SCRIPT%\2D_HLD_OIW_scan_v4__distribute_solutions.py %SETTINGS%
    python %SCRIPT%\2D_HLD_OIW_scan_v4__setup_configuration.py %SETTINGS%
>>> python %SCRIPT%\2D_HLD_OIW_scan_v4__execute.py %SETTINGS%


slot 2:

          A1$        A2         A3         A4         A5         A6
          SMPL WST   WASH1 WST  WASH2 WST  WASH1      empty      empty
          0.5        0.5        0.5        1.4        0          0        

          B1         B2         B3         B4         B5         B6
          empty      empty      empty      empty      empty      empty       
          0          0          0          0          0          0          
          
          C1         C2         C3         C4         C5         C6
          empty      empty      empty      empty      empty      empty       
          0          0          0          0          0          0          
          
          D1         D2         D3         D4         D5         D6
          IRS1       I80/20     I60/40     I40/60     I20/80     IRS2        
          1.4        1.4        1.4        1.4        1.4        1.4        
          
          
slot 9:
          A1         A2         A3         A4         A5
          ORS1       ORS2       -          -          -
          12         12         -          -          -

          B1         B2         B3         B4         B5
          -          -          -          -          -
          -          -          -          -          -
          
          C1         C2         C3         C4         C5
          -          -          -          -          -   
          -          -          -          -          -

```

--- 

# things to address:

- give scan variables [planning | during run| post processing] whenever scan is built for 3i1D hld scan!!!
- work out your example ==> to get experimenta conditions in every frame!!!
- [x] ventilation supplies to share 


```
!pandoc -t slidy -s % -o %:r.html
```

