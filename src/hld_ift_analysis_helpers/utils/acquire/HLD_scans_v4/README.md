---
title: README
---

The folder contains files required to run:

- 3i1D_HLD_scan_v4
- 2D_HLD_scan_v4,
- 2D_HLD_OIW_scan_v4

All these runs are functionally equivalent to previous runs (<give examples of previous versions>). They have been reworked to have similar inputs in scan_settings.json file and have consistent solution locations.


# context

Scans `3i1D_HLD_scan_v4` and `2D_HLD_scan_v4` were designed to characterize system with two surfactants, one oil and one water soluble; both scans use 8 stock solutions to explore system's behaviour versus salinity, EACN and surfactant concentration. `2D_HLD_scan_v4` can be used also to characterize a single surfactant versus salinity and EACN. `2D_HLD_OIW_scan_v4` is a 2D scan meant to probe volatile inner phase samples. All available runs where reworked to have more consistent terminology for the run; The inputs used in settings file are kept - at least attempted - consistent accross all runs. It is meant to allow switch from one type of scan to another in a simple manner.

A basic 2D_HLD scan requires 4 stock solutions, 

- 2 inner run solutions: these are used to be mixed in various proportions and injected into outer solution delivered into a cuvette, 
- 2 outer run solutions: these are also mixed in various proportions and delivered to a measurement cuvette
- the 2 run solutions encode one of scan variables. This could be salinity (e.g. one stock is water and the other is concentrated NaCl solution in water) or EACN (e.g. one stock is heptane and the other is hexadecane).

Most of systems have several formulation parameters and cannot be properly characterized with a single 2D_HLD scan. 8 stock solutions introduces an extra variable (surfactant concentration or ratio of surfactant concentrations) and can be used to quickly and easily mix 4 run solutions needed to run 2D_HLD scan.


# Workflow in brief

- prepare a setting files
- prepare stock and dilution solutions and add their definitions to the solution repository
- prepare run stocks 
- create run configuration files 
- execute a scan 


# Workflows

## terminology

- **inner** solution: a fluid phase that is injected in small quantities via small diameter needle into a bulk liquid.
- **outer** solution: a bulk fluid into which *inner* solution is injected.
- inner/outer **run solution**: solutions that are directly used during the scan to mix various compositions of outer and inner solutions
- inner/outer **stock**; a solution with the highest concentration of surfactant
- inner/outer **diluter**; a soluton with no surfactant but having similar composition to a stock solution (e.g stock has 10g/100mL Tergitol 15-S-15 and 15g/100mL NaCl in water; diluter has 15g/100mL NaCl in water). A mixture of a stock and a related diluter solution gives an intermediate surfactant concentration during run (e.g 3iD_HLD_scan) or run solutions (e.g. 2D_HLD_scan)
- **upward** scan: a scan where an inner phase has lower density than an outer phase; such choice will a) result in a movement of the injected inner fluid upwards, to a top of a cuvette move upward and b) necessitates hook-shaped [J] needle
- **downward** scan: a scan were an inner phase has higher density than an outer phase; such choice will result in a movement of the injected fluid downwards, to a bottom of a cuvette and necessitates a straight [I] needle.


## abbreviations

- `SMPL WST`: Sample waste; here remaining sample is discarded after measurement.
- `WASH1 WST`: Wash 1 waste; here a washing fluid is discarded after first set of washing steps.
- `WASH2 WST`: Wash 2 waste; here a sample fluid is discarded after washing steps with a sample fluid
- `WASH1`: A source vial of a washing fluid
- `IRS1`: An inner run solution 1; 
- `IRS2`: An inner run solution 2;
- `ORS1`: An outer run solution 1;
- `ORS2`: An outer run solution 2;
- `IS1`: A main inner stock 1; 
- `IS2`: A main inner stock 2;
- `OS1`: A main outer stock 1;
- `OS2`: A main outer stock 2;
- `ID1`: An inner diluter 1;
- `ID2`: An inner diluter 2;
- `OD1`: An outer diluter 1;
- `OD1`: An outer diluter 2;
- `WASH`: A location of washing fluid for distribution.


## Workflow: 3i1D_HLD_scan_v4

### setting up scan_settings.json

- The scan requires:
    - inner and outer stock solutions (4 in total)
    - inner and outer diluters (4 in total)

### solution preparation and distribution

- a user is required:
    - to make stock and diluter solutions and enter their definitions into a solution repository
    - to prepare manually opentron configuration given in the section `execution`:
        - fill waste and washing solutons;
        - fill stock and diluter solutions (each 12 mL)


### set up configuration (3i1D_HLD_scan_v4__setup_configuration.py)

### execution (3i1D_HLD_scan_v4__execute.py)


```
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


## Workflow: 2D_HLD_scan_v4

### setting up scan_settings.json

### prepare solutions (2D_HLD_scan_v4__prepare_solutions.py)

```
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

## Workflow: 2D_HLD_OIW_scan_v4

### setting up scan_settings.json

### prepare solutions ()

### distribute solutions: (2D_HLD_OIW_scan_v4__distribute_solutions.py)


```
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

### setup configuration (2D_HLD_OIW_scan_v4__setup_configuration.py)

### execute: (2D_HLD_OIW_scan_v4__execute.py) 

```
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


The scan assumes that a tested system contains two surfactants, one oil and one water soluble.

Setup files assumes that user has:

- 4 main stock solutions:
    - 2 stocks for oil phase, 
    - 2 stocks for aqueous phase. 
- 4 dilution liquids:
    - 2 dilution oils - one for each oil stock
    - 2 dilution liquids for aqueous phase - one for each stock
- surfactant concentration values are used to calculate dilution to obtain run stocks


# Workflow in brief

- prepare setting files
- prepare stock and dilution solutions and add their definitions to solution repository
- prepare run stocks (using `2D_HLD_scan_v3__prepare_solutions.py`)
- create run configuration files (using `2D_HLD_scan_v3__setup_configuration.py`)
- execute 2D HLD scan (using `2D_HLD_scan_v3__execute.py`)

# abbreviations

- `SMPL WST`: Sample waste; here remaining sample is discarded after measurement.
- `WASH1 WST`: Wash 1 waste; here a washing fluid is discarded after first set of washing steps.
- `WASH2 WST`: Wash 2 waste; here a sample fluid is discarded after washing steps with a sample fluid
- `WASH1`: A washing fluid source vial
- `IRS1`: An inner run solution 1; 
- `IRS2`: An inner run solution 2;
- `ORS1`: An outer run solution 1;
- `ORS2`: An outer run solution 2;
- `IS1`: A main inner stock 1; 
- `IS2`: A main inner stock 2;
- `OS1`: A main outer stock 1;
- `OS2`: A main outer stock 2;
- `ID1`: An inner diluter 1;
- `ID2`: An inner diluter 2;
- `OD1`: An outer diluter 1;
- `OD1`: An outer diluter 2;
- `WASH`: A location of washing fluid for distribution;

# settings file

- A setting file contains inputs necessary to prepare run stocks, generate configuration files and execute experimental scan

## general parameters

- `DATA_PATH`: an absolute path to this set of experiment(s)
- `LOG_PATH`: an absolute path to log file generated during a scan
- `CONFIG_PATH`: an absolut path to folder that contains configuration files (opentron, mixing graph, measurement)
- `SOLUTION_REPOSITORY_PATH`: an absolute path to a solution repository
- `c_surfactant_outer_run_solution`: a concentration of a surfactant in an outer run solution used in an experiment; float value that is less or eaqul to `c_surfactant_outer_stock`
- `c_surfactant_outer_stock`: a concentration of a surfactant in main oil stock, it is expected that both oil stocks contain the same concentration. 
- `c_surfactant_inner_run_solution`: a concentration of a surfactant in aqueous phase used in an experiment; float value that is less or eaqul to `c_surfactant_inner_stock`
- `c_surfactant_inner_stock`: a concentration of a surfactant in main aqueous stock, it is expected that both aqueous stocks contain the same concentration. 
    - The necessary dilution is estimated from c_surfactant_stock and c_surfactant_experiment values. A main stock is diluted with a diluter solution: 
        - v_stock = c_surfactant_experiment / c_surfactant_stock * v_total
        - v_diluter = v_total - v_stock

## Stocks

- subsection `stocks` in settings file contains names of various solutions used in experiment. The solutions are stored in the solution repository associated with the experiment(s). The names given in this section have to match exactly names of solutions availabel in the solution repository. E.g. `heptane` and `Heptane` are two different names!
- running solutions are created from main stocks and diluter fluids. Their names can be arbitrarily chosen and definition of these solutions are created during the solution prep step (__add_script_name__).

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

## Configurations

- this section contains labels to derive names of configurations files.
- `blank`: a generic name of blank configuration files used as template
- `start`: a label of start configuration
- `end`: a label of end configuration
- `to_reuse`: a label of configuration to be reused; handy in case when some intermediate solutions are prepared in previous runs and can be reused
- `blank_logout_OIW`: a generic name of blank configuration files used as template for OIW scan with logarithmic spacing in outer solution composition.
- `blank_OIW`:  a generic name of blank configuration files used as template for OIW scan with linear spacing in outer solution composition.

## Scan

- `experiment_metadata`: string entries of user defined metadata; these fields are added as header variable into an experimental data 
- `n_expansions": integer, number of times new data points are added to scan. The manner in which the data points are added depends on scan type:
    - `scan_type == "linear"`: a new data point is added at the midpoint between any two nearest neighbours
        - every expansion `n-1` data point is added, where `n` is number of data points in the scan before expansion
        - total number of points after `m-th` expansion is `2^m + 1`
        

         | total number of expansions | data points in a scan                                         |
         |----------------------------|---------------------------------------------------------------|
         | 0                          | {0, 1}*                                                       |
         | 1                          | {0, 0.5, 1}                                                   |
         | 2                          | {0, 0.25, 0.5, 0.75, 1}                                       |
         | 3                          | {0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1}           |
         | m                          | {0/2^m, 1/2^m, 2/2^m, ..., (2^m - 1)/2^m, 2^m/2^m}            |


         (*) - numbers given are relative concentrations, where stock 1 is considered to have concentration `0` and stock 2 - `1`. 


    - `scan_type == "log" `: a single new data point is added between stock 1 and the lowest dilution of stock 2
        - total number of points after `m-th` expansion is `m+1`

         | total number of expansions | data points in a scan                                         |
         |----------------------------|---------------------------------------------------------------|
         | 0                          | {0, 1}**                                                      |
         | 1                          | {0, 0.5, 1}                                                   |
         | 2                          | {0, 0.25, 0.5, 1}                                             |
         | 3                          | {0, 0.125, 0.25, 0.5, 1}                                      |
         | m                          | {0, 1/2^(m-0), 1/2^(m-1), 1/2^(m-2), ..., 1/2^1, 1/2^0}       |
        
         (**) point {0} is never measured

    - `scan_type == "log_and_mid"`: two new data points are added.
        - concentration_new_1: (c_0 + c_1) / 2
        - concentration_new_2: (concentration_new_1 + c_1) / 2
        - c_0 and c_1 are points with two lowest concentrations in a scan
        - total number of points after `m` expansions is `(m+1) * 2`

         | total number of expansions | data points in a scan                                             |
         |----------------------------|-------------------------------------------------------------------|
         | 0                          | {0, 1}***                                                         |
         | 1                          | {0, 0.5, 0.75, 1}                                                 |
         | 2                          | {0, 0.25, 0.375, 0.5, 0.75, 1}                                    |
         | 3                          | {0, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1}                     |
         | m                          | {0, 2/2^(m+1), 3/2^(m+1), 2/2^m, 3/2^m, ..., 2/2^2, 3/2^2, 2/2^1} |
        
         

- `n_approximation`: integer, not used at the moment, leave value at `0`
- `number_of_outer_solution_points`: integer, number of different outer solution blends. outer solution is linearly blended from two stocks.
- `volume_of_outer_solution`: float, volume of outer solution in uL added to a cuvette; suggested value: 3000 to 3500
- `scan_type`: a string from set {"linear", "log", "log_and_mid"}, unrecognized string defaults to "linear"



# opentron starting conditions

## run stock preparation: layout

```
slot 2:

          A1$        A2         A3         A4         A5         A6
          SMPL WST   WASH1 WST  WASH2 WST  WASH1      -          -  
          0->0.5     0->0.5     0->0.5     0->1.4     -          -        

          B1         B2         B3         B4         B5         B6
          -          -          -          -          -          -           
          -          -          -          -          -          -          
          
          C1         C2         C3         C4         C5         C6
          -          -          -          -          -          -           
          -          -          -          -          -          -          
          
          D1         D2         D3         D4         D5         D6
          -          -          -          -          -          -           
          -          -          -          -          -          -          
          
          
slot 9:
          A1         A2         A3         A4         A5
          RAQ1       RAQ2       RO1        RO2        -
          0->12.5    0->12.5    0->12.5    0->12.5    -

          B1         B2         B3         B4         B5
          SAQ1       SAQ2       SO1        SO2        -
          12.5->     12.5->     12.5->     12.5->     -
          
          C1         C2         C3         C4         C5
          DAQ1       DAQ2       DO1        DO2        WASH
          12.5->     12.5->     12.5->     12.5->     12.5->

```

($) - each well is described with 1) address label, 2) role, 3) volume change; `-` indicates not used or does not change; `X->` indicates that starting volume is known, but final volume varies based on inputs


## scan execution: layout

```
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
          RAQ1       RAQ2       RO1        RO2        -
          12         12         12         12         -

          B1         B2         B3         B4         B5
          -          -          -          -          -
          -          -          -          -          -
          
          C1         C2         C3         C4         C5
          -          -          -          -          -
          -          -          -          -          -

```

($) - each well is described with 1) address label, 2) role, 3) volume; `-` well not used; `empty` - an empty eppendorff to mix inner solutions

Setup requires:



- surfactant_in_oil_1[^note_1]: main stock of oil_1 with surfactant_oil
- surfactant_in_oil_2: main stock of oil_2 with surfactant_oil
- stock_aqueous_1: main stock of aqueous_solution_1 with surfactant_aq
- stock_aqueous_2: main stock of aqueous_solution_2 with surfactant_aq

[^note_1]: [variable name in `scan_settings.json`]: [description]


