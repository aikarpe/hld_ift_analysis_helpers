---
title: README
---

The folder contains files required to run 2D_HLD_scan_v3

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
- `RAQ1`: A run stock of aqueous phase 1; 
- `RAQ2`: A run stock of aqueous phase 2;
- `RO1`: A run stock of oil phase 1;
- `RO2`: A run stock of oil phase 2;
- `SAQ1`: A main stock of aqueous phase 1;
- `SAQ2`: A main stock of aqueous phase 2;
- `SO1`: A main stock of oil phase 1; 
- `SO2`: A main stock of oil phase 2;
- `DAQ1`: A dilution fluid for aqueous phase 1;
- `DAQ2`: A dilution fluid for aqueous phase 2;
- `DO1`: A dilution fluid for oil phase 1;
- `DO2`: A dilution fluid for oil phase 2;
- `WASH`: A location of washing fluid for distribution;

# settings file

- A setting file contains inputs necessary to prepare run stocks, generate configuration files and execute experimental scan

## general parameters

- `DATA_PATH`: an absolute path to this set of experiment(s)
- `LOG_PATH`: an absolute path to log file generated during a scan
- `CONFIG_PATH`: an absolut path to folder that contains configuration files (opentron, mixing graph, measurement)
- `SOLUTION_REPOSITORY_PATH`: an absolute path to solution repository
- `c_surfactant_experiment`: a concentration of a surfactant in oil used in an experiment; float value that is less or eaqul to `c_surfactant_stock`
- `c_surfactant_stock`: a concentration of a surfactant in main oil stock, it is expected that both oil stocks contain the same concentration. 
- `c_surfactant_aq_experiment`: a concentration of a surfactant in aqueous phase used in an experiment; float value that is less or eaqul to `c_surfactant_aq_stock`
- `c_surfactant_aq_stock`: a concentration of a surfactant in main aqueous stock, it is expected that both aqueous stocks contain the same concentration. 
    - The necessary dilution is estimated from c_surfactant_stock and c_surfactant_experiment values. A main stock is diluted with a diluter solution: 
        - v_stock = c_surfactant_experiment / c_surfactant_stock * v_total
        - v_diluter = v_total - v_stock

## Stocks

- subsection `stocks` in settings file contains names of various solutions used in experiment. The solutions are stored in the solution repository associated with the experiment(s). The names given in this section have to match exactly names of solutions availabel in the solution repository. E.g. `heptane` and `Heptane` are two different names!
- running solutions are created from main stocks and diluter fluids. Their names can be arbitrarily chosen and definition of these solutions are created during the solution prep step (__add_script_name__).
- `surfactant_in_oil_1`: a name of main stock of an oil phase 1,
- `surfactant_in_oil_2`: a name of main stock of an oil phase 2,
- `oil_1`: a name of a diluter of an oil phase 1,
- `oil_2`: a name of a diluter of an oil phase 2,
- `stock_aqueous_1`: a name of main stock of an aqueous phase 1,
- `stock_aqueous_2`: a name of main stock of an aqueous phase 2,
- `run_stock_aqueous_1`: a name of run stock of an aqueous phase 1,
- `run_stock_aqueous_2`: a name of run stock of an aqueous phase 2,
- `diluter_aqueous_1`: a name of diluter fluid of an aqueous phase 1,
- `diluter_aqueous_2`: a name of diluter fluid of an aqueous phase 2,
- `run_stock_surf_oil_1`: a name of run stock of an oil phase 1,
- `run_stock_surf_oil_2`: a name of run stock of an oil phase 2,

## Configurations

- this section contains labels to derive names of configurations files.
- `blank`: a generic name of blank configuration files used as template
- `start`: a label of start configuration
- `end`: a label of end configuration
- `to_reuse`: a label of configuration to be reused; handy in case when some intermediate solutions are prepared in previous runs and can be reused

## Scan

- `experiment_metadata`: string entries of user defined metadata; these fields are added into experimental data 
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
- `number_of_oil_points": integer, number of different outer solution blends. outer solution is linearly blended from two stocks.
- `oil_volume": float, volume of outer solution in uL added to a cuvette
- `scan_type": a string from set {"linear", "log", "log_and_mid"}, unrecognized string defaults to "linear"





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

Setup requires:



- surfactant_in_oil_1[^note_1]: main stock of oil_1 with surfactant_oil
- surfactant_in_oil_2: main stock of oil_2 with surfactant_oil
- stock_aqueous_1: main stock of aqueous_solution_1 with surfactant_aq
- stock_aqueous_2: main stock of aqueous_solution_2 with surfactant_aq

[^note_1]: [variable name in `scan_settings.json`]: [description]
