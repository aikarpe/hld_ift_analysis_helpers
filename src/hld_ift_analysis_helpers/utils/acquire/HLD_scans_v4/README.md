---
title: 'README'
date: 2026-11-06
---

The folder contains files required to run:

- 3i1D_HLD_scan_v4
- 2D_HLD_scan_v4,
- 2D_HLD_OIW_scan_v4

All these runs are functionally equivalent to previous runs (<give examples of previous versions>). They have been reworked to have similar inputs in scan_settings.json file and have consistent solution locations.


# context

Hld scans are designed to explore IFT as a function of formulation parameters. A single measurement is a controlled injection of an inner solution into a larger amount (~3mL) of an outer solution. Every measurement gives qualitative assesment of IFT magnitude. At large IFT values, there is a distinct droplet formation; the larger IFT, the larger drop volume is observed. At lof IFT magnitudes, there is a stream of an inner solution observed. In addition to injection dynamics there are other interactions that can lead to gel (rag) phase formations that somewhat complicates qualitative IFT assessment. 

Hld scans are designed to capture influence of 2 parameters - salinity and EACN. For most systems a single two dimensional scan is not sufficient to characterize the system and multiple scans are necessary to properly analyse it. This package (hld_ift_analysis_helpers) contain scripts to facilitate this analysis process.

The scripts included at package/utils/acquire represent individual steps in the workflow. Parameters needed to run these scripts are collected in scan_settings.json file


## workflow in brief

- general note: The most important part of workflow is to execute hld scans. The rest of steps are meant to facilitate preparation of hld scans in systematic and uniform manner as much as possible.

- typical work flow includes following steps:
    - a preparation of a scan_setting file
    - a preparation of input solutions - stocks and dilution - and an addition of their definitions to a solution repository file
    - a preparation of run solutions 
    - a creation of configuration files - opentron configuration, mixing graph, measurement configuration
    - an execution of an hld scan 


--- 

# scans

## terminology

- **inner** solution: a fluid phase that is injected in small quantities via small diameter needle into a bulk liquid.
- **outer** solution: a bulk fluid into which *inner* solution is injected.
- inner/outer **run solution**: solutions that are directly used during the scan to mix various compositions of outer and inner solutions
- inner/outer **stock**; a solution with the highest concentration of surfactant
- inner/outer **diluter**; a soluton with no surfactant but having similar composition to a stock solution (e.g stock has 10g/100mL Tergitol 15-S-15 and 15g/100mL NaCl in water; diluter has 15g/100mL NaCl in water). A mixture of a stock and a related diluter solution gives an intermediate surfactant concentration during run (e.g 3iD_HLD_scan) or run solutions (e.g. 2D_HLD_scan)
- **upward** scan: a scan where an inner phase has lower density than an outer phase; such choice will a) result in a movement of the injected inner fluid upwards, to a top of a cuvette move upward and b) necessitates hook-shaped [J] needle
- **downward** scan: a scan were an inner phase has higher density than an outer phase; such choice will result in a movement of the injected fluid downwards, to a bottom of a cuvette and necessitates a straight [I] needle.

---

## 3i1D_HLD_scan_v4

- `3i1D_HLD_scan_v4` is designed to simultanously vary 3 distinct variables - salinity, EACN, and oil vs aqueous surfactant ration - and locate a parameter space with low IFT for further characterization.
- The scan requires 8 input solutions, their role is given in a table below
- A single variable `x` is changed in discrete fashion from 0 to 1 during a scan.
    - `x` values: {0, 1/N, 2/N, ..., N/N}, where `N = number_of_outer_solution_points - 1` 
    - `c_surfactant_outer_run_solution` and `c_surfactant_outer_stock` determine dilutions at evey sampled point; the volume fraction of each particular input solution in an inner and an outer solution is given below:

     | source solution | fraction                           | target solution |
     |-----------------|------------------------------------|-----------------|
     | inner stock 1   |   `(1-x) * x_dilute * (1-x)`       | inner phase     |
     | inner diluter 1 |   `(1-x) * (1 - x_dilute * (1-x))` | inner phase     |
     | inner stock 2   |   `x     * x_dilute * (1-x)`       | inner phase     |
     | inner diluter 2 |   `x     * (1 - x_dilute * (1-x))` | inner phase     |
     |                 |                                    |                 |
     | outer stock 1   |   `(1-x) * x_dilute * x`           | outer phase     |
     | outer diluter 1 |   `(1-x) * (1 - x_dilute * x)`     | outer phase     |
     | outer stock 2   |   `x     * x_dilute * x`           | outer phase     |
     | outer diluter 2 |   `x     * (1 - x_dilute * x)`     | outer phase     |
 
    - `x_dilute = c_surfactant_outer_run_solution / c_surfactant_outer_stock` 
    - few examples of dilutions are shown below for `x_dilute = {0.1, 0.5, 1.0}`


![](scan_fractions.jpg){width=100%}


## 2D_HLD_scan_v4

- `2D_HLD_scan_v4` is designed to create 2D grid scans for 2 distinct variables - salinity and EACN. It is intended to test systems with two surfactants (one in inner and another in outer solution) and each surfactant is independently controlled by concentration variables (see 2D_HLD_scan_v4, solution preparation section below).
- The scan requires 4 input solutions: 
    - 2 inner run solutions, and
    - 2 outer run solutions
- 2 inner run solutions are mixed in different fractions to create scan along one variable (e.g. salinity) and 2 outer solutions - along another (e.g. EACN).
- During a scan one of outer solution compositions is mixed within a cuvette and all compositions of an inner solution are tested against it; Then a new composition of an outer solution is prepared and testing against all inner solutions performed. The process is continued untill all outer compositions have been tested against all inner solution compositions
- A composition of an inner phase can be varied linerly or logarithmicly; a composition of an outer phase can be varied linearly

## 2D_HLD_OIW_scan_v4

- `2D_HLD_OIW_scan_v4` is a speciallized version of a `2D_HLD_scan_v4` for a volatile inner solution samples.
- Following limitations apply to this scan:
    - only 6 inner solution samples are possible
    - whole process is split into 3 scans, testing 2 inner solutions against all combinations of outer solution compositions
- a composition of an outer phase can be varied lineraly or logarithmicly.


---



# Workflows

# solution preparation

- a solution definition requires:
    - density, g/ml
    - composition of all ingredients as mass fraction (w/w)
- there is a tool to plan solution preparation `solution_repository_editor.py`
- weighing all components befor mixing is easiest way to create solutions (even liquids)
- for a density estimate small aliquots can be weighted and compared with standard liquid with known density (e.g. water)

# new experiment from an old (template) experiment

- use the script `create_new_profile.py` to create a new experiment folder from an old (template) experiment
- edit solution repository with the script `solution_repository_editor.py`

---

## Workflow: 3i1D_HLD_scan_v4

### planning script

- this script is meant only for planning; it does not affect how it is performed!!!
- you can see how different experimental variables change over the 3i1D_scan_v4

```
... more explanation on code should come here...
... the bit below is war code that needs reworking...

import numpy as np
import pandas as pd
from itertools import product


def plan_a_3i1D_scan_solution(
    stock_param_lst,
    fns,
    n_points=6,
    x_diluter=1.0
):
    """
    Generate a 1D scan of component concentrations.

    Parameters
    ----------
    stock_param_lst : dict
        Dictionary containing two stock parameter lists.
    fns : dict
        Dictionary containing the concentration functions.
    n_points : int
        Number of scan points.
    x_diluter : float
        Dilution factor.

    Returns
    -------
    pandas.DataFrame
        Component concentrations across the scan.
    """

    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    # Equivalent to expand.grid() in R
    def my_expand(lst):
        keys = list(lst.keys())[:2]
        values = [lst[k] for k in keys]

        # R expand.grid: first variable changes fastest
        combinations = [
            tuple(reversed(item))
            for item in product(*reversed(values))
        ]

        return pd.DataFrame(combinations, columns=keys)

    # Calculate the concentration of a component
    def component_conc_fn(var, fns):
        def calculate(x, x_diluter):
            if len(var) != len(fns):
                raise ValueError(
                    "The number of stock combinations must match "
                    "the number of concentration functions."
                )

            return sum(
                stock * fn(x, x_diluter)
                for stock, fn in zip(var, fns.values())
            )

        return calculate

    # Expand stock parameter combinations
    df_stocks = my_expand(stock_param_lst)

    component_names = df_stocks.columns

    # Generate scan points
    scan_points = np.linspace(0, 1, n_points)

    # Calculate concentrations for each component
    result = {}

    for name in component_names:

        var = df_stocks[name].tolist()

        fn = component_conc_fn(var, fns)

        result[name] = [
            fn(x, x_diluter)
            for x in scan_points
        ]

    return pd.DataFrame(result)


def plan_3i1D_scan(
    inner_solution_lst,
    outer_solution_lst,
    n_points=6,
    x_diluter=1.0
):
    """
    Generate inner and outer solution concentration scans.

    Returns
    -------
    dict
        Contains inner_solution and outer_solution DataFrames.
    """

    # Inner solution functions
    inner_sol_fns = {
        "inner_diluter_1": lambda x, xd: (1-x) * (1-xd*(1-x)),
        "inner_stock_1":   lambda x, xd: (1-x) * xd * (1-x),
        "inner_diluter_2": lambda x, xd: x * (1-xd*(1-x)),
        "inner_stock_2":   lambda x, xd: x * xd * (1-x)
    }

    # Outer solution functions
    outer_sol_fns = {
        "outer_diluter_1": lambda x, xd: (1-x) * (1-xd*x),
        "outer_stock_1":   lambda x, xd: (1-x) * xd * x,
        "outer_diluter_2": lambda x, xd: x * (1-xd*x),
        "outer_stock_2":   lambda x, xd: x * xd * x
    }

    # Calculate inner solution
    inner_solution = plan_a_3i1D_scan_solution(
        inner_solution_lst,
        inner_sol_fns,
        n_points,
        x_diluter
    )

    # Calculate outer solution
    outer_solution = plan_a_3i1D_scan_solution(
        outer_solution_lst,
        outer_sol_fns,
        n_points,
        x_diluter
    )

    # Combine both solutions (equivalent to cbind)
    combined = pd.concat(
        [inner_solution, outer_solution],
        axis=1
    )

    # Display results (equivalent to knitr::kable)
    print(combined.to_string(index=False))

    # Return results
    out = {
        "inner_solution": inner_solution,
        "outer_solution": outer_solution
    }

    return out


```

This could be run to show variable change over the scan, see example below

| solution        | name and composition                                                  |                           
|-----------------|-----------------------------------------------------------------------|                           
| inner stock 1   |  "10g/100mL Tergitol 15S5 in heptane"                                 |                           
| inner stock 2   |  "10g/100mL Tergitol 15S5 in hexadecane"                              |                              
| inner diluter 1 |  "heptane"                                                            | 
| inner diluter 2 |  "hexadecane"                                                         |    
| outer stock 1   |  "10g/100mL Tergitol 15S9 in water"                                   |                         
| outer stock 2   |  "10g/100mL Tergitol 15S9, 15g/100mL NaCl in water"                   |                       
| outer diluter 1 |  "water"                                                              |         
| outer diluter 2 |  "15g/100ml NaCl in water"                                            |                       

| solution | variable  | range  |
|----------|-----------|--------|
| inner    | cTerg15S5 | [0,10] |
| inner    | EACN      | [7,16] |
| outer    | cTerg15S9 | [0,10] |
| outer    | S         | [0,15] |



```
{python}
inner = dict(cTerg15S5 = [0,10], EACN = [7,16])
outer = dict(cTerg15S9 = [0,10], S    = [0,15])

plan_3i1D_scan(inner, outer, 6, 1.0)
```

```
> cTerg15S5  EACN  cTerg15S9    S
>      10.0   7.0        0.0  0.0
>       8.0   8.8        2.0  3.0
>       6.0  10.6        4.0  6.0
>       4.0  12.4        6.0  9.0
>       2.0  14.2        8.0 12.0
>       0.0  16.0       10.0 15.0
>
>{
>'inner_solution':    cTerg15S5  EACN
>                  0       10.0   7.0
>                  1        8.0   8.8
>                  2        6.0  10.6
>                  3        4.0  12.4
>                  4        2.0  14.2
>                  5        0.0  16.0, 
>'outer_solution':    cTerg15S9     S
>                  0        0.0   0.0
>                  1        2.0   3.0
>                  2        4.0   6.0
>                  3        6.0   9.0
>                  4        8.0  12.0
>                  5       10.0  15.0
>}
```

```
!!! should add on effect of swapping solutions on a scan

`inner = dict(cTerg15S5 = [0,10], EACN = [**7,16**])`
replace with: 
`inner = dict(cTerg15S5 = [0,10], EACN = [**16,7**])`

```

### in brief

- edit scan_settings file
- prepare all solutions and add them to a solution repository
- fill input solution according to an opentron configuration
- create configuration files
- execute run

related commands:

```
conda activate hld_ift0
cd __EXPERIMENT_PATH__
set SCRIPT=__SCRIPT_PATH__
set SETTINGS=scan_settings.json

python %SCRIPT%\new_profile_wizard.py
python %SCRIPT%\solution_repository_editor.py %SETTINGS%

python %SCRIPT%\3i1D_HLD_scan_v4__setup_configuration.py %SETTINGS%
python %SCRIPT%\3i1D_HLD_scan_v4__execute.py %SETTINGS%
```

### setting up scan_settings.json

- The scan requires:
    - inner and outer stock solutions (4 in total)
    - inner and outer diluters (4 in total)
    - washing solution

### solution preparation and distribution

- a user is required:
    - to make stock and diluter solutions and enter their definitions into a solution repository
    - to manually distribute input solutions according to an copentron configuration given in the section `execution`:
        - fill waste and washing solutons;
        - fill stock and diluter solutions (each 12 mL)

### set up configuration (3i1D_HLD_scan_v4__setup_configuration.py)

- create configuration by running the script `3i1D_HLD_scan_v4__setup_configuration.py`

- the aforementioned script supplements blank configuration in the following way:
    - slot 2: `eppendorf_24_reservoir_1500ul`
        - `2/A1`: `sample waste`, 500 uL of washing solution
        - `2/A2`: `wash waste 1`, 500 uL of washing solution
        - `2/A3`: `wash waste 2`, 500 uL of washing solution
        - `2/A4`: `washing solution 1`, 1400 uL of washing solution
    - slot 9: `3dprinted_15_tuberack_15000ul`
        - `9/B1`: `outer stock 1`, 12000 uL
        - `9/B2`: `outer stock 2`, 12000 uL
        - `9/B3`: `inner stock 1`, 12000 uL
        - `9/B4`: `inner stock 2`, 12000 uL
        - `9/C1`: `outer diluter 1`, 12000 uL
        - `9/C2`: `outer diluter 2`, 12000 uL
        - `9/C3`: `inner diluter 1`, 12000 uL
        - `9/C4`: `inner diluter 2`, 12000 uL


### execution (3i1D_HLD_scan_v4__execute.py)

- run the script `3i1D_HLD_scan_v4__execute.py`
- the follow solutions configuration is expected at the beginning of the scan:

 ![](3i1D_scan_v4__execute__slot2.png)
 ![](3i1D_scan_v4__execute__slot9.png)

- A single variable `x` is changed in discrete fashion from 0 to 1 during a scan.
    - `x` values: {0, 1/N, 2/N, ..., N/N}, where `N = number_of_outer_solution_points - 1` 
    - `c_surfactant_outer_run_solution` and `c_surfactant_outer_stock` determine dilutions at evey sampled point; the volume fraction of each particular input solution in an inner and an outer solution is given below:

     | source solution | fraction                           | target solution |
     |-----------------|------------------------------------|-----------------|
     | inner stock 1   |   `(1-x) * x_dilute * (1-x)`       | inner phase     |
     | inner diluter 1 |   `(1-x) * (1 - x_dilute * (1-x))` | inner phase     |
     | inner stock 2   |   `x     * x_dilute * (1-x)`       | inner phase     |
     | inner diluter 2 |   `x     * (1 - x_dilute * (1-x))` | inner phase     |
     |                 |                                    |                 |
     | outer stock 1   |   `(1-x) * x_dilute * x`           | outer phase     |
     | outer diluter 1 |   `(1-x) * (1 - x_dilute * x)`     | outer phase     |
     | outer stock 2   |   `x     * x_dilute * x`           | outer phase     |
     | outer diluter 2 |   `x     * (1 - x_dilute * x)`     | outer phase     |
 
    - `x_dilute = c_surfactant_outer_run_solution / c_surfactant_outer_stock` 
    - few examples of dilutions are shown below for `x_dilute = {0.1, 0.5, 1.0}`



---

## Workflow: 2D_HLD_scan_v4

### in brief

- edit scan_settings file
- prepare all solutions and add them to a solution repository
- prepare run solutions and distribute washing solution
- create configuration files
- execute run

related commands:

```
conda activate hld_ift0
cd __EXPERIMENT_PATH__
set SCRIPT=__SCRIPT_PATH__
set SETTINGS=scan_settings.json

python %SCRIPT%\new_profile_wizard.py
python %SCRIPT%\solution_repository_editor.py %SETTINGS%

python %SCRIPT%\2D_HLD_scan_v4__prepare_solutions.py %SETTINGS%
python %SCRIPT%\2D_HLD_scan_v4__setup_configuration.py %SETTINGS%
python %SCRIPT%\2D_HLD_scan_v4__execute.py %SETTINGS%
```

### setting up scan_settings.json

- requires:
    - inner and outer stocks and diluters should be valid solutions in the solution repository
    - valid and unique names for inner and outer run solutions

### prepare solutions (2D_HLD_scan_v4__prepare_solutions.py)

- run the script `2D_HLD_scan_v4__prepare_solutions.py`
- inputs: 
    - slot 9: `3dprinted_15_tuberack_15000ul`
        - `9/B1`: `outer stock 1`, 12000 uL
        - `9/B2`: `outer stock 2`, 12000 uL
        - `9/B3`: `inner stock 1`, 12000 uL
        - `9/B4`: `inner stock 2`, 12000 uL
        - `9/C1`: `outer diluter 1`, 12000 uL
        - `9/C2`: `outer diluter 2`, 12000 uL
        - `9/C3`: `inner diluter 1`, 12000 uL
        - `9/C4`: `inner diluter 2`, 12000 uL
        - `9/C5`: `washing solution`, 12000 uL

- outputs: 
    - slot 2: `eppendorf_24_reservoir_1500ul`
        - `2/A1`: `sample waste`, 500 uL
        - `2/A2`: `wash waste 1`, 500 uL
        - `2/A3`: `wash waste 2`, 500 uL
        - `2/A4`: `washing solution 1`, 1400 uL
    - slot 9: `3dprinted_15_tuberack_15000ul`
        - `9/A1`: `outer run solution 1`, 12000 uL
        - `9/A2`: `outer run solution 2`, 12000 uL
        - `9/A3`: `inner run solution 1`, 12000 uL
        - `9/A4`: `inner run solution 2`, 12000 uL

- dilution is performed according to the concentration terms given in `scan_settings.json` file

 | source solution | fraction                  | target solution      |
 |-----------------|---------------------------|----------------------|
 | inner stock 1   |   ` x_dilute_inner`       | inner run solution 1 |
 | inner diluter 1 |   ` 1 - x_dilute_inner`   | inner run solution 1 |
 |                 |                           |                      |
 | inner stock 2   |   ` x_dilute_inner`       | inner run solution 2 |
 | inner diluter 2 |   ` 1 - x_dilute_inner`   | inner run solution 2 |
 |                 |                           |                      |
 | outer stock 1   |   ` x_dilute_outer`       | outer run solution 1 |
 | outer diluter 1 |   ` 1 - x_dilute_outer`   | outer run solution 1 |
 |                 |                           |                      |
 | outer stock 2   |   ` x_dilute_outer`       | outer run solution 2 |
 | outer diluter 2 |   ` 1 - x_dilute_outer`   | outer run solution 2 |

     - `x_dilute_outer = c_surfactant_outer_run_solution / c_surfactant_outer_stock` 
     - `x_dilute_inner = c_surfactant_inner_run_solution / c_surfactant_inner_stock` 

 

![](2D_scan_v4__prepare_solutions__slot2.png)
![](2D_scan_v4__prepare_solutions__slot9.png)


### set up configuration 

- create configuration by running the script `2D_HLD_scan_v4__setup_configuration.py`


### execution ()

- run the script `2D_HLD_scan_v4__execute.py`
- inputs: 
    - slot 2: `eppendorf_24_reservoir_1500ul`
        - `2/A1`: `sample waste`, 500 uL
        - `2/A2`: `wash waste 1`, 500 uL
        - `2/A3`: `wash waste 2`, 500 uL
        - `2/A4`: `washing solution 1`, 1400 uL
    - slot 9: `3dprinted_15_tuberack_15000ul`
        - `9/A1`: `outer run solution 1`, 12000 uL
        - `9/A2`: `outer run solution 2`, 12000 uL
        - `9/A3`: `inner run solution 1`, 12000 uL
        - `9/A4`: `inner run solution 2`, 12000 uL



![](2D_scan_v4__execute__slot2.png)
![](2D_scan_v4__execute__slot9.png)



---

## Workflow: 2D_HLD_OIW_scan_v4

### in brief

- edit scan_settings file
- prepare all solutions and add them to a solution repository
- prepare run solutions and distribute washing solution
- mix and distribute inner solutions
- create configuration files
- execute run

related commands:

```
conda activate hld_ift0
cd __EXPERIMENT_PATH__
set SCRIPT=__SCRIPT_PATH__
set SETTINGS=scan_settings.json

python %SCRIPT%\new_profile_wizard.py
python %SCRIPT%\solution_repository_editor.py %SETTINGS%

python %SCRIPT%\2D_HLD_scan_v4__prepare_solutions.py %SETTINGS%
python %SCRIPT%\2D_HLD_OIW_scan_v4__distribute_solutions.py %SETTINGS%
python %SCRIPT%\2D_HLD_OIW_scan_v4__setup_configuration.py %SETTINGS%
python %SCRIPT%\2D_HLD_OIW_scan_v4__execute.py %SETTINGS%
```


### setting up scan_settings.json

### prepare solutions 

- *this step is exactly the same as for scan* `2D_HLD_scan_v4`
- run the script `2D_HLD_scan_v4__prepare_solutions.py`
- inputs: 
    - slot 9: `3dprinted_15_tuberack_15000ul`
        - `9/B1`: `outer stock 1`, 12000 uL
        - `9/B2`: `outer stock 2`, 12000 uL
        - `9/B3`: `inner stock 1`, 12000 uL
        - `9/B4`: `inner stock 2`, 12000 uL
        - `9/C1`: `outer diluter 1`, 12000 uL
        - `9/C2`: `outer diluter 2`, 12000 uL
        - `9/C3`: `inner diluter 1`, 12000 uL
        - `9/C4`: `inner diluter 2`, 12000 uL
        - `9/C5`: `washing solution`, 12000 uL

- outputs: 
    - slot 2: `eppendorf_24_reservoir_1500ul`
        - `2/A1`: `sample waste`, 500 uL
        - `2/A2`: `wash waste 1`, 500 uL
        - `2/A3`: `wash waste 2`, 500 uL
        - `2/A4`: `washing solution 1`, 1400 uL
    - slot 9: `3dprinted_15_tuberack_15000ul`
        - `9/A1`: `outer run solution 1`, 12000 uL
        - `9/A2`: `outer run solution 2`, 12000 uL
        - `9/A3`: `inner run solution 1`, 12000 uL
        - `9/A4`: `inner run solution 2`, 12000 uL

![](2D_scan_v4__prepare_solutions__slot2.png)
![](2D_scan_v4__prepare_solutions__slot9.png)


### distribute solutions: (2D_HLD_OIW_scan_v4__distribute_solutions.py)

- This step takes inner run solutions and creates 4 additional mixtures at following volumetric ratios: {80/20, 60/40, 40/60, 20/80}
- The inner run solutions and the obtained mixtures have to be manually transfered into 1.5mL vials, see next step!

- inputs: 
    - slot 9: `3dprinted_15_tuberack_15000ul`
        - `9/A3`: `inner run solution 1`, 12000 uL
        - `9/A4`: `inner run solution 2`, 12000 uL
- outputs: 
    - slot 9: `3dprinted_15_tuberack_15000ul`
        - `9/B2`: `sample 80/20`, 4000 uL
        - `9/B3`: `sample 60/40`, 4000 uL
        - `9/B4`: `sample 40/60`, 4000 uL
        - `9/B5`: `sample 20/80`, 4000 uL


![](2D_OIW_scan_v4__distribute__slot9.png){width=30%}



### setup configuration (2D_HLD_OIW_scan_v4__setup_configuration.py)

- create configuration by running the script `2D_HLD_OIW_scan_v4__setup_configuration.py`

### execute: (2D_HLD_OIW_scan_v4__execute.py) 

- inputs: 
    - slot 2: `eppendorf_24_reservoir_1500ul`
        - `2/A1`: `sample waste`, 500 uL
        - `2/A2`: `wash waste 1`, 500 uL
        - `2/A3`: `wash waste 2`, 500 uL
        - `2/A4`: `washing solution 1`, 1400 uL
        - `2/D1`: `inner run solution 1`, 1400 uL
        - `2/D2`: `sample 80/20`, 1400 uL
        - `2/D3`: `sample 60/40`, 1400 uL
        - `2/D4`: `sample 40/60`, 1400 uL
        - `2/D5`: `sample 20/80`, 1400 uL
        - `2/D6`: `inner run solution 2`, 1400 uL
    - slot 9: `3dprinted_50_tuberack_50000ul`
        - `9/A1`: `outer run solution 1`, 40000 uL
        - `9/A2`: `outer run solution 2`, 40000 uL


![](2D_OIW_scan_v4__execute__slot2.png)
![](2D_OIW_scan_v4__execute__slot9.png)


---

# Components


## settings file

- A setting file contains inputs necessary to prepare run stocks, generate configuration files and execute experimental scan

### general parameters

- path entries: 
    - `DATA_PATH`: an absolute path to this set of experiment(s)
    - `LOG_PATH`: an absolute path to log file generated during a scan
    - `CONFIG_PATH`: an absolut path to folder that contains configuration files (opentron, mixing graph, measurement)
    - `SOLUTION_REPOSITORY_PATH`: an absolute path to a solution repository
- concentration entries:
    - NOTE: a role of entries can slightly change depending on context, e.g. 3i1D_scan_v4 uses only variables `c_surfactant_outer_run_solution` and `c_surfactant_outer_stock` to determine fractions of all 8 input solutions. See details for various scans and dilution scripts.
    - `c_surfactant_outer_run_solution`: a concentration of a surfactant in an outer run solution used in an experiment; float value that is less or eaqul to `c_surfactant_outer_stock`
    - `c_surfactant_outer_stock`: a concentration of a surfactant in main oil stock, it is expected that both oil stocks contain the same concentration. 
    - `c_surfactant_inner_run_solution`: a concentration of a surfactant in aqueous phase used in an experiment; float value that is less or eaqul to `c_surfactant_inner_stock`
    - `c_surfactant_inner_stock`: a concentration of a surfactant in main aqueous stock, it is expected that both aqueous stocks contain the same concentration. 

### Stocks

- subsection `stocks` in settings file contains names of various solutions used in experiment. The solutions are stored in the solution repository associated with the experiment(s). The names given in this section have to match exactly names of solutions availabel in the solution repository. E.g. `heptane` and `Heptane` are two different names!
- running solutions are created from main stocks and diluter fluids. Their names can be arbitrarily chosen and definition of these solutions are created during the solution prep step (`2D_HLD_scan_v4__prepare_solutions.py`, `2D_HLD_OIW_scan_v4__distribute_solutions.py`)
- detailed explanation:
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

### Configurations

- this section contains labels to derive names of configurations files.
- `blank`: a generic name of blank configuration files used as template
- `start`: a label of start configuration
- `end`: a label of end configuration
- `to_reuse`: a label of configuration to be reused; handy in case when some intermediate solutions are prepared in previous runs and can be reused
- `blank_logout_OIW`: a generic name of blank configuration files used as template for OIW scan with logarithmic spacing in outer solution composition.
- `blank_OIW`:  a generic name of blank configuration files used as template for OIW scan with linear spacing in outer solution composition.

### Scan

- `experiment_metadata`: string entries of user defined metadata; these fields are added as header variable into an experimental data 
- `n_approximation`: integer, not used at the moment, leave value at `0`
- `number_of_outer_solution_points`: integer, number of different outer solution blends. outer solution is linearly blended from two stocks.
- `volume_of_outer_solution`: float, volume of outer solution in uL added to a cuvette; suggested value: 3000 to 3500
- `scan_type`: a string from set {"linear", "log", "log_and_mid"}, unrecognized string defaults to "linear"
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
         |                            |                                                               |
         | (general form)             |                                                               |
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
         |                            |                                                               |
         | (general form)             |                                                               |
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
         |                            |                                                               |
         | (general form)             |                                                               |
         | m                          | {0, 2/2^(m+1), 3/2^(m+1), 2/2^m, 3/2^m, ..., 2/2^2, 3/2^2, 2/2^1} |
        
         
         (***) point {0} is never measured



---

## solution repository

- A solution repository is a file containing all relevant substance/solution definitions
    - a repository is represented as a JSON file
- Following information is saved in a solution repository:
    - a name of a substance or a mixture/solution
    - a density of a given entity, g/ml
    - a list of constituents:
        - a name of a constituent
        - its mass fraction
- hld scans require generation of various solutions. A composition of solution/mixture created by volumetric mixing of two or more components can be easily estimated providing that a) volume, density and mass fractions of each input solution and b) density of resulting solution/mixture is known
    - density of mixture/solution is calculated by as sum of mass of all components divided by sum of volums of all components. (Here we assume that obtained mixture does not exhibit contraction of expansion upon mixing - a reasonable assumption)  
- use `solution_repository_editor.py` to edit entries of the solution repository


---

## scripts


### `solution_repository_editor.py`

- [ ] **add content**!!!

---


### `create_new_profile.py`

- [ ] **add content**!!!


---



### `new_profile_wizard.py`

- [ ] **add content**!!!


---


### `update_labware_well_positions.py`

- [ ] **add content**!!!


---



### ``

- [ ] **add content**!!!


---



### ``

- [ ] **add content**!!!


---







The scan assumes that a tested system contains two surfactants, one oil and one water soluble.

Setup files assumes that user has:

- 4 main stock solutions:
    - 2 stocks for oil phase, 
    - 2 stocks for aqueous phase. 
- 4 dilution liquids:
    - 2 dilution oils - one for each oil stock
    - 2 dilution liquids for aqueous phase - one for each stock
- surfactant concentration values are used to calculate dilution to obtain run stocks


Scans `3i1D_HLD_scan_v4` and `2D_HLD_scan_v4` were designed to characterize system with two surfactants, one oil and one water soluble; both scans use 8 stock solutions to explore system's behaviour versus salinity, EACN and surfactant concentration. `2D_HLD_scan_v4` can be used also to characterize a single surfactant versus salinity and EACN. `2D_HLD_OIW_scan_v4` is a 2D scan meant to probe volatile inner phase samples. All available runs where reworked to have more consistent terminology for the run; The inputs used in settings file are kept - at least attempted - consistent accross all runs. It is meant to allow switch from one type of scan to another in a simple manner.

A basic 2D_HLD scan requires 4 stock solutions, 

- 2 inner run solutions: these are used to be mixed in various proportions and injected into outer solution delivered into a cuvette, 
- 2 outer run solutions: these are also mixed in various proportions and delivered to a measurement cuvette
- the 2 run solutions encode one of scan variables. This could be salinity (e.g. one stock is water and the other is concentrated NaCl solution in water) or EACN (e.g. one stock is heptane and the other is hexadecane).

Most of systems have several formulation parameters and cannot be properly characterized with a single 2D_HLD scan. 8 stock solutions introduces an extra variable (surfactant concentration or ratio of surfactant concentrations) and can be used to quickly and easily mix 4 run solutions needed to run 2D_HLD scan.
```



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


