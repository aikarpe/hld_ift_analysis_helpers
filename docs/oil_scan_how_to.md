---
title: 'oil scan: how to'
date: 2025-06-27
---

# overview

## scope

This documents attempts to summarize experimental procedure(s) of hld-ift scans and to facilitate training on a hld-ift system.

## Covered topics

- HLD-IFT scans:
    - what are they?
    - what type of scans are currently implemented;
    - an anatomy of a scan
- scan and instrumental settings needed to perform a scan
    - solution repository
    - scan settings file
    - a structure of an experimental data
- procedure of setting up and executing a hld-ift scan
    - scripts available
    - planning solution recipe
    - adding solution to solution repository
    - creating scan settings
    - executing a HLD-IFT scan
    - making hook-shaped needle
- description of files/objects/resources
    - solution repository
    - recipe_settings_file
    - `generate_recepies_for_solutions_v2.py`
    - scan_settings_file
    - __make_run_solutions_script__
    - __make_run_config_script__
    - __execute_run_script__
    - __opentrons_pp_settings__
    - __measurement_settings__
    - __mixture_graph__
    - experimental data structure
    - set of experiments structure
    - 

# HLD-IFT scan(s)

A HLD-IFT scan is series of dripping experiments where composition of outer solution (media where sample is dripped into) and inner solution (a media that is dripped into an outer solution). An HLD-IFT scan: 

- creates all test solutions from specified stock solutions,
- performs the dripping experiment,
- records experimental conditions and a video of dripping dynamics

Overall workflow is represented in <diagramm-workflow>

![](images/diagramm_workflow.png) [^diagramm_workflow_definition]

There are two type of scans at present:

- direct scan: a 2D grid scan intended for dripping aqueous solutions into an oil phase. 
- inverse scan: a 2D grid scan for dripping oil phase into an aqueous solutions (with hook-shaped needle); due to organic volatility these scans are split into few (3) chunks.

## Direct HLD-IFT scan

This is procedure to acquire dripping data for a 2D grid

This scan requires:

- 2 stock solutions for inner solution;
- 2 stock solutions for outer solution;
- scan settings file that points to all relevant configurations;

and it creates:

- new folder that is populated with:
    - logging data (`<experiment_name>\log.log`)
    - data file keeping track of various experimental metadata (`<experiment_name>\data.json`)
    - images (`<experiment_name>/<scan_name>/<measurement_name>/<image_name>.jpg`)

There are several helper scripts that help to prepare stock solutions and scan settings for this scan. These scripts assume that user uses particular layout[^note_on_vial_positions_in_layout].

## inverse HLD-IFT scan

This is a modification of a direct HLD-IFT scan to be used with volatile samples. The current version splits whole scan in several (3) chunks and 2 samples (inner solutions) are tested against all selected outer solutions (such procedure limits time that container of a volatile sample has to be left open).
To obtain a single inverse HLD-IFT scan several (3) scans have to be performed and data combined later.


This scan requires:

- 2 stock solutions for inner solution;
- 2 stock solutions for outer solution;
- scan settings file that points to all relevant configurations;

and it creates:

- new folder that is populated with:
    - logging data (`<experiment_name>\log.log`)
    - data file keeping track of various experimental metadata (`<experiment_name>\data.json`)
    - images (`<experiment_name>/<scan_name>/<measurement_name>/<image_name>.jpg`)

---

# procedure

This section aims to describe general procedure to acquire HLD-IFT scans. In general experiment can consist of 1 or multiple scans, the discussed workflow assumes that multiple scans are needed.

Below is a very generic list of operations needed to execute one or more scans:

1. evaluate surfactant solubility in oils to be used for scan
2. plan preparation of stock solution 
3. make stock(s) and add their representation to a solution repository
4. make main configuration 
5. make run stock (and distribute washing solution) 
6. prepare scan configurations
7. check camera
8. execute scan

---

## surfactant solubility in oil (step 1)

1. with neat surfactant measure 2 aliquots of 0.1 g surfactants in 5 mL vials
2. with heptane and hexadecane test solubility (see solubility testing)

### solubility testing

- to 0.1g of surfactant add 0.4 mL of solution (yields ~ 20g/100ml of surfactant), note appearance
- add extra 0.25 mL of solution (~ 13.3g/100ml of surfactant), note appearance
- add extra 0.25 mL of solution (~ 10g/100ml of surfactant), note appearance
- add extra 0.50 mL of solution (~ 6.7g/100ml of surfactant), note appearance
- add extra 0.50 mL of solution (~ 5g/100ml of surfactant), note appearance
- procedure can be terminated whenever there is a clear sign that surfactant is soluble in a chosen liquid

---

## stock solution preparation (steps 2 and 3)

- in the context of an oil scan there are 3 parts relevant to stock solutions:
    - planning: what components and in what quantities to mix
    - actual preparation of stock solution
    - making of stock solution representation for oil scan(s) and adding to a solution repository

### planning

- Providing that surfactant is going to dissolve in a chosen oil, recipe(s) for stock(s) can be generated using `generate_recepies_for_solutions_v2.py`. The script will optimize a binary mixture from solutions (compounds) available in repository. Recipes are estimated from ingredient densities and assume no volumetric changes; the assumption is reasonably good for most solutions and any significant deviations can be accounted for with density measurement.
- to generate a recipe call execute command like:

```
python generate_recepies_for_solutions_v2.py <recipe_file>
```
- recipes needed should be specified in <recipe_file><link_to_settings_section_about_recipe_file> in `recepies` section; it can contain multiple recipes targets
- each recipe should contain following parameters:
    - solution1, solution2: string, a name of solutions (compounds) to be used to generate recipe; names of solutions should be defined in a selected solution repository
    - component_to_target: string, a name of a component which concentration is to be targeted
    - target_concentration: numeric, concentration in g/g (m/m) or g/ml (m/v) units
    - concentration_type: a string, mass ("m/m") or mass to volume concentration ("m/v")
    - quantity: numeric, quantity of final solution needed in grams

- script will output 3 section per each recipe:
    - instructions to make solution
    - a dictionary template to create sock solution representation
    - a serialized solution object of a stock

### stock preparation and density measurement

- make stock according to recipe by dissolving m_surfactant g of surfactant in m_solvent g of solvent
- add 5 ml (v_ro) of di water in 5 mL glass vial, note mass (m_ro_water)
- add 5 ml of stock in 5 ml vial, note mass (m_ro)
- note values of m_surfactant, m_solvent, v_ro, m_ro_water, m_ro

### define solution in a solution repository

- edit `compounds_to_add` section of <recipe_file>  to incorporate all prepared stock solutions:
    - use previously noted values of m_surfactant, m_solvent, v_ro, m_ro_water, m_ro to create dictionary for binary mixture
    - solution name could be anything as long as it describes well your stock; it will be used to reference stock in other scripts

- run script:

```
python solution_repository__edit.py <recipe_file> 
```

After the script execution, a solution repository contains two new items `super_duper_stock_solution_20250627` and `idrosal_sxs40`

- for an example see [section: Example: plan recipe](## Example: plan recipe)

##  make main configuration

Main configuration is json file that contains info about various inputs needed and outputs used needed to make up run stocks and execute scan.
For more details on the entries of a configuration file see __reference_to_settings_main_configuration__.

A main configuration is used by 3 different scripts:

- `2D_HLD_scan_v2__prepare_solutions.py` makes up run buffers and distributes washing solutions
- `2D_HLD_scan_v2__setup_configuration.py` prepares lower level configurations needed for a scan
- `2D_HLD_scan_v2__execute.py`: executes scan

__describe_what_needs_editing_in_main_configuration__

---

# direct scan

### for solution prep

| slot 9 | 1                                                       | 2                                            | 3                             | 4                              | 5                       |
|:-------|:--------------------------------------------------------|:---------------------------------------------|:------------------------------|:-------------------------------|:------------------------|
| A      | A1<br>stock of surfactant in C7<br> 12000               | A2<br>stock of surfactant in C16<br> 12000   | A3<br>stock of C7<br> 12000   | A4<br>stock of C16<br> 12000   | A5<br>water<br> 12000   |
| B      | B1<br>run stock of surfactant in C7<br>     0    12500  |   <br><br>                                   |   <br><br>                    |   <br><br>                     |   <br><br>              |
| C      | C1<br>run stock of surfactant in C16<br>     0    12500 |   <br><br>                                   |   <br><br>                    |   <br><br>                     |   <br><br>              |


|slot 2 |1                                            |2                                           |3                                          |4                                           |
|:------|:--------------------------------------------|:-------------------------------------------|:------------------------------------------|:-------------------------------------------|
|A      |A1<br>unused sample waste<br>     0      500 |A2<br>sample rinse waste<br>     0      500 |A3<br>first rinse waste<br>     0      500 |A4<br>first rinse source<br>     0     1400 |

### for 2D scan

|slot 9 |1                       |2                                        |3                                               |4                                                |
|:------|:-----------------------|:----------------------------------------|:-----------------------------------------------|:------------------------------------------------|
|A      |A1<br>water<br> 12000   |A2<br>stock of NaCl solution<br> 12000   |A3<br>run stock of surfactant in C7<br> 12000   |A4<br>run stock of surfactant in C16<br> 12000   |


|slot 2 |1                                     |2                                    |3                                   |4                                    |
|:------|:-------------------------------------|:------------------------------------|:-----------------------------------|:------------------------------------|
|A      |A1<br>unused sample waste<br>   500   |A2<br>sample rinse waste<br>   500   |A3<br>first rinse waste<br>   500   |A4<br>first rinse source<br>  1400   |
 
---

# inverse scan

### for solution prep

| slot 9 | 1                                                       | 2                                            | 3                                           | 4                                         | 5                                         |
|:-------|:--------------------------------------------------------|:---------------------------------------------|:--------------------------------------------|:------------------------------------------|:------------------------------------------|
| A      | A1<br>stock of surfactant in C7<br> 12000               | A2<br>stock of surfactant in C16<br> 12000   | A3<br>stock of C7<br> 12000                 | A4<br>stock of C16<br> 12000              | A5<br>stock of C16<br> 12000              |
| B      | B1<br>surfactant in C7:C16 5:0<br> 0  4000              | B2<br>surfactant in C7:C16 4:1<br> 0  4000   | B3<br>surfactant in C7:C16 3:2<br> 0  4000  | B4<br>surfactant in C7:C16 2:3<br> 0  4000| B5<br>surfactant in C7:C16 1:4<br> 0  4000|
| C      | C1<br>surfactant in C7:C16 0:5<br> 0  4000              | C2<br><br>                                   | C3<br><br>                                  | C4<br><br>                                | C5<br><br>                                |




|slot 2 |1                                            |2                                           |3                                          |4                                           |
|:------|:--------------------------------------------|:-------------------------------------------|:------------------------------------------|:-------------------------------------------|
|A      |A1<br>unused sample waste<br>     0      500 |A2<br>sample rinse waste<br>     0      500 |A3<br>first rinse waste<br>     0      500 |A4<br>first rinse source<br>     0     1400 |

### for 2D scan

|slot 9 |1                       |2                                        |3                                               |4                                                |
|:------|:-----------------------|:----------------------------------------|:-----------------------------------------------|:------------------------------------------------|
|A      |A1                      |A2                                       |A3<br>water <br> 14500                          |A4<br>NaCl stock<br> 14500                       |


|slot 2 |1                                     |2                                     |3                                     |4                                     |5                                      |6                                      |
|:------|:-------------------------------------|:-------------------------------------|:-------------------------------------|:-------------------------------------|:--------------------------------------|:--------------------------------------|
|A      |A1<br>unused sample waste<br>   500   |A2<br>sample rinse waste<br>   500    |A3<br>first rinse waste<br>   500     |A4<br>first rinse source<br>  1400    | A5                                    | A6                                    |
|B      |B1                                    |B2                                    |B3                                    |B4                                    | B5                                    | B6                                    |
|C      |C1                                    |C2                                    |C3                                    |C4                                    | C5                                    | C6                                    |
|D      |D1<br>surfactant in C7:C16 5:0<br>1400|D2<br>surfactant in C7:C16 4:1<br>1400|D3<br>surfactant in C7:C16 3:2<br>1400|D4<br>surfactant in C7:C16 2:3<br>1400| D5<br>surfactant in C7:C16 1:4<br>1400| D6<br>surfactant in C7:C16 0:5<br>1400|
 
---

# direct scan

### for solution prep

| slot 9 | 1                                                       | 2                                            | 3                             | 4                              | 5                       |
|:-------|:--------------------------------------------------------|:---------------------------------------------|:------------------------------|:-------------------------------|:------------------------|
| A      | A1<br>stock of surfactant in C7<br> 12000               | A2<br>stock of surfactant in C16<br> 12000   | A3<br>stock of C7<br> 12000   | A4<br>stock of C16<br> 12000   | A5<br>water<br> 12000   |
| B      | B1<br>run stock of surfactant in C7<br>     0    12500  |   <br><br>                                   |   <br><br>                    |   <br><br>                     |   <br><br>              |
| C      | C1<br>run stock of surfactant in C16<br>     0    12500 |   <br><br>                                   |   <br><br>                    |   <br><br>                     |   <br><br>              |


|slot 2 |1                                            |2                                           |3                                          |4                                           |
|:------|:--------------------------------------------|:-------------------------------------------|:------------------------------------------|:-------------------------------------------|
|A      |A1<br>unused sample waste<br>     0      500 |A2<br>sample rinse waste<br>     0      500 |A3<br>first rinse waste<br>     0      500 |A4<br>first rinse source<br>     0     1400 |

### for 2D scan

|slot 9 |1                       |2                                        |3                                               |4                                                |
|:------|:-----------------------|:----------------------------------------|:-----------------------------------------------|:------------------------------------------------|
|A      |A1<br>water<br> 12000   |A2<br>stock of NaCl solution<br> 12000   |A3<br>run stock of surfactant in C7<br> 12000   |A4<br>run stock of surfactant in C16<br> 12000   |


|slot 2 |1                                     |2                                    |3                                   |4                                    |
|:------|:-------------------------------------|:------------------------------------|:-----------------------------------|:------------------------------------|
|A      |A1<br>unused sample waste<br>   500   |A2<br>sample rinse waste<br>   500   |A3<br>first rinse waste<br>   500   |A4<br>first rinse source<br>  1400   |
 
---

# inverse scan

### for solution prep

| slot 9 | 1                                                       | 2                                            | 3                                           | 4                                         | 5                                         |
|:-------|:--------------------------------------------------------|:---------------------------------------------|:--------------------------------------------|:------------------------------------------|:------------------------------------------|
| A      | A1<br>stock of surfactant in C7<br> 12000               | A2<br>stock of surfactant in C16<br> 12000   | A3<br>stock of C7<br> 12000                 | A4<br>stock of C16<br> 12000              | A5<br>stock of C16<br> 12000              |
| B      | B1<br>surfactant in C7:C16 5:0<br> 0  4000              | B2<br>surfactant in C7:C16 4:1<br> 0  4000   | B3<br>surfactant in C7:C16 3:2<br> 0  4000  | B4<br>surfactant in C7:C16 2:3<br> 0  4000| B5<br>surfactant in C7:C16 1:4<br> 0  4000|
| C      | C1<br>surfactant in C7:C16 0:5<br> 0  4000              | <br><br>                                     |   <br><br>                                  |   <br><br>                                |   <br><br>                                |




|slot 2 |1                                            |2                                           |3                                          |4                                           |
|:------|:--------------------------------------------|:-------------------------------------------|:------------------------------------------|:-------------------------------------------|
|A      |A1<br>unused sample waste<br>     0      500 |A2<br>sample rinse waste<br>     0      500 |A3<br>first rinse waste<br>     0      500 |A4<br>first rinse source<br>     0     1400 |

### for 2D scan

|slot 9 |1                       |2                                        |3                                               |4                                                |
|:------|:-----------------------|:----------------------------------------|:-----------------------------------------------|:------------------------------------------------|
|A      |                        |                                         |A3<br>water <br> 14500                          |A4<br>NaCl stock<br> 14500                       |


|slot 2 |1                                     |2                                     |3                                     |4                                     |5                                      |6                                      |
|:------|:-------------------------------------|:-------------------------------------|:-------------------------------------|:-------------------------------------|:--------------------------------------|:--------------------------------------|
|A      |A1<br>unused sample waste<br>   500   |A2<br>sample rinse waste<br>   500    |A3<br>first rinse waste<br>   500     |A4<br>first rinse source<br>  1400    |                                       |                                       |
|B      |                                      |                                      |                                      |                                      |                                       |                                       |
|C      |                                      |                                      |                                      |                                      |                                       |                                       |
|D      |D1<br>surfactant in C7:C16 5:0<br>1400|D2<br>surfactant in C7:C16 4:1<br>1400|D3<br>surfactant in C7:C16 3:2<br>1400|D4<br>surfactant in C7:C16 2:3<br>1400| D5<br>surfactant in C7:C16 1:4<br>1400| D6<br>surfactant in C7:C16 0:5<br>1400|
 
---

# Examples

## Example: plan recipe

1. edit/create `__recipes_solutions.json__` file 

- specify valid solution repository where other ingredients are defined
- if needed, specify solutions, ingredients that needs to be added to solution repository (e.g. define new surfactant or solvent that you are going to use)
- specify recipe that you need:
    - solutions to be used (refer to them by name exactly as in solution repository, please note `good_solvent` and `Good_Solvent` are different names!)
    - target component to be used (name exactly)
    - concentration needed
    - concentration type (m/m, m/v) needed
    - total amount needed in grams

Below is an example of `recipes_solutions.json` file prior to request:

```
{
  "solution_repository_path": "C:/Users/admin/Documents/Data/aikars/opentron/Tergitol_15_S_5_OIW_C7_C16/config/solution_repository.json",
  "compounds_to_add": [],
  "recepies": [
    {
      "solution1": "Tergitol_15_S_5",
      "solution2": "hexadecane",
      "component_to_target": "Tergitol_15_S_5",
      "target_concentration": 0.2,
      "concentration_type": "m/v", 
      "quantity": 38.5
    },
    {
      "solution1": "Tergitol_15_S_5",
      "solution2": "heptane",
      "component_to_target": "Tergitol_15_S_5",
      "target_concentration": 0.2,
      "concentration_type": "m/v", 
      "quantity": 36.5
    }
  ]
}
```

2. execute `generate_recepies_for_solutions_v2.py`

```{command-prompt}
>>>cd C:\Users\admin\Documents\Data\aikars\opentron\Tergitol_15_S_5_OIW_C7_C16
>>>set SCRIPT="\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\scripts"
>>>python %SCRIPT%\\generate_recepies_for_solutions_v2.py  recipes_solutions.json
```

- This will generate following output (only partially shown):


```
> 
> ========================================
> optimizing for
>         component: `Tergitol_15_S_5`
>         concentration type: `m/v`
>         target: 0.2 g/ml
> Quantity: 38.5000 g
> Volume: 47.87 ml
> 
> Use:
>     9.5724 g of Tergitol_15_S_5 and
>     28.9276 g of hexadecane
> 
>     m(hexadecane)/m(Tergitol_15_S_5):  3.021995  g/g
> 
> ------- dictionary for inclusion in solution repository -------
> 
> a_dict_name = {
>     "mixture_type": "binary_mix",
>     "name": " 7.5e-01 hexadecane 2.5e-01 Tergitol_15_S_5_20260222",
>     "m_solute":   _9.5724_,
>     "m_solvent":  _28.9276_,
>     "solvent":   "hexadecane",
>     "solute":    "Tergitol_15_S_5",
>     "v_ro":       _1.0_,
>     "m_ro_water": _1.0_,
>     "m_ro":       _0.80429_,
>     "date":       "2026-02-22",
> }
> 
> ------- target solution representation -------
> 
> {
>   "name": " 7.5e-01 hexadecane 2.5e-01 Tergitol_15_S_5",
>   "ro": 0.8042937218822203,
>   "label": "_7.5e-01_hexadecane_2.5e-01_Tergitol_15_S_5",
>   "components": {
>     "hexadecane": {
>       "label": "hexadecane",
>       "w": 0.7513671875000002
>     },
>     "Tergitol_15_S_5": {
>       "label": "Tergitol_15_S_5",
>       "w": 0.2486328124999998
>     }
>   }
> }
> 
> ========================================
> optimizing for
>         component: `Tergitol_15_S_5`
>         concentration type: `m/v`
>         target: 0.2 g/ml
> Quantity: 36.5000 g
> Volume: 49.43 ml
> 
> Use:
>     9.8842 g of Tergitol_15_S_5 and
>     26.6158 g of heptane
> 
>     m(heptane)/m(Tergitol_15_S_5):  2.692752  g/g
> 
> ------- dictionary for inclusion in solution repository -------
> 
> a_dict_name = {
>     "mixture_type": "binary_mix",
>     "name": " 2.7e-01 Tergitol_15_S_5 7.3e-01 heptane_20260222",
>     "m_solute":   _9.8842_,
>     "m_solvent":  _26.6158_,
>     "solvent":   "heptane",
>     "solute":    "Tergitol_15_S_5",
>     "v_ro":       _1.0_,
>     "m_ro_water": _1.0_,
>     "m_ro":       _0.73845_,
>     "date":       "2026-02-22",
> }
> 
> ------- target solution representation -------
> 
> {
>   "name": " 2.7e-01 Tergitol_15_S_5 7.3e-01 heptane",
>   "ro": 0.7384497201499135,
>   "label": "_2.7e-01_Tergitol_15_S_5_7.3e-01_heptane",
>   "components": {
>     "Tergitol_15_S_5": {
>       "label": "Tergitol_15_S_5",
>       "w": 0.27080078124999984
>     },
>     "heptane": {
>       "label": "heptane",
>       "w": 0.7291992187500002
>     }
>   }
> }
> 
> ===========================================================
> 
```


3. prepare a solution (physical step in a lab)

| components | suggested by recipe | actual surfactant, estimate of a solvent | actual surfactant and solvent |
|------------|---------------------|------------------------------------------|-------------------------------|
| Tergitol 15-S-5 |  9.8842 |   __9.899__ |     __9.899__ |
| heptane         | 26.6158 |   26.6556   |    __26.665__ |
|                 |         |             |               |
| Tergitol 15-S-5 |  9.5724 |   __9.572__ |     __9.572__ |
| hexadecane      | 28.9276 |   28.9265   |    __28.956__ |

This step involves a density measurement. We have been taking aliquots (e.g. 5 mL) and wieghing them. Water is used as reference, and density estimates leads to at least 3 values:

- `v_ro`: volume used
- `m_ro_water`: mass of water aliquot
- `m_ro`: mass of solution aliquot


4. edit `recipes_solutions.json` file. Add section "compounds_to_add" (compare content of a file in this step vs content in the step 1.)

```
{
  "solution_repository_path": "C:/Users/admin/Documents/Data/aikars/opentron/Tergitol_15_S_5_OIW_C7_C16/config/solution_repository.json",
  "compounds_to_add": [
  {
    "mixture_type": "binary_mix",
    "name": "20g/100mL Tergitol_15_S_5 in heptane 2026-02-11",
    "m_solute":   9.899,
    "m_solvent":  26.665,
    "solvent":   "heptane",
    "solute":    "Tergitol_15_S_5",
    "v_ro":       5.0,
    "m_ro_water": 5.00044,
    "m_ro":       3.69379,
    "date":       "2026-02-11"
  },
  {
    "mixture_type": "binary_mix",
    "name": "20g/100mL Tergitol_15_S_5 in hexadecane 2026-02-11",
    "m_solute":   9.572,
    "m_solvent":  28.956,
    "solvent":   "hexadecane",
    "solute":    "Tergitol_15_S_5",
    "v_ro":       5.0,
    "m_ro_water": 5.00044,
    "m_ro":       4.10178,
    "date":       "2026-02-11"
  },
  {
    "mixture_type": "pure_compound",
    "ro": 0.963,
    "name": "Tergitol_15_S_5",
    "label": "Tergitol_15_S_5"
  }
  ],
  "recepies": [
    {
      "solution1": "Tergitol_15_S_5",
      "solution2": "hexadecane",
      "component_to_target": "Tergitol_15_S_5",
      "target_concentration": 0.2,
      "concentration_type": "m/v", 
      "quantity": 38.5
    },
    {
      "solution1": "Tergitol_15_S_5",
      "solution2": "heptane",
      "component_to_target": "Tergitol_15_S_5",
      "target_concentration": 0.2,
      "concentration_type": "m/v", 
      "quantity": 36.5
    }
  ]
}
```

5. Add solution definition of stock solutions to solution repository by running:

```
>>>python %SCRIPT%\\solution_repository__edit.py recipes_solutions.json
```

This command edits `solution_repository.json` and adds two solution definitions:

```
...
  "20g/100mL Tergitol_15_S_5 in heptane 2026-02-11": {
    "components": {
      "Tergitol_15_S_5": {
        "label": "Tergitol_15_S_5",
        "w": 0.27073077343835467
      },
      "heptane": {
        "label": "heptane",
        "w": 0.7292692265616453
      }
    },
    "label": "_7.3e-01_heptane_2.7e-01_Tergitol_15_S_5",
    "name": "20g/100mL Tergitol_15_S_5 in heptane 2026-02-11",
    "ro": 0.7386929950164385
  },
  "20g/100mL Tergitol_15_S_5 in hexadecane 2026-02-11": {
    "components": {
      "Tergitol_15_S_5": {
        "label": "Tergitol_15_S_5",
        "w": 0.24844269102990033
      },
      "hexadecane": {
        "label": "hexadecane",
        "w": 0.7515573089700996
      }
    },
    "label": "_2.5e-01_Tergitol_15_S_5_7.5e-01_hexadecane",
    "name": "20g/100mL Tergitol_15_S_5 in hexadecane 2026-02-11",
    "ro": 0.8202838150242778
  },
...
```

Now these stock solutions are ready to be used in other scripts.


---

# Notes

[^outer_solution_mixing_types]: I am planning to add log2 mixing for this scan dimension too!!! __edit_whenever_appropriate__

[^note_on_vial_positions_in_layout]: positions for stocks can be changed, but that requires careful changes in scripts and/or configuration files
!!!

[^diagramm_workflow_definition]: see mermaid diagramm-workflow block

```{mermaid diagramm-workflow}
%%% flowchart TB
flowchart LR
    A((start))
    B{more<br>outer<br>solution<br>compositions?}
    C(mix next outer solution in a cuvette)
    D{more<br>inner<br>solution<br>compositions?}
    E{is<br>inner<br>solution<br>available?}
    F("mix inner solution (sample)")
    G(wash needle)
    H(load sample)
    I(dripping experiment)
    J(discard unused sample)
    K((end))
    A --> B
    B -->|No|K
    B --Yes--> C --> D
    D -->|No|B
    D --Yes--> E
    E --Yes--> G
    E -->|No|F
    F --> G --> H --> I --> J --> D
    classDef centernode fill:#44A194,stroke:#44A194,stroke-width:2px,color:#F4F0E4;
    classDef areanode fill:#537D96,stroke:#44A194,stroke-width:2px,color:#F4F0E4;
    classDef regnode2 fill:#F4F0E4,stroke:#537D96,stroke-width:2px,color:#537D96;
    class A,B,C,D,E,F,G,H,I,J,K centernode
    class B,C,D,E,F,G,H,I,J areanode
    class B,C,D,E,F,G,H,I,J regnode2
    linkStyle 0,1,2,3,4,5,6,7,8,9,10,11,12 stroke:#44A194,stroke-width:2px,fill: none, color: black;
```


```{mermaid experiment-file-structure-1}
flowchart TB
  ROOT((Experiment root))
  log[["log.log"]]
  data[[data.json]]
  scan1@{ shape: procs, label: "scan_001"}
  scan2@{ shape: procs, label: "scan_002"}
  scandot@{ shape: procs, label: "..."}
  scanN@{ shape: procs, label: "scan_00N"}
  scan1c0@{ shape: procs, label: "conc_0.00000"}
  scan1cp@{ shape: procs, label: "..."}
  scan1c1@{ shape: procs, label: "conc_1.00000"}
  scan2c0@{ shape: procs, label: "conc_0.00000"}
  scan2cp@{ shape: procs, label: "..."}
  scan2c1@{ shape: procs, label: "conc_1.00000"}
  scanNc0@{ shape: procs, label: "conc_0.00000"}
  scanNcp@{ shape: procs, label: "..."}
  scanNc1@{ shape: procs, label: "conc_1.00000"}
  img1[[00000.jpg]]
  img2[[00001.jpg]]
  img3[[00002.jpg]]
  img4[[...]]
  ROOT --> log & data & scan1 & scan2 & scandot & scanN
  scan1 --> scan1c0 & scan1cp & scan1c1
  scan2 --> scan2c0 & scan2cp & scan2c1
  scanN --> scanNc0 & scanNcp & scanNc1
  scan1c0 --> img1 & img2 & img3 & img4
```

   
```{mermaid complete-sow-experiment-unknown-surfactant-characterization}
flowchart LR
  Z1((start))
  Z2((done))
  Z1 --> A(Make general stocks: 20g/100mL surfactant in oil) --> B(Add stock solutions to `solution_repository`)
  B --> C{more<br>surfactant<br>concentrations<br>to run?}
  C -->|No|Z2
  subgraph exp [2D HLD-IFT scan]
    direction TB
    S1(Make run stocks)
    S2("Make run (scan) configuration")
    S3(Execute run)
    S1 --> S2 --> S3
  end
  C -->|Yes|exp
  S3 --> C    
```


```{mermaid complete-sow-experiment-unknown-surfactant-characterization-file-structure}
flowchart LR
  ROOT((.. root ..))
  config@{ shape: procs, label: "config"}
  exp1@{ shape: procs, label: "exp_superSurf_05g_C7C16_20260101"}
  exp2@{ shape: procs, label: "exp_superSurf_10g_C7C16_20260102"}
  exp3@{ shape: procs, label: "exp_superSurf_20g_C7C16_20260103"}
  settingsscan[[scan_settings.json]]
  settingssol[[recipes_solutions.json]]
  cpb[[command_prompt_bits.md]]
  c1_em[["config_superSurf_05g_C7C16_20260101__execute_measurement.json"]]
  c1_mg[["config_superSurf_05g_C7C16_20260101__mixing_graph.json"]]
  c1_opp[["config_superSurf_05g_C7C16_20260101__opentron_pp.json"]]
  sol_rep[["solution_repository.json"]]
  crest[[...]]

  ROOT --> config & exp1 & exp2 & exp3 & settingsscan & settingssol & cpb
  config --> c1_em & c1_mg & c1_opp & sol_rep & crest
```

```{mermaid solution-class-object}
flowchart TB
  subgraph Solution
    direction TB
    name[[Name]]
    label[[Label]]
    ro[[Density]]
    subgraph Composition
      direction TB
      subgraph surfactant
          direction TB
          labelsurf[[Label]]
          wsurf[["mass_fraction(w)"]]
      end
      subgraph compoundA [solvent]
          direction TB
          labelscA[[Label]]
          wcA[["mass_fraction(w)"]]
      end
      compoundB[[...]]
      surfactant ~~~ compoundA ~~~ compoundB
    end
    name ~~~ label ~~~ ro ~~~ Composition
  end
```

```{mermaid setup-layout-general}
flowchart TB
  A("PC/laptop") -->|API commands| B(opentron OT-2)
  A --> C(camera)
  B -->|log data|A
  C -->|image data|A
```

```{mermaid settings-description}
flowchart TB
  opentronpp("config_\<scan_profile_name\>__opentron_pp.json")
  meas("config_<scan_profile_name>__execute_measurement.json")
  mixinggraph("config_<scan_profile_name>__mixing_graph.json")
  scan("scan_settings.json")
  solutions("solution_repository.json")
  main(settings) --> opentronpp & meas & mixinggraph & solutions & scan
```
There are several helper scripts that help to prepare stock solutions and scan settings for this scan. These scripts assume that user uses particular layout[^note_on_vial_positions_in_layout].



contains info about:
    - solution repository to be used;
    - opentron_pp configuration
    - measurement configuration

A dripping dynamics is recorded as series of images

---

# settings

Scan settings control scan parameters. Various aspect are discussed in this section.

## scan settings

A collection of settings associated with the scan:

- main scan parameters:
- input/output locations:
- metadata:

### inner solution

Samples are created from two inner solution stocks. They can be mixed according to one of given schemes:

- `linear`: ...
- `log2`: ...
- `logmid???`: ...

__edit_list_all_options_here__

### outer solution 

Outer solutions are created by linear mixing of two outer stock solutions[^outer_solution_mixing_types]


### a recipe (json input) file

This is a json file containing data necessary 1) to plan recipes and 2) add solution representation to a solution repository

```
{
        "solution_repository_path": "D:/temp_data/solution_repository.json",
        "compounds_to_add": [
              {
                "mixture_type": "binary_mix",
                "name": "44perc_IB-45_in_wt",
                "m_solute":   0.44,
                "m_solvent":  0.56,
                "ro":         1.12,
                "solvent":   "water",
                "solute":    "aerosol_ib-45",
                "v_ro":       1.0,
                "m_ro_water": 1.0,
                "m_ro":       1.12,
                "date":       "2025-03-02"
              },
              {
                "mixture_type": "pure_compound",
                "ro": 1.17,
                "name": "idrosal_sxs40",
                "label": "idrosal_sxs40",
                "cas": "1300-72-7",
                "alt_name": "Sodium Xylenesulfonate",
                "note": "idrosal_sxs40 is name for 40% solution of sodium xylenesulfonate"
              }
            ],
        "recepies": [
                {
                    "solution1": "sdbs",
                    "solution2": "water",
                    "component_to_target": "sdbs", 
                    "target_concentration": 0.2,
                    "concentration_type": "m/v", 
                    "quantity": 50
                },
                {
                    "solution1": "idrosal_sxs40",
                    "solution2": "water",
                    "component_to_target": "idrosal_sxs40",
                    "target_concentration": 0.2,
                    "concentration_type": "m/v",
                    "quantity": 50
                },
                {
                    "solution1": "44perc_IB-45_in_wt",
                    "solution2": "water",
                    "component_to_target": "aerosol_ib-45",
                    "target_concentration": 0.2,
                    "concentration_type": "m/v",
                    "quantity": 50
                }
            ]
}
```
        
#### key description

- __`solution_repository_path`__: string, a path to a solution repository file; it is needed for recipe generation and to add stock solution representation
- __`compounds_to_add`__: a list of dictionaries of binary mixture or pure compound definitions; this section is used when solution(s) are added to a solution repository. It can contain zero or more dictionaries that contain definition of mixture and/or pure compound.
    - to define a pure compound a dictionary should contain following keys:
        - __`mixture_type`__: "pure_compound"
        - __`ro`__: density of a compound in g/mL
        - __`name`__: string containing name of a compound
        - __`label`__: string containing label of a compound (NOTE it is strongly encouraged to use same name and label for a pure compound!!!)
        - name, label: a name of this compound, will be used to refer to it in a solution repository; label and name should be the same!!!
        - other keys can be provided (e.g. for refrence purposes), but are ignored
    - to define a mixture a dictionary should contain following keys:
        - __`mixture_type`__: string, should be set to "binary_mix"
        - __`name`__: a name of this mixture, will be used to refer to it in a solution repository
        - __`m_solute`__, __`m_solvent`__: a mass of each component in grams
        - __`ro`__: numeric, a density in g/ml, if this key is present values in keys: __`v_ro`__, __`m_ro_water`__ and __`m_ro`__ are ignored
        - __`solvent`__, __`solute`__: names of solutions as specified in a solution repository
        - __`v_ro`__: numeric, a volume of liquid(s) in mL used to measure density
        - __`m_ro_water`__: a mass of water in grams in reference measurement
        - __`m_ro`__: a mass of a mixture in grams in density measurement; density of a mixture is evaluated as `m_ro/m_ro_water`; the evaluation is skipped if key `ro` is defined
        - other keys can be provided (e.g. for refrence purposes), but are ignored
        -
- __`recepies`__: a list of dictionaries specifying recipe requests; this section is necessary to generate recipes, nothing is added to a solution repository.
    - dictionary to define recipe requests should contain following keys:
        - __`solution1`__, __`solution2`__: string, a name of solutions (compounds) to be used to generate recipe; names of solutions should be defined in a selected solution repository
        - __`component_to_target`__: string, a name of a component which concentration is to be targeted
        - __`target_concentration`__: numeric, concentration in g/g (m/m) or g/ml (m/v) units
        - __`concentration_type`__: a string, mass ("m/m") or mass to volume concentration ("m/v")
        - __`quantity`__: numeric, quantity of final solution needed in grams


##  main configuration

Run configuration is json file with all the parameters needed to make up run stocks and execute scan. A run configuration is used by 3 different scripts:

- `2D_HLD_scan_v2__prepare_solutions.py`: make up run buffers and distributes wash solutions
- `2D_HLD_scan_v2__setup_configuration.py`: prepares configuration for scan
- `2D_HLD_scan_v2__execute.py`: executes run



```
{
    "note": "a json representation of inputs for solution preparation for 2D scan, surfactant in oil",
    "DATA_PATH"   : "C:/Users/admin/Documents/Data/aikars/opentron/Novel810_3.5_C07C16_NaCl",
    "LOG_PATH"    : "C:/Users/admin/Documents/Data/aikars/opentron/Novel810_3.5_C07C16_NaCl/log.log",
    "CONFIG_PATH" : "C:/Users/admin/Documents/Data/aikars/opentron/Novel810_3.5_C07C16_NaCl/config",
    "SOLUTION_REPOSITORY_PATH" : "C:/Users/admin/Documents/Data/aikars/opentron/Novel810_3.5_C07C16_NaCl/config/solution_repository.json",
    "c_surfactant_experiment": 5,
    "c_surfactant_stock": 20,
    "stocks": {
        "surfactant_in_oil_1": "name....",
        "surfactant_in_oil_2": "name ...", 
        "oil_1": "heptane",
        "oil_2": "hexadecane",
        "stock_aqueous_1": "water",
        "stock_aqueous_2": "30g_NaCl_in_100mL_water_20250225",
        "run_stock_surf_oil_1": "name",
        "run_stock_surf_oil_2": "name" 
    },
    "configurations": {
        "blank": "hld_ift_experiment__blank",
        "start": "2025-06-13_20.00g_Novio810-3.5_C7C16_NaCl",
        "end": "2025-06-13_20.00g_Novio810-3.5_C7C16_NaCl_001",
        "to_reuse": {
            "name": "2025-06-12_15.00g_Novio810-3.5_C7C16_NaCl_001",
            "content": [
                { "address": { "slot": "2", "well_location": "C6" } },
                { "address": { "slot": "2", "well_location": "D1" } },
                { "address": { "slot": "2", "well_location": "D2" } },
                { "address": { "slot": "2", "well_location": "D3" } },
                { "address": { "slot": "2", "well_location": "D4" } },
                { "address": { "slot": "2", "well_location": "D5" } },
                { "address": { "slot": "2", "well_location": "D6" } }
            ]
        }
    },
    "scan": {
        "experiment_metadata": {
            "description": "Novio 810-3.5 20g/100ml oil (C7/C16)",
            "needle_dia": 0.312,
            "oil": "C7 to C16",
            "measurement": "pulsed, 20 steps, 6 s pause",
            "scan_type": "2D scan: salinity, oil"
        },
        "n_expansions": 3,
        "n_approximation": 0,
        "number_of_oil_points": 6,
        "oil_volume": 3000,
        "scan_type": "linear"
    }
}
```

- __`json/c_surfactant_stock`__: is surfactant concentration in stock oil; 
- __`json/c_surfactant_experiment`__: is surfactant concentration needed for experiment
- __`json/SOLUTION_REPOSITORY_PATH`__: specifies solution repository to use; stock solution have to be defined (or will be created) in this repository
- __`json/stocks`__: contain names of solutions from solution repository. First 6 solutions have to be defined before executing any of scripts, last two (run_stock_surf_oil_1/2) are created by dilution of general stocks:
    - __`surfactant_in_oil_1`__: a name for main stock in oil 1 (heptane)
    - __`surfactant_in_oil_2`__: a name for main stock in oil 2 (hexadecane)
    - __`oil_1`__: a name for oil 1 (heptane)
    - __`oil_2`__: a name for oil 2 (hexadecane)
    - __`stock_aqueous_1`__: a name for aqueous stock with low NaCl concentration (typically water)
    - __`stock_aqueous_2`__: a name for aqueous stock with high NaCl concentration 
    - __`run_stock_surf_oil_1`__: a name for running stock in oil 1
    - __`run_stock_surf_oil_2`__: a name for running stock in oil 2
- __`json/configurations`__: contain names of configurations. 
    - __`json/configurations/blank`__: a blank configuration of an opentron
    - __`json/configurations/start`__: an opentron configuration at the beginning of HLD scan
    - __`json/configurations/end`__: an opentron configuration written after HLD scan
    - __`json/configurations/to_reuse/name`__: an opentron configuration from which well content is imported into starting configuration
    - __`json/configurations/to_reuse/content`__: specifies wells which have to be added to starting configuration
    - __`json/configurations/blank`__: and json/configurations/to_reuse/name have to exist before running any of scripts
    - __`json/configurations/start&end`__: are created in the process
- __`json/scan/experiment_metadata`__: metadata fields for user to clarify scan
- __`json/scan`__: contains parameters for 2D HLD scan; all parameters have to be specified:
    - __`json/scan/n_expansions`__: number of expansion to perform in a 1D salinity scan
        - !!!__add explanation for linear, log modes__!!!
    - __`json/scan/n_approximation`__: not used at the moment, leave at 0
    - __`json/scan/number_of_oil_points`__: number of different oil mixtures requested; 1st point is neat oil 1; last point is neat oil 2; rest are mixtures with linearly increased amount of oil 2
    - __`json/scan/oil_volume`__: volume of oil to dispense in cuvette (in mkL)
    - __`json/scan/scan_type`__: specifies algorithm of 1d point selection:
        - linear:
        - log:
        - log/???: 
    


---


