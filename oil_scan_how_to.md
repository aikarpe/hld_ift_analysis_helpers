---
title: 'oil scan: how to'
date: 2025-06-27
---

1. evaluate surfactant solubility in oils to be used for scan
2. plan preparation of stock solution 
3. make stock(s) and add their representation to a solution repository
4. make run configuration
5. make run stock (and distribute washing solution) 
6. prepare run configuration
7. check camera
8. execute run

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

## stock solution preparation (step 2 and 3)

- in the context of an oil scan there are 3 parts relevant to stock solutions:
    - planning: what components and in what quantities to mix
    - actual preparation of stock solution
    - making of stock solution representation for oil scan(s) and adding to a solution repository

### planning

- Providing that surfactant is going to dissolve in a chosen oil, recipe(s) for stock(s) can be generated using `generate_recepies_for_solutions_v2.py`. The script will optimize a binary mixture from solutions (compounds) available in repository. Recipes are estimated from ingredient densities.
- to generate call execute command like:

```
python generate_recepies_for_solutions_v2.py <recipe_file>
```
- recipes needed should be specified in recipe_file in `recepies` section; it can contain multiple recipes targets
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

- edit <recipe_file>  to incorporate all prepared stock solutions:
    - use previously noted values of m_surfactant, m_solvent, v_ro, m_ro_water, m_ro to create dictionary for binary mixture
- run script:

```
python solution_repository__edit.py <recipe_file> 
```


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

- solution_repository_path: string, a path to a solution repository file; it is needed for recipe generation and to add stock solution representation
- compounds_to_add: a list of dictionaries of binary mixture or pure compound definitions; this section is used when solution(s) are added to a solution repository
- recepies: a list of dictionaries specifying recipe requests; this section is to generate recipes
- dictionary to define binary mixtures should have following keys:
    - mixture_type: string, should be set to "binary_mix"
    - name: a name of this mixture, will be used to refer to it in a solution repository
    - m_solute, m_solvent: a mass of each component in grams
    - ro: numeric, a density in g/ml, if this key is present values in keys: v_ro, m_ro_water and m_ro are ignored
    - solvent, solute: names of solutions as specified in a solution repository
    - v_ro: numeric, a volume of liquid(s) in mL used to measure density
    - m_ro_water: a mass of water in grams in reference measurement
    - m_ro: a mass of a mixture in grams in density measurement; density of a mixture is evaluated as `m_ro/m_ro_water`; the evaluation is skipped if key `ro` is defined
    - other keys can be provided, but are ignored
- dictionary to define pure compound should have following keys:
    - mixture_type: string, should be set to "pure_compound"
    - name, label: a name of this compound, will be used to refer to it in a solution repository; label and name should be the same!!!
    - ro: numeric, a density in g/ml, if this key is present values in keys: v_ro, m_ro_water and m_ro are ignored
    - v_ro: numeric, a volume of liquid(s) in mL used to measure density
    - m_ro_water: a mass of water in grams in reference measurement
    - m_ro: a mass of a compound in grams in density measurement; density of compound is evaluated as `m_ro/m_ro_water`; the evaluation is skipped if key `ro` is defined
    - other keys can be provided, but are ignored
- dictionary to define recipe requests:
    - solution1, solution2: string, a name of solutions (compounds) to be used to generate recipe; names of solutions should be defined in a selected solution repository
    - component_to_target: string, a name of a component which concentration is to be targeted
    - target_concentration: numeric, concentration in g/g (m/m) or g/ml (m/v) units
    - concentration_type: a string, mass ("m/m") or mass to volume concentration ("m/v")
    - quantity: numeric, quantity of final solution needed in grams


---

### stock preparation and density measurement

- make stock according to recipe by dissolving m_surfactant g of surfactant in m_solvent g of solvent
- add 5 ml (v_ro) of di water in 5 mL glass vial, note mass (m_ro_water)
- add 5 ml of stock in 5 ml vial, note mass (m_ro)

- edit `???repository_init.py???` script by adding dictionary and execute it
- name could be anything as long as it describes well your stock; it will be used to reference stock in other scripts


```
a_dict_name = { # replace variables with actual values
    "mixture_type": "binary_mix",
    "name": "super_duper_stock_solution_20250627",
    "m_solute":   m_surfactant,
    "m_solvent":  m_solvent,
    "solvent":   "water",
    "solute":    "sdbs",
    "v_ro":       v_ro,
    "m_ro_water": m_ro_water,
    "m_ro":       m_ro,
    "date":       "2025-06-27",
}
```

##  run configuration

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

- json/c_surfactant_stock is surfactant concentration in stock oil; json/c_surfactant_experiment is surfactant concentration needed for experiment
- json/SOLUTION_REPOSITORY_PATH specifies solution repository to use; stock solution have to be defined (or will be created) in this repository
- json/stocks contain names of solutions from solution repository. First 6 solutions have to be defined before executing any of scripts, last two (run_stock_surf_oil_1/2) are created by dilution of general stocks:
    - surfactant_in_oil_1: a name for main stock in oil 1 (heptane)
    - surfactant_in_oil_2: a name for main stock in oil 2 (hexadecane)
    - oil_1: a name for oil 1 (heptane)
    - oil_2: a name for oil 2 (hexadecane)
    - stock_aqueous_1: a name for aqueous stock with low NaCl concentration (typically water)
    - stock_aqueous_2: a name for aqueous stock with high NaCl concentration 
    - run_stock_surf_oil_1: a name for running stock in oil 1
    - run_stock_surf_oil_2: a name for running stock in oil 2
- json/configurations contain names of configurations. 
    - json/configurations/blank: a blank configuration of an opentron
    - json/configurations/start: an opentron configuration at the beginning of HLD scan
    - json/configurations/end: an opentron configuration written after HLD scan
    - json/configurations/to_reuse/name: an opentron configuration from which well content is imported into starting configuration
    - json/configurations/to_reuse/content: specifies wells which have to be added to starting configuration
    - json/configurations/blank and json/configurations/to_reuse/name have to exist before running any of scripts
    - json/configurations/start&end are created in the process
- json/scan/experiment_metadata: metadata fields for user to clarify scan
- json/scan contains parameters for 2D HLD scan; all parameters have to be specified:
    -json/scan/n_expansions: number of expansion to perform in a 1D salinity scan
        - __add explanation for linear, log modes__
    - json/scan/n_approximation: not used at the moment, leave at 0
    - json/scan/number_of_oil_points: number of different oil mixtures requested; 1st point is neat oil 1; last point is neat oil 2; rest are mixtures with linearly increased amount of oil 2
    - json/scan/oil_volume: volume of oil to dispense in cuvette (in mkL)
    - json/scan/scan_type: specifies algorithm of 1d point selection:
        - linear:
        - log:
        - log/???: 
    
    

# for solution prep

| slot 9 | col 1                                                       | col 2                                            | col 3                             | col 4                              | col 5                       |
|:-------|:-------------------------------------------------------|:--------------------------------------------|:-----------------------------|:------------------------------|:-----------------------|
| A      | A1<br>stock of surfactant in C7<br> 12000 /             | A2<br>stock of surfactant in C16<br> 12000 / | A3<br>stock of C7<br> 12000 / | A4<br>stock of C16<br> 12000 / | A5<br>water<br> 12000 / |
| B      | B1<br>run stock of surfactant in C7<br>     0 /  12500  | B2<br><br> /                                 | B3<br><br> /                  | B4<br><br> /                   | B5<br><br> /            |
| C      | C1<br>run stock of surfactant in C16<br>     0 /  12500 | C2<br><br> /                                 | C3<br><br> /                  | C4<br><br> /                   | C5<br><br> /            |



|slot 2 |1                                            |2                                           |3                                          |4                                           |
|:------|:--------------------------------------------|:-------------------------------------------|:------------------------------------------|:-------------------------------------------|
|A      |A1<br>unused sample waste<br>     0 /    500 |A2<br>sample rinse waste<br>     0 /    500 |A3<br>first rinse waste<br>     0 /    500 |A4<br>first rinse source<br>     0 /   1400 |

# for 2d scan

|slot 9 |1                       |2                                        |3                                               |4                                                |
|:------|:-----------------------|:----------------------------------------|:-----------------------------------------------|:------------------------------------------------|
|A      |A1<br>water<br> 12000 / |A2<br>stock of NaCl solution<br> 12000 / |A3<br>run stock of surfactant in C7<br> 12000 / |A4<br>run stock of surfactant in C16<br> 12000 / |



|slot 2 |1                                     |2                                    |3                                   |4                                    |
|:------|:-------------------------------------|:------------------------------------|:-----------------------------------|:------------------------------------|
|A      |A1<br>unused sample waste<br>   500 / |A2<br>sample rinse waste<br>   500 / |A3<br>first rinse waste<br>   500 / |A4<br>first rinse source<br>  1400 / |
 
