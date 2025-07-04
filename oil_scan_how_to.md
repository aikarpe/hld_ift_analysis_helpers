---
title: 'oil scan: how to'
date: 2025-06-27
---

1. evaluate surfactant solubility in oils to be used for scan

2. plan preparation of stock solution 
    - __ref_to_script_for_estimates__

3. make stock(s) and add their representation to a solution repository

3.5. make run configuration

4. make run stock (and distribute washing solution) 

5. prepare run configuration

6. execute run

---

## surfactant solubility in oil

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

## stock solution preparation

### planning

Providing that surfactant is going to dissolve in a chosen oil, recipe(s) for stock(s) can be generated using `generate_recepies_for_solutions.py`. The script will optimize a binary mixture from solutions available in repository

Script will require some tweaking:

- repository to use
- (optional) add components not in repository
- referencing solutions to be used
- recipe call(s)
- my (ap) call sequence of edited script in command prompt:


---


```
cd /D C:\Users\Aigar\miniconda3\
set LOC_SRC="D:\temp_data\scripts"
python %LOC_SRC%\\generate_recepies_for_solutions.py
```


---

More details on tweaks are below.

- specify solution repository location (if different from what is already in the script). Change a line:

```
SOLUTION_REPOSITORY_PATH = "D:/temp_data/solution_repository.json"
```

- mixtures can be made from solutions(compounds) that are stored in repository, a new component can be added by providing relevant information as dictionary(s) and adding to repository at runtime. All definitions added to `rep_init` are added to repository. In the example below `idrosal_sxs40_dict` is added, but `ib_45` ignored. See code below.

```
# =========================================================
# add new compounds/solutions, if needed
# =========================================================
ib_45  = {
    "mixture_type": "binary_mix",
    "name": "44perc_IB-45_in_wt",
    "m_solute":   0.44,
    "m_solvent":  1.0 - 0.44,
    "ro":         1.12,
    "solvent":   "water",
    "solute":    "aerosol_ib-45",
    "v_ro":       1.0,
    "m_ro_water": 1.0,
    "m_ro":       1.12,
    "date":       "2025-03-02",
  }
idrosal_sxs40_dict = {
    "mixture_type": "pure_compound",
    "ro": 1.17,
    "name": "idrosal_sxs40",
    "label": "idrosal_sxs40",
    "cas": "1300-72-7",
    "alt_name": "Sodium Xylenesulfonate",
    "note": "idrosal_sxs40 is name for 40% solution of sodium xylenesulfonate"
}

rep_init = [idrosal_sxs40_dict]
for it in rep_init:
    rep.add_item(it)
```

- you have to explicitly reference solution(s) from repository to be used:

```
# =========================================================
# references to solutions
# solution names and component names look up in solution repository!!!
# =========================================================

aot = rep.items["aot"]
heptane = rep.items["heptane"]
hexadecane = rep.items["hexadecane"]
water = rep.items["water"]
ib_45 = rep.items["44perc_IB-45_in_wt"]
nacl_sol = rep.items["30g_NaCl_in_100mL_water"]
idrosal_sxs40 = rep.items["idrosal_sxs40"]
sdbs = rep.items["sdbs"]
```

- finally actual recipe is generated by calling `create_recepie_for()`:

```
# =========================================================
# recepies needed
# =========================================================

print("============================= sdbs in di water")
create_recepie_for(sdbs, water, "sdbs", 0.2, "m/v", 50) 
print("============================= idrosal_sxs40 in di water")
create_recepie_for(idrosal_sxs40, water, "idrosal_sxs40", 0.2, "m/v", 50) 
print("============================= ib-45 in wt")
create_recepie_for(ib_45, water, "aerosol_ib-45", 0.2, "m/v", 50) 
```

- description of create_recepie_for():
    - call: create_recepie_for(sol1, sol2, component, target, concentration_type, amount_needed = 1, method = "Nelder-Mead")
    - arguments:
        Solution sol1: a solution to use
        Solution sol2: a solution to use
        str component: a name of component for which target concentration is specified
        str concentration_type: {"m/m": mass to mass of solution, "m/v": mass of component to volume of solution}, any other string defaults to "m/m"
        float amount_needed: mass of solution needed in grams
    - returns: Solution or None, if recipe fails

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
 
