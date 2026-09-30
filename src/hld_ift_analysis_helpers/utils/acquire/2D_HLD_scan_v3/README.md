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


Setup requires:



- surfactant_in_oil_1[^note_1]: main stock of oil_1 with surfactant_oil
- surfactant_in_oil_2: main stock of oil_2 with surfactant_oil
- stock_aqueous_1: main stock of aqueous_solution_1 with surfactant_aq
- stock_aqueous_2: main stock of aqueous_solution_2 with surfactant_aq

[^note_1]: [variable name in `scan_settings.json`]: [description]
