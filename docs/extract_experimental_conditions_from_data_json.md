---
title: 'help for `extract_experimental_conditions_from_data_json.py`'
date: 2026-11-20
---

The script `extract_experimental_conditions_from_data_json.py` extracts inner and outer solution composition from a selected data.json file and prints out as markdown table(s):

- a general table with all available parameters (scan, measurement, densities, component concentrations)
- single component tables for each available variable

# execution

```
python extract_experimental_conditions_from_data_json.py [--settings_file <path-to-settings-file>] [--weight_fraction] [--output_file <path-to-output-file>] source
```

- source: source of data.json file(s); can be a path to file, a path to file containing list of pathes or a folder
    - if settings_file option is used, source is ignored, but must be present in command, use `""`.
- settings_file: a path to settings file: default value, ``
    - if specified dialog opens to select a data.json file from experiment specified in the settings file
- weight_fraction: if present, a weight fraction for each component is reported, otherwise m/v concentration is used in units g/ml.
- output_file: a path to an output file where composition data is saved, default value, `` (file is saved @`<path_to_data.json>/data.json.components.csv`)

- for example, see `HLD_scans_v4/command_prompt_bits.md`
