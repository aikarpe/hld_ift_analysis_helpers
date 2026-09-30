---
title: "new_profile_wizard.py help"
date: 2026-05-06
---




```{mermaid new_profile_wizard.py-steps}
flowchart TD
  source(source folder) --> use_def_sol_rep(use default solution repository?)
  use_def_sol_rep -->|No| source_sol_rep(choose solution repository)

  source_sol_rep --> target(choose target folder)
  use_def_sol_rep -->|Yes| target

  subgraph copy [make/copy]
    direction LR
    c1(make folder)
    c1 --> c2(copy settings)
    c2 --> c3(copy command prompt bits)
    c3 --> c4(copy solution repository)
    c4 --> c5(copy blank configurations)
  end

  target --> copy

  subgraph cmd_prompt_edits [command prompt bit edits]
    ce1(change working directory)
  end

  copy --> cmd_prompt_edits
  
  cmd_prompt_edits --> settings_edits(edit settings files)
```

