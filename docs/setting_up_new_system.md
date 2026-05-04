---
title: 'setting up new pc for hld-ift'
date: 2026-05-04
---

All instructions (except for part of OT-2 software) are meant to be installed on a pc interacting with OT-2!

# perparing laptop for hld-ift

## opentron software

- install opentrons app: version 8.2.0 (program for a pc)
- manually install matching version of software on OT-2 itself!!!

## camea control software

- install pylon
https://www.baslerweb.com/en/downloads/software/?downloadCategory.values.label.data=pylon

## install git

- see their website: https://git-scm.com/install/windows

---

## prepare miniconda

- NOTE: all commands given in this subsection are meant to be executed in Anaconda prompt!!!

- install anaconda (miniconda), see https://www.anaconda.com/download
- create new environment,
- activate the environment,
- and install python there:

```{anaconda prompt}
conda create -n hld_ift0 python=3.13
conda activate hld_ift0
conda install pip`
```

---

## clone packages using git

- commands in this section are for `git bash` terminal!!!

- create folder where to store packages (i'm using `~/code` for example)

- to clone package in `git bash` terminal run following:

```{git bash}
cd ~/code
git clone http://<token_string>@github.com/aikarpe/hld_ift_http.git 
git clone http://<token_string>@github.com/aikarpe/hld_ift_analysis_helpers.git 
```
where `<token_string>` should be substituted with token that gives you necessary permissions. They look like ugly long strings:

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

---


## install hld_ift packages

- NOTE: all commands given in this subsection are meant to be executed in Anaconda prompt!!!

- install packages:

```{anaconda prompt}
pip install C:\Users\Admin\code\hld_ift_http
pip install C:\Users\Admin\code\hld_ift_analysis_helpers
```







