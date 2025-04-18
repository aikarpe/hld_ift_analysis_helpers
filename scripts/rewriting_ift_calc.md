---
title: 'notes on rewriting ift calculation'
date: 
---

1. find where needle is, use region Nx of needle width (N = 2)
2. find where needle ends (vertical pos)
3. for needle region only: get edges as lines
    - dia_needle_px: distance between edges
    - scale: dia_needle / dia_needle_px
    - best perpendicular line to edges
4. get contour at the tip: get outer bits
