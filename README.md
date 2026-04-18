# Night-safety
From Wills to Temple meads
weights:
  alpha: 1.3 
  beta:  1.1 
  cctv_weight: 3
  light_weight: 1
========== ROUTE SUMMARY ==========
  A  Distance only:             2355m  |  31 min
  B  Distance + Crime:          2490m  |  33 min
  C  Distance + Crime + Safety: 2646m  |  35 min
====================================


========== SAFETY ANALYSIS ==========
Metric                      A (shortest)     B (+crime)       C (full)
----------------------------------------------------------------------
Distance                          2355m         2490m         2646m
Lights per km                      20.4          18.5          27.2
CCTV per km                         0.4           0.4           3.8
Lit coverage                      54.9%         47.5%         65.5%
Dark segments                     1063m         1308m          914m
Avg crime risk                    0.343         0.269         0.309
Max crime risk                    0.883         0.883         0.634

  Route C vs Route A:
    Detour:            12.4%
    Lights per km:   +33.5%
    CCTV per km:     +789.8%
    Lit coverage:    +19.3%
    Dark segments:   -16.3%
    Avg crime risk:  -11.3%
=====================================
