# Step 11 split (MCT-LTDiag + PLC-CECT)

Patients: 878; train 614, val 88, test 176 (69.9 %, 10.0 %, 20.0 %).
Strata used: L0 23, L1 1, L2 1, L3 8

## Patients by tumour type

| Dataset | diagnosis | train | val | test |
|---|---|---|---|---|
| MCT-LTDiag | BCLM | 80 | 12 | 23 |
| MCT-LTDiag | CRLM | 72 | 10 | 21 |
| MCT-LTDiag | HCC | 73 | 10 | 20 |
| MCT-LTDiag | HH | 67 | 10 | 19 |
| MCT-LTDiag | ICC | 70 | 10 | 20 |
| PLC-CECT | CHCC | 60 | 8 | 17 |
| PLC-CECT | CN | 57 | 9 | 17 |
| PLC-CECT | HCC | 66 | 9 | 19 |
| PLC-CECT | ICC | 69 | 10 | 20 |

## Patients by lesion count

| Dataset | lesion_count_group | train | val | test |
|---|---|---|---|---|
| MCT-LTDiag | 1 | 214 | 29 | 60 |
| MCT-LTDiag | 2+ | 148 | 23 | 43 |
| PLC-CECT | 0 | 57 | 9 | 17 |
| PLC-CECT | 1 | 171 | 23 | 50 |
| PLC-CECT | 2+ | 24 | 4 | 6 |

## Patients by largest-lesion size bin

| Dataset | max_diameter_bin | train | val | test |
|---|---|---|---|---|
| MCT-LTDiag | 10-19 | 34 | 5 | 7 |
| MCT-LTDiag | 20-29 | 57 | 7 | 19 |
| MCT-LTDiag | <10 | 2 | 1 | 1 |
| MCT-LTDiag | >=30 | 269 | 39 | 76 |
| PLC-CECT | 10-19 | 4 | 0 | 1 |
| PLC-CECT | 20-29 | 12 | 0 | 6 |
| PLC-CECT | >=30 | 179 | 27 | 49 |
| PLC-CECT | none | 57 | 9 | 17 |

## Lesions in the release by class (1 malignant, 2 haemangioma)

| Dataset | class | train | val | test |
|---|---|---|---|---|
| MCT-LTDiag | 1 | 686 | 94 | 223 |
| MCT-LTDiag | 2 | 170 | 19 | 40 |
| PLC-CECT | 1 | 248 | 32 | 66 |

## Malignant lesions by in-plane size (native scan)

| Size | train | val | test |
|---|---|---|---|
| <10 | 71 | 14 | 26 |
| 10-19 | 239 | 26 | 78 |
| 20-29 | 177 | 20 | 53 |
| >=30 | 447 | 66 | 132 |

Plan v2 < 10 mm rule: internal test holds 26 malignant lesions < 10 mm (below the 50 needed: the stratum is merged into < 20 mm for the internal test).

Every tumour type of each dataset in every split: yes
