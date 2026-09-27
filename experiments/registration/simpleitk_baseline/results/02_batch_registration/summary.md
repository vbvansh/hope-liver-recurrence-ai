# HCC-TACE-Seg registration summary (91 patients, 40 skipped/failed)

Scores inside the healthy liver (tumour excluded). Values: median [25th-75th percentile].

| Level | MAE (HU), lower = better | NCC, higher = better | PSNR (dB), higher = better | Liver coverage (%) |
|---|---|---|---|---|
| none | 92.61 [67.21-122.36] | 0.03 [-0.02-0.06] | 10.22 [8.15-12.32] | 99.66 [75.34-100.00] |
| rigid | 38.06 [30.36-48.18] | 0.13 [0.07-0.21] | 16.78 [15.12-18.68] | 100.00 [100.00-100.00] |
| deformable | 30.86 [27.39-39.46] | 0.21 [0.14-0.29] | 18.61 [17.23-20.22] | 100.00 [100.00-100.00] |

- Rigid better than none: 83/91 patients
- Deformable better than rigid: 79/91 patients
- Rigid movement needed: shift 44.20 [29.75-123.70] mm, rotation 3.91 [2.75-5.37] deg
- Healthy-liver brightness shift after/before: -0.22 [-9.12-6.77] HU; MAE once that shift is removed: 29.02 [24.86-34.08] HU
- Aorta brightness before 165.00 [132.50-223.00] HU vs after 172.47 [135.15-242.96] HU (similar values = same contrast phase)
- Warp folding inside liver: 0.00 [0.00-0.00] % of voxels (should be 0); tumour region volume factor 1.03 [0.93-1.17] (1 = untouched)
- Tumour median HU before 95.00 [80.50-112.50] vs after 97.62 [76.11-118.11]
- Time per patient: 105.00 [79.00-147.00] s

Best 5 (lowest deformable MAE): HCC_042 (19), HCC_014 (19), HCC_072 (20), HCC_058 (20), HCC_081 (20)
Worst 5 (highest deformable MAE): HCC_027 (53), HCC_088 (53), HCC_018 (55), HCC_017 (84), HCC_039 (91)

Skipped / failed:
- HCC_001: ValueError: no comparable contrast series after TACE for 'C-A-P'
- HCC_009: ValueError: no comparable contrast series after TACE for 'LIVER 3 PHASE CAP'
- HCC_009: ValueError: no comparable contrast series after TACE for 'LIVER 3 PHASE CAP'
- HCC_009: ValueError: no comparable contrast series after TACE for 'LIVER 3 PHASE CAP'
- HCC_021: KeyError: -223.91
- HCC_021: FileNotFoundError: [Errno 2] No such file or directory: '/home/jovyan/Vansh_HA26001/work/HCC_021/before/00000001.dcm'
- HCC_021: RuntimeError: could not download series 1.3.6.1.4.1.14519.5.2.1.1706.8374.104290748697100005420912116573
- HCC_029: ValueError: no comparable contrast series after TACE for 'Recon 3 LIVER 3 PHASE CAP'
- HCC_029: ValueError: no comparable contrast series after TACE for 'Recon 3 LIVER 3 PHASE CAP'
- HCC_029: ValueError: no comparable contrast series after TACE for 'Recon 3 LIVER 3 PHASE CAP'
- HCC_036: FileNotFoundError: [Errno 2] No such file or directory: '/home/jovyan/Vansh_HA26001/work/HCC_036/before/00000001.dcm'
- HCC_036: RuntimeError: could not download series 1.3.6.1.4.1.14519.5.2.1.1706.8374.336005218259978985736646945152
- HCC_036: RuntimeError: could not download series 1.3.6.1.4.1.14519.5.2.1.1706.8374.336005218259978985736646945152
- HCC_040: FileNotFoundError: [Errno 2] No such file or directory: '/home/jovyan/Vansh_HA26001/work/HCC_040/before/00000001.dcm'
- HCC_040: FileNotFoundError: [Errno 2] No such file or directory: '/home/jovyan/Vansh_HA26001/work/HCC_040/seg/00000001.dcm'
- HCC_040: RuntimeError: could not download series 1.3.6.1.4.1.14519.5.2.1.1706.8374.136107204035054365392640033610
- HCC_048: ValueError: segmentation does not say which CT series it was drawn on
- HCC_048: ValueError: segmentation does not say which CT series it was drawn on
- HCC_048: ValueError: segmentation does not say which CT series it was drawn on
- HCC_068: ValueError: no comparable contrast series after TACE for '2.5 SOFT'
- HCC_068: ValueError: no comparable contrast series after TACE for '2.5 SOFT'
- HCC_068: ValueError: no comparable contrast series after TACE for '2.5 SOFT'
- HCC_085: ValueError: no comparable contrast series after TACE for 'Recon 3 LIVER 2PHASE CAP'
- HCC_085: ValueError: no comparable contrast series after TACE for 'Recon 3 LIVER 2PHASE CAP'
- HCC_085: ValueError: no comparable contrast series after TACE for 'Recon 3 LIVER 2PHASE CAP'
- HCC_089: ValueError: no comparable contrast series after TACE for 'Recon 3 LIVER 2PHASE CAP'
- HCC_089: ValueError: no comparable contrast series after TACE for 'Recon 3 LIVER 2PHASE CAP'
- HCC_089: ValueError: no comparable contrast series after TACE for 'Recon 3 LIVER 2PHASE CAP'
- HCC_093: RuntimeError: could not download series 1.3.6.1.4.1.14519.5.2.1.1706.8374.407713635068455927228285104709
- HCC_093: RuntimeError: could not download series 1.3.6.1.4.1.14519.5.2.1.1706.8374.407713635068455927228285104709
- HCC_093: RuntimeError: could not download series 1.3.6.1.4.1.14519.5.2.1.1706.8374.407713635068455927228285104709
- HCC_095: ValueError: no comparable contrast series after TACE for '2.5 SOFT'
- HCC_095: ValueError: no comparable contrast series after TACE for '2.5 SOFT'
- HCC_095: ValueError: no comparable contrast series after TACE for '2.5 SOFT'
- HCC_098: ValueError: no comparable contrast series after TACE for 'Recon 3 LIVER 2PHASE CAP'
- HCC_099: ValueError: no comparable contrast series after TACE for '2.5 STANDARD'
- HCC_098: ValueError: no comparable contrast series after TACE for 'Recon 3 LIVER 2PHASE CAP'
- HCC_099: ValueError: no comparable contrast series after TACE for '2.5 STANDARD'
- HCC_098: ValueError: no comparable contrast series after TACE for 'Recon 3 LIVER 2PHASE CAP'
- HCC_099: ValueError: no comparable contrast series after TACE for '2.5 STANDARD'