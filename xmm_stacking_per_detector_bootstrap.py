"""
XMM-Newton galaxy stacking analysis with per-detector loading and bootstrap direction test.

Changes from the original comb-based code:
  1. Three detector switches (use_mos1, use_mos2, use_pn) replace the single comb file.
     Files are loaded as mos1S001-*, mos2S002-*, pnS003-* from each galaxy's data_dir.
     The mask file is shared across all detectors.
  2. Direction test uses bootstrap (n_boot draws from 5 galaxies x 3 detectors = 15 samples)
     instead of exhaustive combinations_with_replacement over 5 comb profiles.
"""
