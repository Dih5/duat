.. _structure:
# Config file structure in duat
By running
```python
from duat import config
print(config.ConfigFile.get_structure())
```
you can see an updated description of how the Config file is described in duat.
The output is the following:

- simulation (Section)
- node_conf (Section)
- grid (Section)
- time_step (Section)
- restart (Section)
- space (Section)
- time (Section)
- el_mag_fld (Section)
- emf_bound (Section)
- smooth (Section)
- diag_emf (Section)
- particles (Section)
- species_list (SpeciesList)
  - 0 (Species)
    - species (Section)
    - profile (Section)
    - spe_bound (Section)
    - diag_species (Section)
  - 1 ...
- cathode_list (SpeciesList)
  - 0 (Species)
    - species (Section)
    - profile (Section)
    - spe_bound (Section)
    - diag_species (Section)
  - 1 ...
- neutral_list (NeutralList)
  - 0 (Neutral)
    - neutral (Section)
    - profile (Section)
    - diag_neutral (Section)
    - species (Section)
    - spe_bound (Section)
    - diag_species (Section)
  - 1 ...
- neutral_mov_ions_list (NeutralMovIonsList)
  - 0 (Section)
  - 1 ...
- collisions (Section)
- zpulse_list (ZpulseList)
  - 0 (Section)
  - 1 ...
- current (Section)
- smooth_current (SmoothCurrent)
