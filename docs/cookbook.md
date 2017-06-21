# duat cookbook

## Create a OSIRIS config file
```python
from duat import config
sim = config.ConfigFile(1)  # 1D
# Use index notation to modify or create parameters and sections
sim["grid"]["nx_p"] = 4000
# Check the generated fortran code
print(sim)
```

## Run a config file
```python
from duat import run
# [...]
# Create a ConfigFile instance sim
myrun = run.run_config(sim, "/home/user/myrun", prefix="", blocking=False, clean_dir=True)
# The returned Run instance offers some useful info. Check the API documentation
```

## Run a variation
```python
from duat import config, run
# [...]
# Create a ConfigFile instance sim
parameter = ["zpulse_list", 0, "a0"] # Parameter to change
values = [0.2, 0.5, 0.7, 1.0, 1.2] # Values to take
var = config.Variation((parameter, values)) # Add more 2-tuples for cartesian product of parameter variation
run_list = run.run_variation(sim, var, "/home/user/myvar", caller=3) # Create three threads executing simulations
```

## Plot results with matplotlib
```python
# [...]
# Obtain a Run instance named myrun
# Diagnostic objects can be used even if the run is in progress
diagnostics = myrun.get_diagnostic_list()
for d in diagnostics:
    print(d)
# Suppose diagnostics[0] is 1D. Then:
fig, ax = diagnostics[0].time_1d_colormap()
# Now export, show, customize or whatever
```