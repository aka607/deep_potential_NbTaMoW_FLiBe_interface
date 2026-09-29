# deep_potential_NbTaMoW_FLiBe_interface
A collection of .ipynb scripts for Jupyter Notebook for training a potential energy model from ab initio molecular dynamics simulations using the deep learning package Deepmd kit

## Install DeePMD-kit
See https://docs.deepmodeling.com/projects/deepmd/en/v2.2.3/getting-started/install.html for installation instructions.

> **Note:** VASP simulation data can be downloaded from Zenodo at https://doi.org/10.5281/zenodo.15699534 for preparing the training and the testing data from molecular dynamics trajectories for the NbTaMoW_FLiBe interface.

## Preparing the training and testing data

The training data used by DeePMD-kit comprises the atom type, simulation box,
atom coordinates, atom forces, system energy, and virial. `data/data_preparation.ipynb`
converts a VASP AIMD trajectory (`OUTCAR_10ps`) into DeePMD `npy` systems and
splits it into a training set and a validation (testing) set.

**1. Load the trajectory and split into training / validation sets:**

```python
import dpdata
import numpy as np

# Load the labeled system (types, box, coords, energy, forces, virials) from the VASP OUTCAR.
system = dpdata.LabeledSystem("./OUTCAR_10ps", fmt="vasp/outcar")
nframes = len(system)
print(f"# the system contains {len(system)} frames")

# Randomly hold out 50 frames as the validation (testing) set; the rest are training.
rng = np.random.default_rng()
index_validation = rng.choice(nframes, size=50, replace=False)
index_training = list(set(range(nframes)) - set(index_validation))
data_training = system.sub_system(index_training)
data_validation = system.sub_system(index_validation)

# Write each set out in DeePMD npy format.
data_training.to_deepmd_npy("salt-alloy/data/training_data")
data_validation.to_deepmd_npy("salt-alloy/data/validation_data")

print(f"# the training data contains {len(data_training)} frames")
print(f"# the validation data contains {len(data_validation)} frames")
```

```
# the system contains 463 frames
# the training data contains 413 frames
# the validation data contains 50 frames
```

**2. Inspect the atom-type mapping (`type_map.raw`):**

```python
! cat salt-alloy/data/training_data/type_map.raw
```

```
Li
Be
F
Mo
Nb
Ta
W
```

**3. Count the number of atoms of each type:**

```python
types, counts = np.unique(np.loadtxt("salt-alloy/data/training_data/type.raw", dtype=int), return_counts=True)
for t, c in zip(types, counts):
    print(f"type {t}: {c}")
```

```
type 0: 28   # Li
type 1: 14   # Be
type 2: 56   # F
type 3: 20   # Mo
type 4: 20   # Nb
type 5: 20   # Ta
type 6: 20   # W
```

## Training the potential

The model is trained with `dp train input.json` (see `training/training.ipynb`).
Training progress is recorded in `lcurve.out`; the learning curve below plots the
energy and force RMSE for the training and validation sets against the number of
training steps:

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

with open("lcurve.out") as f:
    headers = f.readline().split()[1:]
lcurve = pd.DataFrame(np.loadtxt("lcurve.out"), columns=headers)
legends = ["rmse_e_val", "rmse_e_trn", "rmse_f_val", "rmse_f_trn"]
for legend in legends:
    plt.loglog(lcurve["step"], lcurve[legend], label=legend)
plt.legend()
plt.xlabel("Training steps")
plt.ylabel("Loss")
plt.show()
```

![Training learning curve](training/lcurve.png)

## Validating the potential

After freezing the model to `graph.pb`, use it to predict energies for the
training set and compare them against the DFT reference. A tight cluster along
the `y = x` line indicates good agreement between the deep potential and DFT:

```python
import dpdata

training_systems = dpdata.LabeledSystem("../data/training_data", fmt="deepmd/npy")
predict = training_systems.predict("graph.pb")
```

```python
import matplotlib.pyplot as plt
import numpy as np

plt.scatter(training_systems["energies"], predict["energies"])

x_range = np.linspace(plt.xlim()[0], plt.xlim()[1])

plt.plot(x_range, x_range, "r--", linewidth=0.25)
plt.xlabel("Energy of DFT")
plt.ylabel("Energy predicted by deep potential")
plt.plot()
```

![DFT vs deep-potential energy parity](training/energy_parity.png)
