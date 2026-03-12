# cs15 — Fortran/Python Verlet Simulations

Physics simulations using the Velocity Verlet integration method, built to compare a **Fortran-compiled** physics engine against a **pure Python/NumPy** implementation.

## Project Structure

```
cs15-fortran-python-verlet/
├── src/                        # Fortran source files
│   ├── physics_engine.f90      # Core physics routines (verlet, collisions, gravity, constraints)
│   ├── adder.f90               # Scratch / learning file
│   └── hello_word.f90          # Scratch / learning file
│
├── simulations/
│   ├── fortran/                # Simulations using the compiled Fortran extension
│   │   ├── chain.py
│   │   ├── emitted_particles.py
│   │   ├── orbit.py
│   │   └── particles.py
│   └── pure/                   # Identical simulations using pure Python / NumPy only
│       ├── chain.py
│       ├── emitted_particles.py
│       ├── orbit.py
│       └── particles.py
│
├── build/                      # Compiled artifacts (Windows, Python 3.12)
│   ├── phys_engine.cp312-win_amd64.pyd
│   ├── adder.cp312-win_amd64.pyd
│   ├── libgcc_s_seh_64-1.dll
│   ├── libgfortran_64-5.dll
│   ├── libquadmath_64-0.dll
│   ├── libwinpthread_64-1.dll
│   └── physics_engine/.libs/   # Additional GFortran runtime DLLs
│
├── .vscode/settings.json       # Pins Python 3.12 interpreter
├── .gitignore
└── README.md
```

## Simulations

| Simulation | Description |
|---|---|
| `particles` | 5 000 coloured particles with gravity and elastic collisions |
| `emitted_particles` | Particles spawned from a point, accumulating up to 2 000 |
| `orbit` | 100 planets orbiting a central sun under Newtonian gravity |
| `chain` | Mouse-draggable rope chain with distance constraints |

## Running

```bash
# Fortran-accelerated
python simulations/fortran/particles.py
python simulations/fortran/orbit.py
python simulations/fortran/emitted_particles.py
python simulations/fortran/chain.py

# Pure Python / NumPy
python simulations/pure/particles.py
python simulations/pure/orbit.py
python simulations/pure/emitted_particles.py
python simulations/pure/chain.py
```

**Requirements:** Python 3.12, `pygame`, `numpy`

```bash
pip install pygame numpy
```

## Physics Engine (`src/physics_engine.f90`)

| Routine | Purpose |
|---|---|
| `verlet_step` | Velocity Verlet integration — updates positions and velocities |
| `resolve_collisions` | O(n²) elastic collision response between equal-mass particles |
| `apply_constraints` | Iterative distance constraint for chain links |
| `apply_orbital_gravity` | Newtonian gravity toward a fixed sun |

## Rebuilding the Extension

```bash
cd src
python -m numpy.f2py -c physics_engine.f90 -m phys_engine
# then move the resulting .pyd and any .dll files into build/
```
