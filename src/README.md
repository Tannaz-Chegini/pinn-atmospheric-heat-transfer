# Source modules

These files expose the main reusable model, physics, training, and evaluation code in a conventional `src/` layout.

The function and class bodies were extracted from the executed study notebooks rather than rewritten. The common core (`AnchoredCoordinateModel`, `AnchoredForcingModel`, `HeatingClosure`, `pde_residual`, training routines, and common evaluation helpers) was verified to be identical across all seven uploaded study notebooks.

The modules intentionally do **not** replace the notebooks as runnable experiment entry points. Several original functions depend on experiment-specific globals such as domain coordinates, pressure levels, training hyperparameters, and device selection. Those values differ by experiment and remain exactly documented in each executed notebook. Refactoring them into a fully standalone package would change function signatures and would require a fresh validation run.
