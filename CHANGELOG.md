# pyReCo's Change Log

Noteworthy changes of new releases of pyReCo have been documented in this file since version
0.0.2.

## Unreleased


### Breaking Changes

* Reservoir edge direction is now consistent between the dynamics and the graph
  analysis: `weights[u, v]` is the edge `u -> v`, and the state update is
  `r(t+1) = g(r(t) @ weights + ...)`. Previously the update used `weights @ r(t)`,
  so edge `(u, v)` transmitted activity `v -> u`. Trajectories for a fixed
  reservoir matrix are now those of its transpose; results are statistically
  equivalent but not bitwise reproducible against earlier versions.


### Minor Improvements

* Prepared release metadata updates for `v2.1.0`.
* Confirmed automated publish flow via GitHub Release (`published`) to PyPI.


### Bugfixes

* None


## pyReCo-0.0.3

### Improvements

* Improvements go here


## pyReCo-0.1.0

### New Features

* New features go here

### Improvements

* Improvements go here

### Bugfixes

* Bugfixes go here


## pyReCo-1.0.0

### Breaking Changes

* Breaking changes go here


### New Features

* New features go here


### Improvements

* Improvements go here

### Bugfixes

* Bugfixes go here
