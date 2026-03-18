# Google Summer of Code 2026 Proposal

## Flexible Graph Construction for Data-Driven Weather Models

**Organization:** MLLAM (Machine Learning for Limited Area Models)  
**Project:** [Flexible Graph Construction](https://github.com/mllam/neural-lam/wiki/GSoC-ideas#1-flexible-graph-construction) (Idea #1)  
**Repositories:** [weather-model-graphs](https://github.com/mllam/weather-model-graphs) (wmg), [neural-lam](https://github.com/mllam/neural-lam)  
**Project Length:** 350 hours  
**Difficulty:** Medium  
**Mentors:** Hauke Schulz ([@observingClouds](https://github.com/observingClouds)), Leif Denby ([@leifdenby](https://github.com/leifdenby)), Joel Oskarsson ([@joeloskarsson](https://github.com/joeloskarsson))

---

## Table of Contents

1. [Personal Information](#1-personal-information)
2. [Project Abstract](#2-project-abstract)
3. [Motivation & Problem Statement](#3-motivation--problem-statement)
4. [Alignment with Mentor Priorities & Existing Roadmap](#4-alignment-with-mentor-priorities--existing-roadmap)
5. [Current Architecture Deep-Dive](#5-current-architecture-deep-dive)
6. [Proposed Solution & Technical Design](#6-proposed-solution--technical-design)
7. [Detailed Implementation Plan (Phase-by-Phase)](#7-detailed-implementation-plan-phase-by-phase)
8. [Timeline & Milestones](#8-timeline--milestones)
9. [Testing Strategy](#9-testing-strategy)
10. [Deliverables](#10-deliverables)
11. [Prior Contributions & Community Engagement](#11-prior-contributions--community-engagement)
12. [About Me](#12-about-me)
13. [Related Work & References](#13-related-work--references)
14. [Appendix: Architecture Diagrams & Technical Details](#14-appendix-architecture-diagrams--technical-details)

---

## 1. Personal Information

| Field | Details |
|-------|---------|
| **Name** | Prajwal [Your Last Name] |
| **University** | [Your University] |
| **Degree** | [Your Degree, e.g., B.Tech in Computer Science] |
| **Email** | [your.email@example.com] |
| **GitHub** | [github.com/prajwal-tech07](https://github.com/prajwal-tech07) |
| **LinkedIn** | [linkedin.com/in/yourprofile] |
| **Timezone** | [e.g., UTC+5:30] |
| **Available hours/week** | 30–35 hours |

---

## 2. Project Abstract

The [GSoC ideas page](https://github.com/mllam/neural-lam/wiki/GSoC-ideas#1-flexible-graph-construction) describes the core challenge:

> *"The challenge is to explore and implement a methodology that can create well-balanced neural network grids based on different data structures, from irregularly structured atmospheric model output to sparse ship-observations."*

Currently, both `weather-model-graphs` and `neural-lam` construct graph architectures (Keisler flat, GraphCast multiscale, Oskarsson hierarchical) that **assume regular rectangular grids**. The mesh generation in `weather-model-graphs` ( [`create_single_level_2d_mesh_graph()`](https://github.com/mllam/weather-model-graphs/blob/main/src/weather_model_graphs/create/mesh/mesh.py) ) is hardcoded to use `networkx.grid_2d_graph()`, and `neural-lam`'s [`create_graph_from_datastore()`](https://github.com/mllam/neural-lam/blob/main/neural_lam/create_graph.py) explicitly blocks non-regular datastores with `raise NotImplementedError`.

This project will:

1. **Decouple mesh layout from mesh connectivity** by introducing a `mesh_layout` parameter (Issue [#71](https://github.com/mllam/weather-model-graphs/issues/71)) — separating *how mesh nodes are spatially arranged* from *how mesh levels are connected*
2. **Add a `"prebuilt"` mesh pathway** (Issue [#70](https://github.com/mllam/weather-model-graphs/issues/70)) — allowing users to inject arbitrary mesh topologies (icosahedral, Voronoi, ICON grids, observation networks) directly into `create_all_graph_components()`
3. **Implement Delaunay-based and density-adaptive mesh generators** for irregular point distributions
4. **Address critical wmg v0.4.0 blockers** — convex hull cropping ([#40](https://github.com/mllam/weather-model-graphs/issues/40)), g2m node assertion ([#42](https://github.com/mllam/weather-model-graphs/issues/42)), and level attribute consistency ([#45](https://github.com/mllam/weather-model-graphs/issues/45)) — as identified in Joel's ["Crucial things to merge"](https://github.com/mllam/neural-lam/issues/138) checklist
5. **Replace `create_graph.py` in neural-lam** with a wmg-backed `build_rectangular_graph.py` ([#83](https://github.com/mllam/neural-lam/issues/83)), completing the wmg integration

The end result: a graph construction system that accepts **any** spatial distribution of data points and produces balanced, efficient graphs for message-passing neural weather models — while maintaining full backward compatibility with existing rectangular grid workflows.

---

## 3. Motivation & Problem Statement

### 3.1 The Encode-Process-Decode Architecture

Graph-based neural weather models use an **encode-process-decode** architecture where three graph components orchestrate information flow:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Grid Nodes │────>│  Mesh Nodes │────>│  Grid Nodes │
│  (Data)     │ G2M │  (Latent)   │ M2G │  (Output)   │
└─────────────┘     └─────────────┘     └─────────────┘
     ENCODE             PROCESS             DECODE
```

- **Grid nodes** represent spatial locations where atmospheric data exists
- **Mesh nodes** form a latent computational graph where message-passing (via `InteractionNet`) occurs
- **G2M** (grid-to-mesh) encodes data onto the mesh; **M2G** (mesh-to-grid) decodes predictions back
- **M2M** (mesh-to-mesh) processes information within the mesh — this is where the rectangular assumption lives

The quality and topology of the mesh directly determines model expressiveness and prediction skill.

### 3.2 The Rectangular Grid Bottleneck

The mesh generation pipeline is locked to rectangular layouts in two coupled places:

**In `weather-model-graphs` — [`mesh.py`](https://github.com/mllam/weather-model-graphs/blob/main/src/weather_model_graphs/create/mesh/mesh.py):**
```python
def create_single_level_2d_mesh_graph(xy, nx, ny):
    # Mesh nodes placed on linspace grid
    lx = np.linspace(xm + dx/2, xM - dx/2, nx)
    ly = np.linspace(ym + dy/2, yM - dy/2, ny)
    # Connectivity is always grid_2d_graph + hardcoded diagonal edges
    g = networkx.grid_2d_graph(len(lx), len(ly))
    g.add_edges_from(...)  # diagonal edges ← ONLY works for rectangles
```

**In `neural-lam` — [`create_graph.py`](https://github.com/mllam/neural-lam/blob/main/neural_lam/create_graph.py):**
```python
def create_graph_from_datastore(datastore, ...):
    if isinstance(datastore, BaseRegularGridDatastore):
        xy = datastore.get_xy(category="state", stacked=False)
    else:
        raise NotImplementedError(
            "Only graph creation for BaseRegularGridDatastore is supported"
        )
```

As Joel noted in [Issue #17](https://github.com/mllam/weather-model-graphs/issues/17), the current archetypes are **"regional rectangular LAM adaptations"** of the global graphs proposed by Keisler and GraphCast — not the original triangular/icosahedral meshes. The existing code only handles the rectangular case.

### 3.3 Conceptual Coupling Problem (Identified in Issue [#71](https://github.com/mllam/weather-model-graphs/issues/71))

Today, `m2m_connectivity="flat"` implies **both**:
1. "The mesh is a single-level flat structure" (connectivity topology)
2. "The mesh nodes are laid out on a rectilinear grid" (spatial layout)

These are **distinct concepts** that should be separable. For example:
- A **hierarchical graph with triangular mesh** at each level is valid but cannot be expressed
- A **flat graph with a user-provided icosahedral mesh** makes sense but is impossible
- A **rectilinear mesh with haversine distance** (PR [#37](https://github.com/mllam/weather-model-graphs/pull/37)) creates non-equidistant nodes on a sphere

### 3.4 Real-World Use Cases Currently Blocked

| Data Source | Grid Type | Blocked By |
|-------------|-----------|------------|
| MEPS / DANRA NWP output | Regular rectangular | ✅ Supported |
| ERA5 on lat-lon grid | Regular rectangular | ✅ Supported |
| ICON model output | Icosahedral triangular | `grid_2d_graph` assumption |
| IFS reduced Gaussian grid | Irregular | `BaseRegularGridDatastore` check |
| MPAS global mesh | Unstructured Voronoi | `grid_2d_graph` assumption |
| Weather station networks | Scattered points | `BaseRegularGridDatastore` check |
| Ship/buoy observations | Sparse irregular | `BaseRegularGridDatastore` check |
| Satellite swath data | Irregular scanning | `BaseRegularGridDatastore` check |
| Multi-source blended datasets | Mixed density | No density-adaptive mesh |

### 3.5 Why This Matters Now

Per Joel's ["Crucial things to merge"](https://github.com/mllam/neural-lam/issues/138) list, the integration of wmg into neural-lam is a top priority for enabling boundary/interior datastore training. Several **v0.4.0-milestoned issues** ([#40](https://github.com/mllam/weather-model-graphs/issues/40), [#42](https://github.com/mllam/weather-model-graphs/issues/42)) must be resolved before wmg can reliably replace `create_graph.py`. This project addresses these blockers while simultaneously extending wmg to handle irregular grids.

---

## 4. Alignment with Mentor Priorities & Existing Roadmap

This section maps every proposed deliverable to **actual GitHub issues, PRs, and mentor discussions** to demonstrate that this work directly advances the MLLAM roadmap.

### 4.1 Joel's "Crucial Things to Merge" Checklist ([neural-lam #138](https://github.com/mllam/neural-lam/issues/138))

Joel's September 2025 comment in Issue #138 provides the definitive priority list for wmg integration. Here is how this project maps to each item:

| Joel's Checklist Item | Status | This Project's Contribution |
|----------------------|--------|----------------------------|
| **Separate g2m connectivity for boundary/interior** | In-progress (Joel's branch) | Will ensure flexible mesh handles decode_mask correctly |
| **Decoding mask in m2g** ([wmg #34](https://github.com/mllam/weather-model-graphs/pull/34)) | ✅ Merged | No action needed — already merged |
| **Convex hull cropping of mesh nodes** ([wmg #40](https://github.com/mllam/weather-model-graphs/issues/40)) | Open, v0.4.0 milestone | **Phase 1 deliverable** — implement via `scipy.spatial.ConvexHull` |
| **Assert all nodes in g2m have degree > 0** ([wmg #42](https://github.com/mllam/weather-model-graphs/issues/42)) | Open, v0.4.0 milestone | **Phase 1 deliverable** — validation in `create_all_graph_components()` |
| **Fix node-drop bug on edge filtering** ([wmg #51](https://github.com/mllam/weather-model-graphs/pull/51)) | ✅ Merged | No action needed |
| **Grid only connects to bottom level in hierarchical** | Joel's branch | Will integrate during hierarchical mesh refactor |
| **Replace `create_graph.py` with wmg-backed script** ([neural-lam #83](https://github.com/mllam/neural-lam/issues/83)) | Open, v0.6.0 milestone | **Phase 4 deliverable** — `build_rectangular_graph.py` |
| **Move edge_index reindexing to `load_graph()`** | Joel's branch | **Phase 4 deliverable** — clean up during integration |
| **Rescale graph features by max coordinate** | Joel's branch | **Phase 4 deliverable** — implement in `load_graph()` or save step |

### 4.2 weather-model-graphs v0.4.0 Milestone Issues

| Issue | Opened By | Core Requirement | This Project |
|-------|-----------|-----------------|-------------|
| [#40](https://github.com/mllam/weather-model-graphs/issues/40) — Convex hull cropping | Joel | Prevent mesh nodes "hanging freely" above regions with no grid nodes | **Implement** using Joel's prototype at `mesh_chull_filtering` branch |
| [#42](https://github.com/mllam/weather-model-graphs/issues/42) — Assert all nodes in g2m | Joel | Detect disconnected nodes early — "causes a lot of issues downstream" | **Implement** assertion + auto-fix option |
| [#45](https://github.com/mllam/weather-model-graphs/issues/45) — Level attribute consistency | Hauke | `level` (int) vs `levels` (str) inconsistency breaks `pyg.from_networkx()` | **Implement** unified `from_level`/`to_level` per agreed-upon design |

### 4.3 My Own Prior Issues (Demonstrating Understanding)

| Issue | Title | How It Connects |
|-------|-------|----------------|
| [#70](https://github.com/mllam/weather-model-graphs/issues/70) — Prebuilt mesh support | `m2m_connectivity="prebuilt"` option | Enables arbitrary mesh topologies without internal generation |
| [#71](https://github.com/mllam/weather-model-graphs/issues/71) — `mesh_layout` decoupling | Separate layout from connectivity | The core architectural change for flexible construction |

These two issues form the **foundation of my proposed architecture** and were designed after studying PRs [#37](https://github.com/mllam/weather-model-graphs/pull/37) (haversine/global), [#55](https://github.com/mllam/weather-model-graphs/pull/55) (ICON grids), and the full codebase.

### 4.4 Mentor Communication Style (From Issue Observation)

Understanding how mentors work helps plan effective collaboration:

- **Joel (@joeloskarsson):** Prefers concise, focused changes. Values prototype-first approach (has working branches for most features). Cares deeply about correctness — e.g., "nodes with no connections in g2m... causes a lot of issues downstream."
- **Leif (@leifdenby):** Maintains the roadmap and milestone system. Open to architectural improvements. Values clear API design — e.g., designed the `level` vs `levels` distinction deliberately.
- **Hauke (@observingClouds):** Focuses on practical robustness and upstream compatibility. Raised the PyG `from_networkx` incompatibility issue (#45).

### 4.5 Connection to GSoC Idea #4 (Global Weather Forecasting)

The flexible graph construction project also **unblocks Idea #4** (global forecasting), since global graphs require non-rectangular mesh topologies (icosahedral/triangular). By implementing the `mesh_layout` parameter and prebuilt mesh support, this project provides the infrastructure for future global graph construction without requiring a separate effort.

---

## 5. Current Architecture Deep-Dive

### 5.1 weather-model-graphs Package Layout

```
weather-model-graphs/src/weather_model_graphs/
├── __init__.py                         # Public API
├── create/
│   ├── __init__.py
│   ├── archetype.py                    # High-level named architectures
│   │   ├── create_keisler_graph()      # Flat, single-scale → m2m_connectivity="flat"
│   │   ├── create_graphcast_graph()    # Flat, multi-scale → m2m_connectivity="flat_multiscale"
│   │   └── create_oskarsson_hierarchical_graph()  # → m2m_connectivity="hierarchical"
│   │
│   ├── base.py                         # ★ CORE: create_all_graph_components()
│   │   ├── create_all_graph_components()  # Orchestrates g2m + m2m + m2g
│   │   └── connect_nodes_across_graphs()  # KD-tree connectivity (already flexible!)
│   │
│   ├── grid/
│   │   └── grid.py                     # Grid node creation (trivial: xy → nodes)
│   │
│   └── mesh/
│       ├── mesh.py                     # ★ BOTTLENECK: create_single_level_2d_mesh_graph()
│       │   └── Uses networkx.grid_2d_graph + linspace → RECTANGULAR ONLY
│       └── kinds/
│           ├── flat.py                 # create_flat_multiscale_mesh_graph()
│           └── hierarchical.py         # create_hierarchical_multiscale_mesh_graph()
│                                       # Uses "level" (int) and "levels" (str) ← Issue #45
│
├── networkx_utils.py                   # split_graph_by_edge_attribute(), prepend_node_index(), etc.
├── save.py                             # to_pyg() — converts networkx → PyG .pt files
└── visualise/
    └── plot_2d.py                      # 2D graph plotting
```

### 5.2 How `create_all_graph_components()` Works (base.py)

This is the **central orchestrator**. Understanding its flow is critical:

```python
def create_all_graph_components(
    coords,                    # [N, 2] data point coordinates
    m2m_connectivity,          # "flat" | "flat_multiscale" | "hierarchical"
    m2m_connectivity_kwargs,   # nx, ny, max_num_levels, etc.
    g2m_connectivity,          # "nearest_neighbour" | "nearest_neighbours" | "within_radius" | "containing_rectangle"
    m2g_connectivity,          # same options as g2m
    g2m_connectivity_kwargs,
    m2g_connectivity_kwargs,
    coords_crs=None,
    graph_crs=None,
    decode_mask=None,          # [N] bool: which grid nodes get m2g edges
    return_components=False,
):
```

**Flow:**
1. Create grid graph from `coords` (each point → node with `pos` attribute)
2. Create mesh graph based on `m2m_connectivity` value:
   - `"flat"` → `create_flat_singlescale_mesh_graph(xy, **kwargs)` → internally calls `create_single_level_2d_mesh_graph(xy, nx, ny)` ← **the bottleneck**
   - `"flat_multiscale"` → `create_flat_multiscale_mesh_graph(xy, **kwargs)` → calls same function multiple times with coarser nx/ny
   - `"hierarchical"` → `create_hierarchical_multiscale_mesh_graph(xy, **kwargs)` → multi-level with up/down edges
3. Connect grid ↔ mesh via `connect_nodes_across_graphs()` (uses `scipy.spatial.KDTree` — **already flexible**)
4. Return combined graph or separate components

**Key insight:** The g2m/m2g connection layer is already topology-agnostic. Only step 2 needs to be generalized.

### 5.3 What `create_single_level_2d_mesh_graph()` Actually Does (mesh.py)

```python
def create_single_level_2d_mesh_graph(xy, nx, ny):
    # 1. Compute bounding box from data coords
    xm, xM = xy[:, 0].min(), xy[:, 0].max()
    ym, yM = xy[:, 1].min(), xy[:, 1].max()
    
    # 2. Place mesh nodes on UNIFORM linspace grid
    lx = np.linspace(xm + dx/2, xM - dx/2, nx)  # ← Regular spacing assumed
    ly = np.linspace(ym + dy/2, yM - dy/2, ny)   # ← Regular spacing assumed
    
    # 3. Create rectangular grid graph
    g = networkx.grid_2d_graph(len(lx), len(ly))  # ← RECTANGULAR topology
    
    # 4. Add diagonal edges
    g.add_edges_from([((x,y), (x+1,y+1)) ...])   # ← Assumes grid structure
    
    # 5. Add node positions
    for i, (x, y) in enumerate(itertools.product(lx, ly)):
        g.nodes[(i // ny, i % ny)]["pos"] = np.array([x, y])
    
    return g
```

**Problems:**
- `np.linspace` creates equally-spaced nodes regardless of data density
- `grid_2d_graph` creates 4-connectivity (+ diagonals = 8-connectivity) → only valid for rectangular layouts
- Node naming uses `(i, j)` tuples → tightly coupled to grid indexing
- No convex hull cropping → mesh nodes can exist far from data points (Issue #40)

### 5.4 neural-lam's Graph Pipeline (Current State)

```
neural-lam/neural_lam/
├── create_graph.py        # Graph creation CLI + functions
│   ├── mk_2d_graph()       # ← DUPLICATES wmg mesh.py functionality
│   ├── create_graph()      # ← Hardcoded rectangular creation
│   └── create_graph_from_datastore()  # ← Blocks BaseDatastore
│
├── utils.py               # load_graph() — loads .pt files into model buffers
│   └── load_graph()        # Returns dict of BufferList tensors
│       # Expected files: m2m_edge_index.pt, g2m_edge_index.pt, m2g_edge_index.pt
│       #                 *_features.pt, mesh_features.pt
│       #                 mesh_up_edge_index.pt, mesh_down_edge_index.pt (hierarchical)
│
├── datastore/base.py      # BaseDatastore → get_xy(stacked=True) → [N, 2]
│                           # BaseRegularGridDatastore → get_xy(stacked=False) → [2, Nx, Ny]
│
├── models/
│   ├── base_graph_model.py # Loads graph, creates embedders, sets up GNN pipeline
│   │   └── NOTE: "mesh nodes MUST have first num_mesh_nodes indices" ← Joel wants to remove this
│   ├── graph_lam.py         # Flat model (single m2m InteractionNet step)
│   ├── hi_lam.py            # Hierarchical model (up/process/down steps)
│   └── hi_lam_parallel.py   # Parallel hierarchical variant
│
└── interaction_net.py       # InteractionNet: pyg.nn.MessagePassing subclass
    └── Currently does edge_index reindexing internally
        ← Joel wants to move this to load_graph() (Issue #138)
```

**Key problems identified by Joel in Issue #138:**
1. `create_graph.py` duplicates wmg functionality → should be replaced with wmg call
2. `InteractionNet` does edge_index reindexing → should move to `load_graph()`
3. `base_graph_model.py` has "mesh nodes MUST have first indices" constraint → should be removed
4. Graph features not rescaled by max coordinate → model training issues

### 5.5 Current Data Flow (End-to-End)

```
                         CURRENT END-TO-END FLOW
                         ========================

  DataStore                                                    Model
  ┌──────────────────┐                                    ┌──────────────┐
  │ BaseRegular      │                                    │ GraphLAM /   │
  │ GridDatastore    │                                    │ HiLAM /      │
  │                  │  get_xy(stacked=False)              │ HiLAMParallel│
  │ • MEPS example   │──────[2, Nx, Ny]──────┐           │              │
  │ • DANRA example  │                       │           │   Uses:      │
  └──────────────────┘                       ▼           │ • InterNet   │
                                    ┌─────────────────┐  │ • BufferList │
  ┌──────────────────┐              │ create_graph.py │  │              │
  │ BaseDatastore    │              │ (neural-lam)    │  └───────▲──────┘
  │ (irregular data) │              │                 │          │
  │                  │  ✗ BLOCKED   │ mk_2d_graph()   │          │
  │ • ICON output    │──── raise ──>│ (duplicate)     │   ┌──────┴──────┐
  │ • Station obs    │  NotImpl.    │                 │   │ load_graph()│
  │ • Ship tracks    │              │ Saves .pt files │──>│ (utils.py)  │
  └──────────────────┘              └─────────────────┘   └─────────────┘
                                           │
                    ┌──────────────────────-┘
                    │ weather-model-graphs
                    │ (NOT CURRENTLY USED
                    │  by neural-lam)
                    └──────────────────────>
```

---

## 6. Proposed Solution & Technical Design

### 6.1 High-Level Architecture Change

The core architectural change is captured in two complementary proposals:

1. **Issue [#71](https://github.com/mllam/weather-model-graphs/issues/71) — `mesh_layout` parameter:** Decouple mesh node layout (spatial arrangement) from mesh connectivity (hierarchical structure)
2. **Issue [#70](https://github.com/mllam/weather-model-graphs/issues/70) — `"prebuilt"` mesh support:** Allow injection of arbitrary mesh topologies

Together, these transform `create_all_graph_components()` from a rectangular-only system to a fully flexible one:

```python
# PROPOSED new signature for create_all_graph_components()
def create_all_graph_components(
    coords,                        # [N, 2] — any spatial distribution
    m2m_connectivity,              # "flat" | "flat_multiscale" | "hierarchical" | "prebuilt"
    mesh_layout="rectilinear",     # NEW: "rectilinear" | "triangular" | "from_graph"
    mesh_layout_kwargs={},         # NEW: kwargs for the layout function
    m2m_connectivity_kwargs={},
    g2m_connectivity="nearest_neighbours",
    m2g_connectivity="nearest_neighbours",
    g2m_connectivity_kwargs={},
    m2g_connectivity_kwargs={},
    crop_mesh_to_convex_hull=False,  # NEW: Issue #40
    assert_all_nodes_in_g2m=True,    # NEW: Issue #42
    coords_crs=None,
    graph_crs=None,
    decode_mask=None,
    return_components=False,
):
```

### 6.2 Proposed Data Flow (End-to-End)

```
                         PROPOSED END-TO-END FLOW
                         =========================

  ANY DataStore                                              Model
  ┌──────────────────┐                                  ┌──────────────┐
  │ BaseRegular      │  get_xy()                        │ GraphLAM /   │
  │ GridDatastore    │──────────┐                       │ HiLAM /      │
  └──────────────────┘          │                       │ HiLAMParallel│
                                │  [N, 2]               │              │
  ┌──────────────────┐          │                       └───────▲──────┘
  │ BaseDatastore    │──────────┤                               │
  │ (irregular)      │          │                        ┌──────┴──────┐
  └──────────────────┘          │                        │ load_graph()│
                                ▼                        │ (updated)   │
                    ┌───────────────────────────┐        │ • Reindex   │
                    │  build_rectangular_graph.py │       │ • Rescale   │
                    │  (neural-lam, replaces      │       └──────▲──────┘
                    │   create_graph.py)           │              │
                    │                              │       .pt files
                    │  Calls weather-model-graphs: │              │
                    │  ┌────────────────────────┐  │   ┌──────────┴────────┐
                    │  │ create_all_graph_       │  │   │  wmg.save.to_pyg() │
                    │  │   components()          │──┤──>│  (existing)         │
                    │  │                         │  │   └─────────────────────┘
                    │  │  m2m_connectivity=...   │  │
                    │  │  mesh_layout=...  ←NEW  │  │
                    │  │  crop_to_hull=True ←NEW │  │
                    │  │  assert_g2m=True   ←NEW │  │
                    │  └────────────────────────┘  │
                    └───────────────────────────┘

  mesh_layout dispatch:
  ┌────────────────────────────────────────────────────┐
  │ "rectilinear" → grid_2d_graph (existing, backward │
  │                  compatible)                       │
  │ "triangular"  → Delaunay triangulation (NEW)       │
  │ "from_graph"  → User-provided mesh (NEW, #70)     │
  │ "hexagonal"   → Hexagonal lattice (NEW)            │
  │ "density_adaptive" → Voronoi-based (NEW)           │
  └────────────────────────────────────────────────────┘
```

### 6.3 Component 1: `mesh_layout` Decoupling (Issue [#71](https://github.com/mllam/weather-model-graphs/issues/71))

**The key architectural insight:** `m2m_connectivity` and `mesh_layout` control different things:

| Parameter | Controls | Examples |
|-----------|----------|----------|
| `m2m_connectivity` | How mesh levels relate to each other | flat (one level), flat_multiscale (merged levels), hierarchical (up/down edges) |
| `mesh_layout` | How nodes are spatially arranged within a single level | rectilinear, triangular, hexagonal, from_graph |

**Combinatorial possibilities enabled:**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    MESH LAYOUT × CONNECTIVITY MATRIX                   │
├─────────────────────┬──────────────┬──────────────┬───────────────────┤
│                     │ flat         │ flat_multi   │ hierarchical      │
├─────────────────────┼──────────────┼──────────────┼───────────────────┤
│ rectilinear         │ ✅ Keisler   │ ✅ GraphCast │ ✅ Oskarsson      │
│ (current default)   │   (existing) │   (existing) │    (existing)     │
├─────────────────────┼──────────────┼──────────────┼───────────────────┤
│ triangular          │ ✅ NEW       │ ✅ NEW       │ ✅ NEW            │
│ (Delaunay)          │ Irreg. flat  │ Irreg. multi │ Irreg. hierarchal │
├─────────────────────┼──────────────┼──────────────┼───────────────────┤
│ hexagonal           │ ✅ NEW       │ ✅ NEW       │ ✅ NEW            │
│                     │ Isotropic    │ Isotropic    │ Isotropic hier.   │
├─────────────────────┼──────────────┼──────────────┼───────────────────┤
│ from_graph          │ ✅ NEW (#70) │ N/A          │ ✅ NEW (#70)      │
│ (user-provided)     │ ICON, MPAS   │ (manual)     │ Custom levels     │
├─────────────────────┼──────────────┼──────────────┼───────────────────┤
│ density_adaptive    │ ✅ NEW       │ ✅ NEW       │ ✅ NEW            │
│ (Voronoi-based)     │ Obs. nets    │ Mixed src    │ Multi-res obs     │
└─────────────────────┴──────────────┴──────────────┴───────────────────┘
```

**Implementation — refactoring `create_single_level_2d_mesh_graph()`:**

```python
# weather-model-graphs/src/weather_model_graphs/create/mesh/mesh.py

# EXISTING function preserved for backward compatibility
def create_single_level_2d_mesh_graph(xy, nx, ny):
    """Original rectangular mesh creation — unchanged."""
    ...  # existing code

# NEW dispatch function
def create_single_level_mesh_graph(xy, mesh_layout="rectilinear", **layout_kwargs):
    """
    Create a single-level mesh graph using the specified layout strategy.
    
    Parameters
    ----------
    xy : np.ndarray [N, 2]
        Data point coordinates (used for extent computation)
    mesh_layout : str
        Layout strategy: "rectilinear", "triangular", "hexagonal",
        "density_adaptive", "from_graph"
    **layout_kwargs
        Layout-specific parameters
        
    Returns
    -------
    networkx.DiGraph
        Mesh graph with 'pos', 'type' node attributes and 'len', 'vdiff' edge attributes
    """
    if mesh_layout == "rectilinear":
        nx = layout_kwargs.get("nx", 3)
        ny = layout_kwargs.get("ny", 3)
        return create_single_level_2d_mesh_graph(xy, nx, ny)
    elif mesh_layout == "triangular":
        return _create_triangular_mesh(xy, **layout_kwargs)
    elif mesh_layout == "hexagonal":
        return _create_hexagonal_mesh(xy, **layout_kwargs)
    elif mesh_layout == "density_adaptive":
        return _create_density_adaptive_mesh(xy, **layout_kwargs)
    elif mesh_layout == "from_graph":
        mesh_graph = layout_kwargs.pop("mesh_graph")
        _validate_mesh_graph(mesh_graph)
        return mesh_graph
    else:
        raise ValueError(f"Unknown mesh_layout: {mesh_layout}")
```

### 6.4 Component 2: Prebuilt Mesh Support (Issue [#70](https://github.com/mllam/weather-model-graphs/issues/70))

The `"prebuilt"` pathway enables injecting any user-created mesh:

```python
# In create_all_graph_components() — new m2m_connectivity option
elif m2m_connectivity == "prebuilt":
    mesh_graph = m2m_connectivity_kwargs.pop("mesh_graph", None)
    if mesh_graph is None:
        raise ValueError(
            "When m2m_connectivity='prebuilt', a 'mesh_graph' must be "
            "provided in m2m_connectivity_kwargs."
        )
    _validate_mesh_graph(mesh_graph)
    graph_components["m2m"] = mesh_graph
    grid_connect_graph = mesh_graph
```

**Validation helper:**

```python
def _validate_mesh_graph(mesh_graph):
    """
    Validate that a user-provided mesh graph has the required structure.
    
    Required:
    - networkx.DiGraph
    - All nodes have 'pos' attribute (2-element numeric array)
    - All nodes have 'type' == 'mesh'
    - All edges have 'len' and 'vdiff' attributes (or compute them)
    """
    if not isinstance(mesh_graph, networkx.DiGraph):
        raise TypeError(f"Expected networkx.DiGraph, got {type(mesh_graph)}")
    
    for node, data in mesh_graph.nodes(data=True):
        if "pos" not in data:
            raise ValueError(f"Node {node} missing 'pos' attribute")
        if data.get("type") != "mesh":
            raise ValueError(f"Node {node} has type='{data.get('type')}', expected 'mesh'")
    
    # Auto-compute edge features if missing
    for u, v, data in mesh_graph.edges(data=True):
        if "len" not in data or "vdiff" not in data:
            pos_u = np.array(mesh_graph.nodes[u]["pos"])
            pos_v = np.array(mesh_graph.nodes[v]["pos"])
            vdiff = pos_u - pos_v
            mesh_graph.edges[u, v]["vdiff"] = vdiff
            mesh_graph.edges[u, v]["len"] = np.linalg.norm(vdiff)
```

**User-facing example:**

```python
import networkx as nx
import weather_model_graphs as wmg

# Create an ICON icosahedral mesh externally
icon_mesh = create_icon_mesh_from_file("icon_grid_description.nc")  # Returns nx.DiGraph

# Plug it into wmg
graph = wmg.create.archetype.create_all_graph_components(
    coords=my_data_coords,
    m2m_connectivity="prebuilt",
    m2m_connectivity_kwargs=dict(mesh_graph=icon_mesh),
    g2m_connectivity="nearest_neighbours",
    m2g_connectivity="nearest_neighbours",
    g2m_connectivity_kwargs=dict(max_num_neighbours=4),
    m2g_connectivity_kwargs=dict(max_num_neighbours=4),
)
```

### 6.5 Component 3: Delaunay-based Triangular Mesh Layout

For irregular point distributions, Delaunay triangulation provides a natural mesh connectivity:

```
┌──────────────────────────────────────────────────────────────┐
│            MESH CONNECTIVITY COMPARISON                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  CURRENT: grid_2d_graph + diagonals     NEW: Delaunay        │
│                                                              │
│    o───o───o───o       Only works       o─────o              │
│    │╲ │╲ │╲ │         for              ╱│╲   ╱│╲            │
│    │ ╲│ ╲│ ╲│         rectangles!     ╱ │ ╲╱  │ ╲           │
│    o───o───o───o                     o──┼──o──┼──o           │
│    │╲ │╲ │╲ │                         ╲ │ ╱╲  │ ╱           │
│    │ ╲│ ╲│ ╲│                          ╲│╱   ╲│╱            │
│    o───o───o───o                        o─────o              │
│                                                              │
│  Properties of Delaunay triangulation:                       │
│  • Works for ANY point distribution                          │
│  • Maximizes minimum angles (avoids thin triangles)          │
│  • O(N log N) construction via scipy.spatial.Delaunay        │
│  • Natural dual: Voronoi diagram (useful for area weights)   │
│  • For rectangular grids, produces EQUIVALENT connectivity   │
│    (8-neighbor pattern matches grid_2d + diagonals)          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
# New file: weather-model-graphs/src/weather_model_graphs/create/mesh/layouts.py

def _create_triangular_mesh(
    xy: np.ndarray,
    mesh_node_distance: float = None,
    num_mesh_nodes: int = None,
    max_edge_length: float = None,
    boundary_margin: float = 0.1,
    seed: int = 42,
) -> networkx.DiGraph:
    """
    Create a mesh graph using Delaunay triangulation.
    
    Steps:
    1. Place mesh nodes uniformly across data extent (Poisson disk sampling)
    2. Build Delaunay triangulation of mesh node positions
    3. Convert triangle edges to bidirectional DiGraph edges
    4. Optionally prune edges exceeding max_edge_length
    
    Parameters
    ----------
    xy : np.ndarray [N, 2]
        Data point coordinates (used for extent computation only)
    mesh_node_distance : float, optional
        Target minimum distance between mesh nodes
    num_mesh_nodes : int, optional
        Target number of mesh nodes (alternative to mesh_node_distance)
    max_edge_length : float, optional
        Maximum allowed edge length (removes long edges at domain boundaries)
    boundary_margin : float
        Fractional margin around data extent for mesh nodes
    seed : int
        Random seed for reproducibility
    """
    from scipy.spatial import Delaunay
    
    # 1. Determine mesh node positions
    if mesh_node_distance is not None:
        mesh_positions = _poisson_disk_sample(
            bounds=_compute_bounds(xy, margin=boundary_margin),
            radius=mesh_node_distance,
            seed=seed,
        )
    elif num_mesh_nodes is not None:
        # Estimate distance from desired count
        extent = _compute_extent(xy)
        area = extent[0] * extent[1]
        mesh_node_distance = np.sqrt(area / num_mesh_nodes)
        mesh_positions = _poisson_disk_sample(
            bounds=_compute_bounds(xy, margin=boundary_margin),
            radius=mesh_node_distance,
            seed=seed,
        )
    else:
        raise ValueError("Provide either mesh_node_distance or num_mesh_nodes")
    
    # 2. Delaunay triangulation
    tri = Delaunay(mesh_positions)
    
    # 3. Build directed graph
    g = networkx.DiGraph()
    for i, pos in enumerate(mesh_positions):
        g.add_node(i, pos=pos, type="mesh")
    
    # Extract unique edges from simplices
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = simplex[i], simplex[j]
                edges.add((min(u, v), max(u, v)))
    
    for u, v in edges:
        pos_u, pos_v = mesh_positions[u], mesh_positions[v]
        d = np.linalg.norm(pos_u - pos_v)
        if max_edge_length is not None and d > max_edge_length:
            continue
        vdiff = pos_u - pos_v
        g.add_edge(u, v, len=d, vdiff=vdiff)
        g.add_edge(v, u, len=d, vdiff=-vdiff)
    
    return g
```

### 6.6 Component 4: Density-Adaptive Mesh for Irregular Observations

For sparse/clustered data (weather stations, ship tracks), mesh nodes should be placed with density proportional to data density:

```
┌──────────────────────────────────────────────────────────────┐
│            DENSITY-ADAPTIVE MESH NODE PLACEMENT              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Data points:          Mesh nodes (uniform):                 │
│  · · · · · ·          ○   ○   ○   ○   ← Too many mesh      │
│  · · · · · ·          ○   ○   ○   ○      nodes over ocean   │
│  · · · · · ·          ○   ○   ○   ○      (sparse data)      │
│       ·               ○   ○   ○   ○                         │
│             ·         ○   ○   ○   ○                         │
│     ·                 ○   ○   ○   ○   ← Too few mesh        │
│                                          nodes over land     │
│  (Dense land,                            (dense data)        │
│   sparse ocean)                                              │
│                                                              │
│  Mesh nodes (density-adaptive):                              │
│  ○ ○ ○ ○ ○ ○         ← Dense mesh where data is dense      │
│  ○ ○ ○ ○ ○ ○                                                │
│       ○               ← Sparse mesh where data is sparse    │
│           ○                                                  │
│     ○                                                        │
│                                                              │
│  Method: Voronoi cell areas → local density → mesh spacing  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

```python
def _create_density_adaptive_mesh(
    xy: np.ndarray,
    base_mesh_distance: float,
    density_scaling: float = 0.5,
    min_mesh_distance: float = None,
    max_mesh_distance: float = None,
    seed: int = 42,
) -> networkx.DiGraph:
    """
    Place mesh nodes with density proportional to local data density.
    
    Algorithm:
    1. Compute Voronoi diagram of data points
    2. Estimate local density from Voronoi cell areas
    3. Map density → local mesh spacing
    4. Place mesh nodes with variable-radius Poisson disk sampling
    5. Build Delaunay connectivity
    """
    from scipy.spatial import Voronoi
    
    # 1. Estimate density
    vor = Voronoi(xy)
    cell_areas = _compute_voronoi_cell_areas(vor)
    density = 1.0 / cell_areas  # points per unit area
    
    # 2. Map density to mesh spacing
    density_norm = density / density.max()
    # Higher density → smaller mesh spacing
    local_spacing = base_mesh_distance / (density_norm ** density_scaling)
    local_spacing = np.clip(local_spacing, min_mesh_distance, max_mesh_distance)
    
    # 3. Variable-radius sampling
    mesh_positions = _variable_radius_poisson_disk(
        bounds=_compute_bounds(xy),
        density_field=density_norm,
        base_radius=base_mesh_distance,
        scaling=density_scaling,
        seed=seed,
    )
    
    # 4. Delaunay connectivity
    return _build_delaunay_graph(mesh_positions)
```

### 6.7 Component 5: Convex Hull Cropping (Issue [#40](https://github.com/mllam/weather-model-graphs/issues/40))

Joel's description: *"This would prevent parts of the mesh 'hanging freely' above regions where there are no grid nodes. Such mesh nodes are not connected to the grid, making them useless."*

Joel already has a prototype at [`mesh_chull_filtering` branch](https://github.com/joeloskarsson/weather-model-graphs/tree/mesh_chull_filtering).

```python
def crop_mesh_to_convex_hull(
    mesh_graph: networkx.DiGraph,
    grid_coords: np.ndarray,
    margin: float = 0.0,
) -> networkx.DiGraph:
    """
    Remove mesh nodes that fall outside the convex hull of grid coordinates.
    
    Based on Joel's prototype: joeloskarsson/weather-model-graphs@c824c23
    
    Parameters
    ----------
    mesh_graph : networkx.DiGraph
        Mesh graph to crop
    grid_coords : np.ndarray [N, 2]
        Grid point coordinates
    margin : float
        Buffer distance around convex hull
        
    Returns
    -------
    networkx.DiGraph
        Cropped mesh graph
    """
    from scipy.spatial import ConvexHull, Delaunay
    
    hull = ConvexHull(grid_coords)
    # Use Delaunay for faster point-in-hull testing
    hull_delaunay = Delaunay(grid_coords[hull.vertices])
    
    # Identify mesh nodes inside hull (+ margin)
    nodes_to_keep = []
    for node, data in mesh_graph.nodes(data=True):
        pos = np.array(data["pos"])
        if hull_delaunay.find_simplex(pos) >= 0:
            nodes_to_keep.append(node)
        elif margin > 0:
            # Check if within margin distance of hull
            dist = _point_to_hull_distance(pos, hull, grid_coords)
            if dist <= margin:
                nodes_to_keep.append(node)
    
    return mesh_graph.subgraph(nodes_to_keep).copy()
```

### 6.8 Component 6: G2M Node Assertion (Issue [#42](https://github.com/mllam/weather-model-graphs/issues/42))

Joel's description: *"It can cause a lot of issues downstream if there are nodes with no connections in g2m."*

```python
def assert_all_nodes_in_g2m(
    g2m_graph: networkx.DiGraph,
    grid_graph: networkx.DiGraph,
    mesh_graph: networkx.DiGraph,
    fix_disconnected: bool = False,
):
    """
    Assert that all nodes in g2m have at least one connection.
    
    Based on Joel's prototype: joeloskarsson/weather-model-graphs@base.py#L199-L207
    
    Parameters
    ----------
    g2m_graph : networkx.DiGraph
        Grid-to-mesh graph
    grid_graph : networkx.DiGraph
        Grid graph
    mesh_graph : networkx.DiGraph
        Mesh graph
    fix_disconnected : bool
        If True, add nearest-neighbor edges for disconnected nodes
    """
    grid_nodes = set(grid_graph.nodes())
    mesh_nodes = set(mesh_graph.nodes())
    
    # Check grid nodes have outgoing edges (they should send to mesh)
    grid_nodes_in_g2m = set(u for u, v in g2m_graph.edges())
    disconnected_grid = grid_nodes - grid_nodes_in_g2m
    
    # Check mesh nodes have incoming edges (they should receive from grid)
    mesh_nodes_in_g2m = set(v for u, v in g2m_graph.edges())
    disconnected_mesh = mesh_nodes - mesh_nodes_in_g2m
    
    if disconnected_grid or disconnected_mesh:
        msg = []
        if disconnected_grid:
            msg.append(f"{len(disconnected_grid)} grid nodes have no g2m edges")
        if disconnected_mesh:
            msg.append(f"{len(disconnected_mesh)} mesh nodes have no g2m edges")
        
        if fix_disconnected:
            # Add nearest-neighbor connections for disconnected nodes
            _fix_disconnected_nodes(g2m_graph, disconnected_grid, 
                                    disconnected_mesh, grid_graph, mesh_graph)
        else:
            raise ValueError(
                "Disconnected nodes in g2m graph: " + "; ".join(msg) +
                ". Use crop_mesh_to_convex_hull=True or fix_disconnected=True."
            )
```

### 6.9 Component 7: Level Attribute Consistency (Issue [#45](https://github.com/mllam/weather-model-graphs/issues/45))

Hauke identified that `hierarchical.py` uses both `"level"` (int, for same-level edges) and `"levels"` (str like `"0>1"`, for inter-level edges). This breaks PyG's `from_networkx` conversion which expects consistent types.

The agreed-upon solution (confirmed by Joel, Leif, and Hauke):

```python
# In hierarchical.py — replace current edge attributes

# BEFORE (current):
G.edges[u, v]["level"] = i                    # int, same-level edges
G_down.edges[u, v]["levels"] = f"{from_level}>{to_level}"  # str, inter-level

# AFTER (proposed — agreed in discussion):
G.edges[u, v]["from_level"] = i               # int, same-level
G.edges[u, v]["to_level"] = i                 # int, same-level (from_level == to_level)
G_down.edges[u, v]["from_level"] = from_level  # int, inter-level
G_down.edges[u, v]["to_level"] = to_level      # int, inter-level
G_up.edges[u, v]["from_level"] = from_level    # int, inter-level
G_up.edges[u, v]["to_level"] = to_level        # int, inter-level
```

This ensures:
- All edges have consistent attribute names and types
- `from_level == to_level` indicates same-level edges (replaces `"level"`)
- `from_level != to_level` indicates inter-level edges (replaces `"levels"`)
- PyG `from_networkx()` works without errors

### 6.10 Component 8: neural-lam Integration — Replace `create_graph.py` (Issue [#83](https://github.com/mllam/neural-lam/issues/83))

Joel's plan from Issue #138:

> *"Change main graph creation script. Remove `create_graph.py` and replace it with `build_rectangular_graph.py`, which calls wmg. This is the switch to using wmg in neural-lam."*

Joel already has working code at [`joeloskarsson/neural-lam-dev/blob/research/neural_lam/build_rectangular_graph.py`](https://github.com/joeloskarsson/neural-lam-dev/blob/research/neural_lam/build_rectangular_graph.py).

**My implementation plan:**

```python
# neural-lam/neural_lam/build_graph.py (NEW — replaces create_graph.py)

"""
Build graph for neural-lam training using weather-model-graphs.

This replaces the old create_graph.py which duplicated wmg functionality.
Resolves: neural-lam#83, neural-lam#4
"""

import argparse
from pathlib import Path
import weather_model_graphs as wmg
from neural_lam.datastore import init_datastore

def build_graph(
    datastore_config_path: str,
    graph_dir_path: str,
    archetype: str = "keisler",          # keisler | graphcast | oskarsson
    mesh_layout: str = "rectilinear",     # NEW: rectilinear | triangular | hexagonal | ...
    mesh_node_distance: float = None,
    n_max_levels: int = None,
    crop_to_convex_hull: bool = False,
    create_plot: bool = False,
    **kwargs,
):
    """
    Build neural network graph from datastore coordinates using wmg.
    
    For regular grid datastores, uses the existing rectangular pipeline.
    For irregular datastores, uses the new flexible mesh layouts.
    """
    datastore = init_datastore(datastore_config_path)
    
    # Get coordinates — works for both regular and irregular datastores
    xy = datastore.get_xy(category="state", stacked=True)  # [N, 2]
    
    # Select archetype
    archetype_fn = {
        "keisler": wmg.create.archetype.create_keisler_graph,
        "graphcast": wmg.create.archetype.create_graphcast_graph,
        "oskarsson": wmg.create.archetype.create_oskarsson_hierarchical_graph,
    }[archetype]
    
    # Build graph
    graph = archetype_fn(
        coords=xy,
        mesh_layout=mesh_layout,  # NEW parameter flows through
        crop_mesh_to_convex_hull=crop_to_convex_hull,
        **kwargs,
    )
    
    # Save to .pt files
    graph_dir = Path(graph_dir_path)
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    wmg.save.to_pyg(
        graph=graph,
        output_directory=str(graph_dir),
    )
    
    if create_plot:
        wmg.visualise.plot_graph(graph, save_path=graph_dir / "graph.png")


def main():
    parser = argparse.ArgumentParser(description="Build graph for neural-lam")
    parser.add_argument("--datastore_config", required=True)
    parser.add_argument("--graph_dir", required=True)
    parser.add_argument("--archetype", default="keisler",
                       choices=["keisler", "graphcast", "oskarsson"])
    parser.add_argument("--mesh_layout", default="rectilinear",
                       choices=["rectilinear", "triangular", "hexagonal",
                               "density_adaptive", "from_graph"])
    parser.add_argument("--mesh_node_distance", type=float, default=None)
    parser.add_argument("--n_max_levels", type=int, default=None)
    parser.add_argument("--crop_to_convex_hull", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    
    build_graph(
        datastore_config_path=args.datastore_config,
        graph_dir_path=args.graph_dir,
        archetype=args.archetype,
        mesh_layout=args.mesh_layout,
        mesh_node_distance=args.mesh_node_distance,
        n_max_levels=args.n_max_levels,
        crop_to_convex_hull=args.crop_to_convex_hull,
        create_plot=args.plot,
    )

if __name__ == "__main__":
    main()
```

**Additional changes in neural-lam per Joel's #138 checklist:**

1. **Move edge_index reindexing from `InteractionNet` to `load_graph()`:**
```python
# In utils.py load_graph():
# After loading edge indices, perform reindexing so InteractionNet
# doesn't need to do it at every forward pass
```

2. **Rescale graph features by maximum coordinate:**
```python
# In utils.py load_graph() or in build_graph():
max_coord = xy.max()
graph_features /= max_coord
```

3. **Remove the "mesh nodes MUST have first indices" constraint** from `base_graph_model.py`

### 6.11 Multi-Level Coarsening for Irregular Grids

For hierarchical and multiscale graphs with irregular mesh nodes, we can't use `nx/refinement_factor` to get coarser levels. Instead, use **Farthest Point Sampling (FPS)**:

```
┌────────────────────────────────────────────────────────────────┐
│         HIERARCHICAL MESH COARSENING (IRREGULAR)               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Level 0 (finest):  ~400 nodes                                 │
│  ·  ·  ·  ·  ·  ·   Delaunay triangulation                    │
│  ·  ·  ·  ·  ·  ·   of uniformly/adaptively placed nodes      │
│  ·  ·  ·  ·  ·  ·                                             │
│         │                                                      │
│         ▼ Farthest Point Sampling (factor k)                   │
│                                                                │
│  Level 1 (coarser): ~44 nodes                                  │
│  ·     ·     ·       FPS preserves spatial coverage             │
│     ·     ·          + re-triangulated connectivity             │
│  ·     ·     ·                                                 │
│         │                                                      │
│         ▼ Farthest Point Sampling (factor k)                   │
│                                                                │
│  Level 2 (coarsest): ~5 nodes                                  │
│  ·        ·          FPS + Delaunay                             │
│        ·                                                       │
│                                                                │
│  Inter-level: Nearest-neighbor up/down edges                   │
│  (Same pattern as existing hierarchical approach)              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

```python
def farthest_point_subsample(
    positions: np.ndarray,
    num_target: int,
    seed_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select subset using farthest point sampling (FPS).
    Greedy algorithm ensuring maximum spatial coverage.
    
    Returns (selected_positions, selected_indices)
    """
    N = len(positions)
    assert num_target <= N
    
    selected = [seed_index]
    min_distances = np.full(N, np.inf)
    
    for _ in range(num_target - 1):
        last_selected = positions[selected[-1]]
        dists = np.linalg.norm(positions - last_selected, axis=1)
        min_distances = np.minimum(min_distances, dists)
        min_distances[selected] = -1
        next_idx = np.argmax(min_distances)
        selected.append(next_idx)
    
    selected = np.array(selected)
    return positions[selected], selected
```

---

## 7. Detailed Implementation Plan (Phase-by-Phase)

### Pre-GSoC Community Bonding Period (May 8 – June 1)

**Objective:** Merge prior contributions, finalize design with mentors, set up development environment.

| # | Task | Deliverable |
|---|------|-------------|
| B.1 | Get Issue [#70](https://github.com/mllam/weather-model-graphs/issues/70) (prebuilt mesh) reviewed and accepted | Design agreement from mentors |
| B.2 | Get Issue [#71](https://github.com/mllam/weather-model-graphs/issues/71) (mesh_layout) reviewed and accepted | Architecture sign-off |
| B.3 | Study Joel's prototype branches in depth | Code notes + integration plan |
| B.4 | Set up CI/CD test environment locally | Can run full test suites for both repos |
| B.5 | Draft POC: Delaunay mesh from random points | Working Jupyter notebook demo |

### Phase 1: Critical wmg v0.4.0 Fixes (Weeks 1–3, ~75 hours)

**Rationale:** These are **prerequisites** for reliable wmg integration into neural-lam. Joel explicitly listed them as "crucial things to merge" in Issue #138.

#### Week 1: Convex Hull Cropping (Issue [#40](https://github.com/mllam/weather-model-graphs/issues/40))

| # | Task | File(s) | Hours |
|---|------|---------|-------|
| 1.1 | Implement `crop_mesh_to_convex_hull()` based on Joel's prototype | `create/mesh/mesh.py` or new `create/mesh/cropping.py` | 10 |
| 1.2 | Add `crop_mesh_to_convex_hull` parameter to `create_all_graph_components()` | `create/base.py` | 4 |
| 1.3 | Unit tests — rectangular grid, L-shaped domain, with/without margin | `tests/test_convex_hull.py` | 6 |
| 1.4 | Add visualization showing before/after cropping | `visualise/` | 3 |

**Acceptance criteria:**
- Mesh nodes outside grid convex hull are removed
- Configurable margin allows buffer zone
- All existing tests still pass (backward compatible — defaults to `False`)
- Visual comparison matches Joel's before/after screenshots from Issue #40

#### Week 2: G2M Node Assertion (Issue [#42](https://github.com/mllam/weather-model-graphs/issues/42))

| # | Task | File(s) | Hours |
|---|------|---------|-------|
| 2.1 | Implement degree-0 node detection in `create_all_graph_components()` | `create/base.py` | 8 |
| 2.2 | Implement auto-fix: add nearest-neighbor edges for disconnected nodes | `create/base.py` | 6 |
| 2.3 | Add `assert_all_nodes_in_g2m` parameter | `create/base.py` | 3 |
| 2.4 | Unit tests — disconnected grid nodes, disconnected mesh nodes | `tests/test_g2m_assertion.py` | 6 |

**Acceptance criteria:**
- Clear error message when nodes have degree 0 in g2m
- Optional auto-fix adds nearest-neighbor connections
- Works for both grid→mesh and mesh→grid directions

#### Week 3: Level Attribute Consistency (Issue [#45](https://github.com/mllam/weather-model-graphs/issues/45))

| # | Task | File(s) | Hours |
|---|------|---------|-------|
| 3.1 | Replace `"level"` (int) and `"levels"` (str) with `from_level`/`to_level` | `create/mesh/kinds/hierarchical.py` | 8 |
| 3.2 | Update `save.py` and `networkx_utils.py` to use new attribute names | `save.py`, `networkx_utils.py` | 5 |
| 3.3 | Update tests that reference old attribute names | `tests/` | 4 |
| 3.4 | Verify PyG `from_networkx()` works with new attributes | Integration test | 4 |
| 3.5 | Add deprecation warnings for old attribute access | `create/mesh/kinds/hierarchical.py` | 3 |

**Acceptance criteria:**
- All edges have `from_level` (int) and `to_level` (int) attributes
- Same-level edges: `from_level == to_level`
- Inter-level edges: `from_level != to_level`
- PyG `from_networkx()` conversion works without errors
- Old attributes removed or deprecated with warnings

### Phase 2: Flexible Mesh Layouts (Weeks 4–6, ~80 hours)

**Rationale:** This is the **core feature** — introducing `mesh_layout` parameter and new layout strategies.

#### Week 4: `mesh_layout` Refactoring + Triangular Layout

| # | Task | File(s) | Hours |
|---|------|---------|-------|
| 4.1 | Add `mesh_layout` + `mesh_layout_kwargs` parameters to `create_all_graph_components()` | `create/base.py` | 8 |
| 4.2 | Create `create_single_level_mesh_graph()` dispatch function | `create/mesh/mesh.py` | 5 |
| 4.3 | Implement `_create_triangular_mesh()` using Poisson disk + Delaunay | `create/mesh/layouts.py` (NEW) | 12 |
| 4.4 | Unit tests — triangular mesh from random points, from regular grid, edge cases | `tests/test_mesh_layouts.py` | 8 |

#### Week 5: Additional Layouts + Grid Detection

| # | Task | File(s) | Hours |
|---|------|---------|-------|
| 5.1 | Implement `_create_hexagonal_mesh()` | `create/mesh/layouts.py` | 8 |
| 5.2 | Implement `_create_density_adaptive_mesh()` using Voronoi areas | `create/mesh/layouts.py` | 12 |
| 5.3 | Implement `detect_grid_type()` auto-detection utility | `create/mesh/detection.py` (NEW) | 8 |
| 5.4 | Unit tests for hexagonal, density-adaptive, detection | `tests/test_mesh_layouts.py` | 6 |

#### Week 6: Prebuilt Mesh Pathway + Multi-Level Extension

| # | Task | File(s) | Hours |
|---|------|---------|-------|
| 6.1 | Implement `m2m_connectivity="prebuilt"` pathway (Issue #70) | `create/base.py` | 6 |
| 6.2 | Implement `_validate_mesh_graph()` helper | `create/mesh/mesh.py` | 4 |
| 6.3 | Implement FPS-based multi-level coarsening for non-rectangular meshes | `create/mesh/coarsening.py` (NEW) | 10 |
| 6.4 | Wire up `mesh_layout` through `flat_multiscale` and `hierarchical` paths | `create/mesh/kinds/` | 6 |
| 6.5 | Integration tests — all layout × connectivity combinations | `tests/test_flexible_integration.py` | 6 |

**Acceptance criteria for Phase 2:**
- `mesh_layout="rectilinear"` produces **identical** results to current code (backward compat)
- `mesh_layout="triangular"` works for arbitrary point distributions
- `mesh_layout="from_graph"` accepts user-provided meshes
- All three `m2m_connectivity` options work with all `mesh_layout` options
- Existing archetypes default to `mesh_layout="rectilinear"` (no behavior change)

---

### << MIDTERM EVALUATION — End of Week 6 >>

**Midterm deliverables:**
1. wmg v0.4.0 blocker issues resolved (#40, #42, #45)
2. `mesh_layout` parameter fully functional in `create_all_graph_components()`
3. Delaunay, hexagonal, and density-adaptive mesh layouts implemented
4. Prebuilt mesh support enabled
5. Full test coverage for all new features

---

### Phase 3: Archetype Updates + New Archetypes (Weeks 7–9, ~70 hours)

#### Week 7: Update Existing Archetypes to Accept `mesh_layout`

| # | Task | File(s) | Hours |
|---|------|---------|-------|
| 7.1 | Add `mesh_layout` parameter to `create_keisler_graph()` | `create/archetype.py` | 4 |
| 7.2 | Add `mesh_layout` parameter to `create_graphcast_graph()` | `create/archetype.py` | 4 |
| 7.3 | Add `mesh_layout` parameter to `create_oskarsson_hierarchical_graph()` | `create/archetype.py` | 6 |
| 7.4 | Backward compatibility test — all archetypes produce same output with defaults | `tests/test_backward_compat.py` | 8 |

#### Week 8: New Convenience Archetypes

| # | Task | File(s) | Hours |
|---|------|---------|-------|
| 8.1 | `create_flexible_flat_graph()` — auto-detects grid type, selects layout | `create/archetype.py` | 8 |
| 8.2 | `create_flexible_hierarchical_graph()` — hierarchical with any layout | `create/archetype.py` | 8 |
| 8.3 | Unit + integration tests for new archetypes | `tests/test_flexible_archetypes.py` | 8 |

#### Week 9: Visualization + Documentation for wmg

| # | Task | File(s) | Hours |
|---|------|---------|-------|
| 9.1 | Update `plot_2d.py` to handle non-rectangular mesh visualizations | `visualise/plot_2d.py` | 6 |
| 9.2 | Jupyter notebook: "Creating Graphs for Irregular Data" | `docs/notebooks/` | 10 |
| 9.3 | Update README with new capabilities and usage examples | `README.md` | 4 |
| 9.4 | CHANGELOG entries for wmg | `CHANGELOG.md` | 2 |

### Phase 4: neural-lam Integration (Weeks 10–11, ~60 hours)

**Rationale:** Implement Joel's vision from Issue #83 — replace `create_graph.py` with wmg-backed pipeline.

#### Week 10: Build Graph Script + load_graph Updates

| # | Task | File(s) | Hours |
|---|------|---------|-------|
| 10.1 | Implement `build_graph.py` (replaces `create_graph.py`) | `neural_lam/build_graph.py` (NEW) | 12 |
| 10.2 | Move edge_index reindexing from `InteractionNet` to `load_graph()` (per #138) | `neural_lam/utils.py`, `neural_lam/interaction_net.py` | 8 |
| 10.3 | Add graph feature rescaling by max coordinate (per #138) | `neural_lam/utils.py` | 4 |
| 10.4 | Remove "mesh nodes MUST have first indices" constraint from `base_graph_model.py` | `neural_lam/models/base_graph_model.py` | 4 |
| 10.5 | Update CLI entry points | `pyproject.toml` | 2 |

#### Week 11: Integration Testing + Irregular Datastore Support

| # | Task | File(s) | Hours |
|---|------|---------|-------|
| 11.1 | Create `DummyIrregularDatastore` for testing | `tests/dummy_irregular_datastore.py` | 6 |
| 11.2 | Test: irregular coords → wmg graph → .pt → load_graph → model forward pass | `tests/test_flexible_pipeline.py` | 10 |
| 11.3 | Test backward compat: existing MEPS example still works with new script | `tests/test_training.py` | 6 |
| 11.4 | Fix any issues found during integration | Various | 8 |

### Phase 5: Documentation, Benchmarks & Polish (Week 12, ~25 hours)

| # | Task | File(s) | Hours |
|---|------|---------|-------|
| 12.1 | Tutorial notebook: "End-to-End: From Irregular Data to Weather Prediction" | `docs/notebooks/` | 8 |
| 12.2 | Performance benchmarks: construction time for different grid sizes/types | `benchmarks/` | 6 |
| 12.3 | API documentation review — all new functions have NumPy-style docstrings | All new files | 4 |
| 12.4 | CHANGELOG entries for neural-lam | `CHANGELOG.md` | 2 |
| 12.5 | Final review, edge case testing, PR cleanup | All | 5 |

---

## 8. Development Timeline & Deliverables

This integrated timeline maps weekly progress directly to concrete deliverables (D1–D15) across both repositories.

| Phase & Timeline | Focus Area | Key Deliverables & Milestones |
| :--- | :--- | :--- |
| **Community Bonding**<br>*(May 8 – Jun 1)* | **Pre-coding & Design** | • Weekly syncs with mentors to finalize API design<br>• POC Notebook for `mesh_layout` parameter |
| **Phase 1**<br>*(Weeks 1–3)* | **Bug Fixes & Stabilizing `wmg` v0.4** | • **D1:** Convex hull cropping (Issue #40)<br>• **D2:** G2M node assertion (Issue #42)<br>• **D3:** Level attribute fix (Issue #45)<br>🚀 **Milestone 1:** `wmg` v0.4.0 ready for release. |
| **Phase 2**<br>*(Weeks 4–6)* | **Flexible Mesh Architecture** | • **D4:** Two-step `mesh_layout` architecture (Issue #71)<br>• **D5–D8:** Triangular (Delaunay), Hexagonal, & Density-Adaptive layouts + Grid Auto-detection<br>• **D9, D10:** Prebuilt meshes & FPS multi-level coarsening |
| **Midterm Evaluation** | *Deliverable Checkpoint* | *All Phase 1/Phase 2 PRs merged & tests passing.* |
| **Phase 3**<br>*(Weeks 7–9)* | **Archetypes & Integration** | • **D11, D12:** Update existing archetypes & build new flexible ones to utilize custom layouts.<br>🚀 **Milestone 2:** Non-rectangular mesh pipelines operational. |
| **Phase 4**<br>*(Weeks 10–11)* | **Neural-LAM Bridge** | • **D13:** Create `build_graph.py` (Issue #83) replacing `create_graph.py`<br>• **D14:** `load_graph()` updates (Issue #138)<br>🚀 **Milestone 3:** End-to-End Irregular Data Forward Pass. |
| **Phase 5**<br>*(Week 12)* | **Polish & Documentation** | • **D15:** Tutorials (Irregular Data Setup, E2E Prediction)<br>• **D16:** Benchmarks and API NumPy-style Docstrings.<br>🚀 **Final Delivery** |

### Output Artifacts (Summary)
Instead of an exhaustive file tree, the final output guarantees:
1. **Core Feature:** `mesh_layout` dispatch mechanism merged into `weather-model-graphs/src/weather_model_graphs/create/base.py`.
2. **New Layout Library:** A robust `layouts.py` handling Delaunay, Hexagonal, and Density-adaptive topologies.
3. **Neural-LAM Hub:** A fully functional `build_graph.py` serving as the definitive bridge between PyG HeteroData and LAM modeling.
4. **Comprehensive Test Suite:** Integration, property-based invariants, and backwards-compatibility regression tests ensuring default arrays (like Keisler grids) remain unaffected.

---

## 11. Prior Contributions & Community Engagement

### 11.1 Issues Opened on weather-model-graphs

| Issue | Title | Status | Description |
|-------|-------|--------|-------------|
| [#70](https://github.com/mllam/weather-model-graphs/issues/70) | Support user-provided (prebuilt) mesh graphs | Open | Proposed `m2m_connectivity="prebuilt"` to inject arbitrary mesh topologies |
| [#71](https://github.com/mllam/weather-model-graphs/issues/71) | Add `mesh_layout` argument to decouple topology from connectivity | Open | Proposed the core architectural separation of layout vs connectivity |

These issues were crafted after:
- Reading the entire wmg codebase (archetype.py, base.py, mesh.py, flat.py, hierarchical.py, save.py, networkx_utils.py)
- Studying PRs [#37](https://github.com/mllam/weather-model-graphs/pull/37) (haversine), [#55](https://github.com/mllam/weather-model-graphs/pull/55) (ICON grids), and [#34](https://github.com/mllam/weather-model-graphs/pull/34) (decode mask)
- Analyzing Joel's prototype branches (`mesh_chull_filtering`, `non_decode_g2m_options`)
- Understanding how neural-lam consumes wmg output via `load_graph()` and `base_graph_model.py`

### 11.2 Codebase Familiarity Demonstrated

Through the issues and this proposal, I have demonstrated deep understanding of:

1. **The `create_all_graph_components()` orchestration flow** — how grid, mesh, g2m, and m2g components are composed
2. **The exact bottleneck** — `create_single_level_2d_mesh_graph()` with its `grid_2d_graph` dependency
3. **The KD-tree connectivity layer** — why g2m/m2g already works for irregular data
4. **The neural-lam consumption pipeline** — `load_graph()` → `BufferList` → `InteractionNet`
5. **Mentor priorities** — Issues #40, #42, #45 as v0.4.0 blockers; Issue #83 as v0.6.0 goal
6. **The scope distinction** — these are LAM adaptations, not the original global architectures (Issue #17)

### 11.3 Engagement with Community Discussions

- Referenced Issue [#17](https://github.com/mllam/weather-model-graphs/issues/17) in my issues to show awareness of scope
- Cross-referenced PRs [#37](https://github.com/mllam/weather-model-graphs/pull/37) and [#55](https://github.com/mllam/weather-model-graphs/pull/55) to show how `mesh_layout` solves problems they raise
- Connected my proposals to the GSoC "Flexible graph construction" idea explicitly
- Plan to join the MLLAM Slack GSoC channel for ongoing discussion

---

## 12. About Me

### 12.1 Background

[Write 2-3 paragraphs about your academic background, relevant coursework (graph theory, ML, numerical methods, meteorology/atmospheric science), and programming experience]

**Key technical skills relevant to this project:**
- **Python scientific stack:** numpy, scipy, networkx, matplotlib — daily usage
- **PyTorch / PyTorch Geometric:** model implementation and custom MessagePassing layers
- **Computational geometry:** Delaunay triangulation, Voronoi diagrams, convex hulls (scipy.spatial)
- **Geospatial tools:** xarray, cartopy, pyproj — atmospheric data handling
- **Software engineering:** Git workflow (branching, rebasing, PRs), pytest, CI/CD, documentation

### 12.2 Relevant Projects

[Include specific projects demonstrating:]
- Graph neural network implementation
- Computational geometry algorithms
- Scientific data processing pipelines
- Open source contributions

### 12.3 Why This Project

[Write a personal statement connecting:]
- Interest in weather/climate science and ML applications
- Experience with graph-based methods
- How this project aligns with your career goals
- What you hope to learn: production-quality open source engineering, atmospheric modeling

### 12.4 Communication Plan

| Channel | Frequency | Purpose |
|---------|-----------|---------|
| **MLLAM Slack (#gsoc)** | Daily | Quick questions, async updates |
| **Weekly written update** | Weekly | Summary of completed work, blockers, next steps |
| **Video sync with mentors** | Bi-weekly (30 min) | Design decisions, code review, feedback |
| **GitHub PRs** | Per-phase | Code review, incremental merging |
| **Availability** | [Your timezone] | Overlap with European mentors: [specify hours, e.g., 14:00–18:00 CEST] |

**My approach to mentor interaction** (based on observation):
- **Joel:** Short, focused PRs. Will tag him for correctness review, especially on connectivity and coarsening logic.
- **Leif:** Architectural decisions and milestone/roadmap alignment. Will consult on API design.
- **Hauke:** Practical robustness and upstream compatibility. Will consult on testing and edge cases.

---

## 13. Related Work & References

### 13.1 Key GitHub Issues & PRs Addressed

| Reference | Repository | Description | Role in Project |
|-----------|-----------|-------------|----------------|
| [neural-lam #83](https://github.com/mllam/neural-lam/issues/83) | neural-lam | Add graph creation using wmg | Core: Phase 4 |
| [neural-lam #138](https://github.com/mllam/neural-lam/issues/138) | neural-lam | Crucial things to merge (Joel's checklist) | Priority alignment |
| [neural-lam #164](https://github.com/mllam/neural-lam/issues/164) | neural-lam | Graph construction discussion | Motivation |
| [neural-lam #4](https://github.com/mllam/neural-lam/issues/4) | neural-lam | Adapt graph gen to general limited areas | Historical context |
| [wmg #40](https://github.com/mllam/weather-model-graphs/issues/40) | weather-model-graphs | Convex hull cropping | Phase 1 |
| [wmg #42](https://github.com/mllam/weather-model-graphs/issues/42) | weather-model-graphs | Assert all nodes in g2m | Phase 1 |
| [wmg #45](https://github.com/mllam/weather-model-graphs/issues/45) | weather-model-graphs | Level attribute consistency | Phase 1 |
| [wmg #17](https://github.com/mllam/weather-model-graphs/issues/17) | weather-model-graphs | Clarify LAM/global scope | Documentation |
| [wmg #37](https://github.com/mllam/weather-model-graphs/pull/37) | weather-model-graphs | Haversine distance (global graphs) | mesh_layout motivation |
| [wmg #55](https://github.com/mllam/weather-model-graphs/pull/55) | weather-model-graphs | ICON model grid support | Prebuilt mesh motivation |
| [wmg #34](https://github.com/mllam/weather-model-graphs/pull/34) | weather-model-graphs | Decoding mask for m2g | Already merged |
| [wmg #51](https://github.com/mllam/weather-model-graphs/pull/51) | weather-model-graphs | Fix node-drop bug | Already merged |
| [wmg #70](https://github.com/mllam/weather-model-graphs/issues/70) | weather-model-graphs | Prebuilt mesh support (mine) | Phase 2 |
| [wmg #71](https://github.com/mllam/weather-model-graphs/issues/71) | weather-model-graphs | mesh_layout decoupling (mine) | Phase 2 |

### 13.2 Academic Papers

| Paper | Relevance |
|-------|-----------|
| Keisler (2022), "Forecasting Global Weather with Graph Neural Networks" | Flat single-scale mesh. Current `create_keisler_graph()` is the LAM adaptation. |
| Lam et al. (2023), "GraphCast" | Multi-scale icosahedral mesh. Current `create_graphcast_graph()` is the rectangular adaptation. |
| Oskarsson et al. (2023), "Graph-based Neural Weather Prediction for Limited Area Modeling" | Hierarchical mesh. Core architecture of neural-lam. Directly authored by mentor Joel. |
| Oskarsson et al. (2024), "Probabilistic Weather Forecasting with Hierarchical Graph Neural Networks" | Extended hierarchical approach; demonstrates the need for flexible graph structures. |
| Battaglia et al. (2016), "Interaction Networks" | Foundation for `InteractionNet` GNN used in message passing. |
| Bridson (2007), "Fast Poisson disk sampling in arbitrary dimensions" | Algorithm for uniform node placement. |

### 13.3 Software Dependencies

| Library | Version | Usage |
|---------|---------|-------|
| `scipy.spatial.Delaunay` | ≥1.13.0 | Triangulation-based mesh connectivity |
| `scipy.spatial.Voronoi` | ≥1.13.0 | Density estimation for adaptive placement |
| `scipy.spatial.ConvexHull` | ≥1.13.0 | Convex hull cropping (Issue #40) |
| `scipy.spatial.KDTree` | ≥1.13.0 | Already used for g2m/m2g connectivity |
| `networkx` | ≥3.0 | Primary graph data structure |
| `numpy` | ≥1.26.4 | Array operations |
| `torch_geometric` | optional | Output format via `save.to_pyg()` |

---

## 14. Appendix: Architecture Diagrams & Technical Details

### A1. Overall System Architecture: Current vs Proposed

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          CURRENT SYSTEM                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   Data Source              Graph Construction              ML Model        │
│   ┌──────────────┐        ┌────────────────────┐         ┌──────────────┐  │
│   │ Regular Grid │───────>│ create_graph.py    │────────>│ GraphLAM     │  │
│   │ Only         │[Nx,Ny] │ (neural-lam)       │ .pt     │ HiLAM        │  │
│   │              │        │ DUPLICATES wmg     │ files   │              │  │
│   └──────────────┘        └────────────────────┘         └──────────────┘  │
│                                                                            │
│   ┌──────────────┐        ┌────────────────────┐                           │
│   │ weather-     │        │ Mostly UNUSED by   │                           │
│   │ model-graphs │        │ neural-lam         │                           │
│   └──────────────┘        └────────────────────┘                           │
│                                                                            │
│   ┌──────────────┐        ┌────────────────────┐                           │
│   │ Irregular    │───✗───>│ NotImplementedError│                           │
│   │ Data         │        └────────────────────┘                           │
│   └──────────────┘                                                         │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

                               ▼ ▼ ▼

┌────────────────────────────────────────────────────────────────────────────┐
│                          PROPOSED SYSTEM                                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   Data Source              Graph Construction              ML Model        │
│   ┌──────────────┐        ┌────────────────────────────┐  ┌──────────────┐ │
│   │ Regular Grid │──┐     │ weather-model-graphs       │  │ GraphLAM     │ │
│   └──────────────┘  │     │ ┌────────────────────────┐ │  │ HiLAM        │ │
│                     ├────>│ │ create_all_graph_       │ │ │ HiLAMParallel│ │
│   ┌──────────────┐  │     │ │   components()         │ │  │              │ │
│   │ Irregular    │──┤     │ │                         │ │  │             │ │
│   │ Grid         │  │     │ │ mesh_layout= ← NEW     │ │  └──────▲───────┘ │
│   └──────────────┘  │     │ │ crop_hull=   ← NEW     │ │         │         │
│                     │     │ │ assert_g2m=  ← NEW     │ │   .pt files       │
│   ┌──────────────┐  │     │ └────────────────────────┘ │         │         │
│   │ Scattered    │──┘     └────────────┬───────────────┘  ┌──────┴───────┐ │
│   │ Observations │                     │                  │ build_graph  │ │
│   └──────────────┘                     │                  │ .py (new)    │ │
│                            [N, 2]      │                  │ calls wmg    │ │
│                            coords      │                  └──────────────┘ │
│                                        │                                   │
│                                        ▼                                   │
│                               wmg.save.to_pyg()                            │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### A2. Mesh Generation Pipeline Detail

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  FLEXIBLE MESH GENERATION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT: xy coordinates [N, 2]                                           │
│  │                                                                      │
│  ▼                                                                      │
│  ┌───────────────────────────────┐                                      │
│  │ 1. DETECT GRID TYPE (optional)│                                      │
│  │    Compute NN distances        │      ┌─────────────────┐            │
│  │    Check CV of spacing         │─────>│ GridType enum   │            │
│  │    Check reshape-ability       │      │ RECTANGULAR     │            │
│  └───────────────────────────────┘      │ IRREGULAR_UNIFORM│            │
│                                          │ IRREGULAR_CLUST. │            │
│                                          │ SPARSE           │            │
│                                          └────────┬────────┘            │
│                                                   │                     │
│  ┌────────────────────────────────────────────────┼─────────────────┐   │
│  │ 2. SELECT mesh_layout STRATEGY                 ▼                 │   │
│  │                                                                  │   │
│  │  RECTANGULAR ──────> grid_2d_graph (existing, backward compat)   │   │
│  │  TRIANGULAR ───────> Poisson disk + Delaunay triangulation       │   │
│  │  HEXAGONAL ────────> Hexagonal lattice placement + Delaunay      │   │
│  │  DENSITY_ADAPTIVE ─> Voronoi density → variable Poisson disk     │   │
│  │  FROM_GRAPH ───────> User-provided mesh graph (Issue #70)        │   │
│  │  (or user override via mesh_layout parameter)                    │   │
│  └────────────────────────────────────────────────┬─────────────────┘   │
│                                                   │                     │
│                                                   ▼                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 3. OPTIONAL: CROP TO CONVEX HULL (Issue #40)                    │   │
│  │    Remove mesh nodes outside grid point convex hull + margin    │   │
│  └─────────────────────────────────────┬────────────────────────────┘   │
│                                        │                                │
│                                        ▼                                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 4. BUILD CONNECTIVITY (based on m2m_connectivity)               │   │
│  │                                                                  │   │
│  │  FLAT ─────────────> Keep single-level mesh as-is                │   │
│  │  FLAT_MULTISCALE ──> FPS coarsen → Delaunay at each level       │   │
│  │                      → merge all levels into one graph           │   │
│  │  HIERARCHICAL ─────> FPS coarsen → Delaunay at each level       │   │
│  │                      → add up/down inter-level edges             │   │
│  │                      → from_level, to_level attributes (#45)     │   │
│  │  PREBUILT ─────────> Use as-is (Issue #70)                       │   │
│  └─────────────────────────────────────┬────────────────────────────┘   │
│                                        │                                │
│                                        ▼                                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 5. CONNECT GRID ↔ MESH (existing — already flexible!)          │   │
│  │    g2m: KD-tree based (nearest_neighbour/s, within_radius)      │   │
│  │    m2g: KD-tree based (nearest_neighbour/s, within_radius)      │   │
│  └─────────────────────────────────────┬────────────────────────────┘   │
│                                        │                                │
│                                        ▼                                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 6. VALIDATE (Issue #42)                                         │   │
│  │    Assert all grid nodes have ≥1 g2m edge                       │   │
│  │    Assert all mesh nodes have ≥1 g2m edge                       │   │
│  │    Optionally auto-fix with nearest-neighbor fallback           │   │
│  └─────────────────────────────────────┬────────────────────────────┘   │
│                                        │                                │
│                                        ▼                                │
│  OUTPUT: networkx.DiGraph (combined) or dict of components              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### A3. Data Type Support Matrix (Before → After)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA TYPE SUPPORT MATRIX                          │
├──────────────────────────┬──────────┬───────────┬───────────────────────┤
│ Data Type                │ Before   │ After     │ Layout Used           │
├──────────────────────────┼──────────┼───────────┼───────────────────────┤
│ Regular Rectangular Grid │ ✅ Full  │ ✅ Full   │ rectilinear (default) │
│ Regular Hexagonal Grid   │ ❌       │ ✅ Full   │ triangular            │
│ Reduced Gaussian Grid    │ ❌       │ ✅ Full   │ triangular            │
│ ICON Icosahedral Grid    │ ❌       │ ✅ Full   │ from_graph / triang.  │
│ MPAS Unstructured Mesh   │ ❌       │ ✅ Full   │ from_graph            │
│ Weather Station Network  │ ❌       │ ✅ Full   │ density_adaptive      │
│ Satellite Swath Data     │ ❌       │ ✅ Full   │ density_adaptive      │
│ Ship/Buoy Observations   │ ❌       │ ✅ Full   │ density_adaptive      │
│ Mixed-Source Blend       │ ❌       │ ✅ Full   │ density_adaptive      │
│ User-Provided Custom     │ ❌       │ ✅ Full   │ from_graph (#70)      │
├──────────────────────────┼──────────┼───────────┼───────────────────────┤
│ Graph Types Available:   │          │           │                       │
│  Flat (single-scale)     │ ✅ Rect  │ ✅ Any    │                       │
│  Flat multiscale         │ ✅ Rect  │ ✅ Any    │                       │
│  Hierarchical            │ ✅ Rect  │ ✅ Any    │                       │
│  Prebuilt (custom)       │ ❌       │ ✅ Any    │                       │
└──────────────────────────┴──────────┴───────────┴───────────────────────┘
```

### A4. Computational Complexity Analysis

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    COMPUTATIONAL COMPLEXITY                               │
├──────────────────────────────────┬───────────────┬───────────────────────┤
│ Operation                        │ Complexity    │ Typical Time          │
├──────────────────────────────────┼───────────────┼───────────────────────┤
│ Grid type detection              │ O(N log N)    │ <1s for N=100k       │
│ Poisson disk sampling            │ O(M)          │ <2s for M=10k mesh   │
│ Density-adaptive sampling        │ O(N log N)    │ <3s for N=100k data  │
│ Delaunay triangulation           │ O(M log M)    │ <1s for M=10k mesh   │
│ Convex hull computation          │ O(N log N)    │ <0.5s for N=100k     │
│ Farthest point sampling          │ O(M × K)      │ <2s for M=10k, K=5   │
│ KD-tree construction (existing)  │ O(N log N)    │ <1s for N=100k       │
│ G2M/M2G connectivity (existing)  │ O(N log M)    │ <3s for N=100k       │
├──────────────────────────────────┼───────────────┼───────────────────────┤
│ Total pipeline (typical):        │               │ <15s for N=100k      │
├──────────────────────────────────┴───────────────┴───────────────────────┤
│ N = data points, M = mesh nodes, K = hierarchy levels                    │
│ Memory: O(N + M + E), where E ≈ 6M for Delaunay (avg degree ~6)        │
└──────────────────────────────────────────────────────────────────────────┘
```

### A5. Poisson Disk Sampling Algorithm (Bridson 2007)

```python
def _poisson_disk_sample(bounds, radius, k=30, seed=None):
    """
    Bridson's algorithm for Poisson disk sampling.
    Time: O(N), Space: O(N) where N = number of generated points
    
    Used for mesh node placement with guaranteed minimum spacing.
    """
    rng = np.random.default_rng(seed)
    xmin, ymin, xmax, ymax = bounds
    cell_size = radius / np.sqrt(2)
    
    # Background grid for O(1) neighbor lookups
    nx = int(np.ceil((xmax - xmin) / cell_size))
    ny = int(np.ceil((ymax - ymin) / cell_size))
    grid = -np.ones((nx, ny), dtype=int)
    
    points = []
    active = []
    
    # Seed point
    p0 = rng.uniform([xmin, ymin], [xmax, ymax])
    points.append(p0)
    active.append(0)
    gi, gj = _grid_coords(p0, xmin, ymin, cell_size)
    grid[gi, gj] = 0
    
    while active:
        idx = rng.integers(len(active))
        point = points[active[idx]]
        found = False
        
        for _ in range(k):
            # Sample from annulus [radius, 2*radius]
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(radius, 2 * radius)
            candidate = point + dist * np.array([np.cos(angle), np.sin(angle)])
            
            if (_in_bounds(candidate, bounds) and 
                _no_nearby_points(candidate, grid, points, radius, cell_size, xmin, ymin)):
                points.append(candidate)
                active.append(len(points) - 1)
                gi, gj = _grid_coords(candidate, xmin, ymin, cell_size)
                grid[gi, gj] = len(points) - 1
                found = True
                break
        
        if not found:
            active.pop(idx)
    
    return np.array(points)
```

---

## Summary

This proposal directly addresses [GSoC Idea #1: Flexible Graph Construction](https://github.com/mllam/neural-lam/wiki/GSoC-ideas#1-flexible-graph-construction) by:

1. **Resolving v0.4.0 blockers** ([#40](https://github.com/mllam/weather-model-graphs/issues/40), [#42](https://github.com/mllam/weather-model-graphs/issues/42), [#45](https://github.com/mllam/weather-model-graphs/issues/45)) that both Joel and Leif have prioritized
2. **Decoupling mesh layout from connectivity** ([#71](https://github.com/mllam/weather-model-graphs/issues/71)) — the core architectural change enabling flexibility
3. **Supporting prebuilt meshes** ([#70](https://github.com/mllam/weather-model-graphs/issues/70)) — enabling ICON, MPAS, and custom mesh topologies
4. **Implementing Delaunay, hexagonal, and density-adaptive mesh layouts** — covering irregular NWP grids to sparse observations
5. **Replacing `create_graph.py` with wmg-backed pipeline** ([#83](https://github.com/mllam/neural-lam/issues/83)) — completing the wmg integration per Joel's [#138](https://github.com/mllam/neural-lam/issues/138) checklist

Every deliverable maps to an open issue or documented mentor priority. The implementation preserves full backward compatibility while enabling graph construction from **any spatial data distribution** — from regular rectangular grids to scattered ship observations.

---

*Proposal prepared by: Prajwal [Your Last Name]*  
*Last updated: February 27, 2026*  
*GitHub: [github.com/prajwal-tech07](https://github.com/prajwal-tech07)*
