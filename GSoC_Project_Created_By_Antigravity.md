# Google Summer of Code 2026 Proposal

## Flexible Graph Construction: A Unified Pipeline for Universal Graph Topologies in Neural Weather Prediction

| | |
| :--- | :--- |
| <ul><li>**Organization:** MLLAM (Machine Learning for Limited Area Models)</li></ul> | <ul><li>**Project Length:** 350 hours (Large)</li></ul> |
| <ul><li>**Project:** [Flexible Graph Construction](https://github.com/mllam/neural-lam/wiki/GSoC-ideas#1-flexible-graph-construction) (Idea #1)</li></ul> | <ul><li>**Difficulty:** Medium</li></ul> |
| <ul><li>**Repositories:** [weather-model-graphs](https://github.com/mllam/weather-model-graphs) (WMG), [neural-lam](https://github.com/mllam/neural-lam)</li></ul> | <ul><li>**Mentors:** Hauke Schulz ([@observingClouds](https://github.com/observingClouds)), Leif Denby ([@leifdenby](https://github.com/leifdenby)), Joel Oskarsson ([@joeloskarsson](https://github.com/joeloskarsson))</li></ul>

---

## Table of Contents

1. [About Me](#1-about-me)
2. [Community Engagement](#2-community-engagement)
3. [Project Abstract](#3-project-abstract)
4. [Motivation & Problem Statement](#4-motivation--problem-statement)
5. [Current Architecture Deep-Dive](#5-current-architecture-deep-dive)
6. [Proposed Solution & Technical Design](#6-proposed-solution--technical-design)
7. [Advanced Research Contributions (Layers 4 & 5)](#7-advanced-research-contributions-layers-4--5)
8. [Weekly Timeline](#8-weekly-timeline)
9. [Testing Strategy](#9-testing-strategy)
10. [Deliverables](#10-deliverables)
11. [Risk Mitigation](#11-risk-mitigation)
12. [References](#12-references)
13. [Other Commitments](#13-other-commitments)

---

## 1. About Me

| Field | Details |
|-------|---------|
| **Name** | Prajwal [Your Last Name] |
| **GitHub** | [prajwal-tech07](https://github.com/prajwal-tech07) |
| **University** | [Your University] |
| **Degree / Year** | [e.g., B.Tech Computer Science, 3rd year] |
| **Expected Graduation** | [e.g., May 2027] |
| **Email** | [your.email@example.com] |
| **Timezone** | UTC+5:30 (IST) |
| **Available hours/week** | 30–35 hours |

[Write 2–3 paragraphs about your academic background, relevant coursework (graph theory, ML, numerical methods, atmospheric science), programming experience, and why this project excites you.]

**Key technical skills directly relevant to this project:**
- **Graph neural networks:** PyTorch Geometric, custom `MessagePassing` layers, `HeteroData` objects
- **Computational geometry:** Delaunay triangulation, Voronoi diagrams, convex hulls, spectral mesh analysis (`scipy.spatial`)
- **Scientific Python:** numpy, scipy, networkx, xarray, cartopy, pyproj
- **Software engineering:** Git branching/rebasing, pytest, CI/CD, NumPy-style docstrings

<div style="page-break-after: always;"></div>

## 2. Community Engagement

I have been actively contributing to both repositories with **substantive architectural PRs** — not cosmetic fixes:

> **Strategic Vision & Core Insight:** Bridging the gap via `HeteroData`
> Through deep engagement with the codebase and active community discussions, I conceptualized and proposed the architectural idea of "bridging the gap" between WMG and Neural-LAM by migrating to PyTorch Geometric's `HeteroData` structure. This shift—from fragmented multi-tensor storage to a unified, heterogeneous graph foundation—streamlines the graph construction pipeline and serves as the driving centerpiece of this GSoC proposal.

### 2.1 weather-model-graphs Contributions

| PR / Issue | Title | Status | Impact & Milestone |
|------------|-------|--------|---------------------|
| [**PR #81**](https://github.com/mllam/weather-model-graphs/pull/81) | `mesh_layout` two-step architecture | **Under review** *(leifdenby: "95% done, well done!")* | **Core refactor** — decouples layout & connectivity (Target: v0.4) |
| [PR #91](https://github.com/mllam/weather-model-graphs/pull/91) | `mesh_layout='prebuilt'` support | Open | Enables arbitrary mesh injection (Target: v0.5) |
| [PR #92](https://github.com/mllam/weather-model-graphs/pull/92) | `mesh_layout='triangular'` (Delaunay) | Open | Enables non-rectangular meshes (Target: v0.5) |
| [Issue #97](https://github.com/mllam/weather-model-graphs/issues/97) | `validate_graph_components()` | Open | Pre-export structural validation |
| [Issue #98](https://github.com/mllam/weather-model-graphs/issues/98) | Node-ID-to-tensor-index mapping | Open | Lossless WMG ↔ neural-lam round-trips |

### 2.2 neural-lam Contributions

| PR | Title | Status | Impact |
|----|-------|--------|--------|
| [**PR #258**](https://github.com/mllam/neural-lam/pull/258) | Area weights for metric computation | **Under review** | `cos(lat)` weighting through all 6 metrics |

### 2.3 Depth of Understanding

These contributions required reading **every source file** in both repos — `create/base.py`, `coords.py`, `flat.py`, `hierarchical.py`, `save.py`, `networkx_utils.py` in WMG; `create_graph.py` (614 lines), `base_graph_model.py`, `graph_lam.py`, `hi_lam.py`, `hi_lam_parallel.py`, `utils.py`, `metrics.py`, `interaction_net.py` in neural-lam.

<div style="page-break-after: always;"></div>

## 3. Project Abstract

> *"The challenge is to explore and implement a methodology that can create well-balanced neural network grids based on different data structures, from irregularly structured atmospheric model output to sparse ship-observations."*
> — [GSoC Ideas Page](https://github.com/mllam/neural-lam/wiki/GSoC-ideas#1-flexible-graph-construction)

This project delivers a **five-layer solution**. Layers 1–3 are the **core deliverables** (guaranteed within GSoC). Layers 4 and 5 are **modular stretch goals** — each is self-contained and can be tackled independently in the final weeks once the foundation is merged.

```mermaid
graph TD

%% ---------- STYLE DEFINITIONS ----------
classDef layer fill:#ffffff,stroke:#3b82f6,stroke-width:2px,color:#1e293b,rx:8px
classDef titleStyle fill:#1e3a8a,color:#ffffff,stroke:none,font-weight:bold,rx:4px
classDef highlight fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#0f172a,rx:8px

%% ---------- TOP SECTION: DELIVERABLES ----------
subgraph TRACKS [" "]
    direction LR
    CORE["★ CORE DELIVERABLES<br/>(Guaranteed, Weeks 1–6)"]:::titleStyle
    STRETCH["★ STRETCH GOALS<br/>(Modular, Weeks 7–11)"]:::titleStyle
end

%% ---------- MIDDLE SECTION: CORE LAYERS ----------
subgraph CORE_ROW ["CORE ARCHITECTURE"]
    direction TB
    L1["<div style='width:320px; text-align:left; padding:5px;'> <b>Layer 1: Foundation</b><br/>● mesh_layout PR#81 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br/>● triangular PR#92 (prebuilt PR#91)<br/>● v0.4.0 fixes #40/#42/#45</div>"]:::layer
    
    L2["<div style='width:320px; text-align:left; padding:5px;'> <b>Layer 2: Bridge — Issue #384</b><br/>● build_graph.py replaces create_graph.py<br/>● GraphFormatValidator<br/>● Enhanced to_pyg()</div>"]:::layer
    
    L3["<div style='width:320px; text-align:left; padding:5px;'> <b>Layer 3: Architecture — Issue #385</b><br/>● pyg.HeteroData migration<br/>● adapter → refactor → native loading</div>"]:::layer
end

%% ---------- BOTTOM SECTION: STRETCH LAYERS ----------
subgraph ADV_ROW ["ADVANCED RESEARCH"]
    direction TB
    L4["<div style='width:320px; text-align:left; padding:5px;'> <b>Layer 4: Advanced Research</b><br/>● Quality Metrics · AMR · xr.DataTree<br/>● Multi-source fusion<br/>● Density-adaptive</div>"]:::layer
    
    L5["<div style='text-align:left; width:320px; padding:5px;'> <b>Layer 5: Cutting-Edge Innovations</b><br/>● Spherical Coordination System<br/>● Topology Benchmark<br/>● Learned coarsening  ● Stretched-grid<br/>● Dynamic edges  ● Analysis dashboard</div>"]:::layer
end

%% ---------- CONNECTIONS ----------
CORE ==> CORE_ROW
STRETCH ==> ADV_ROW

%% Sequence Flow
L1 --> L2
L2 --> L3
L3 -.-> L4
L4 --> L5

%% ---------- STYLING ----------
style TRACKS fill:none,stroke:none
style CORE_ROW fill:#f8fafc,stroke:#cbd5e1
style ADV_ROW fill:#f8fafc,stroke:#cbd5e1
```

<div style="page-break-after: always;"></div>

## 4. Motivation & Problem Statement

### 4.1 The Encode-Process-Decode Architecture

neural-lam uses an **Encoder-Processor-Decoder** GNN architecture. Atmospheric variables at grid nodes are encoded into a latent space, processed on a mesh graph through multiple rounds of message passing, and decoded back to grid predictions. The architecture uses three graph components: **g2m** (grid-to-mesh encoder edges), **m2m** (mesh-to-mesh processor edges), and **m2g** (mesh-to-grid decoder edges).

```mermaid
flowchart LR

%% ---------- STYLE DEFINITIONS ----------
classDef dataNode fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#0f172a,rx:8px
classDef modelNode fill:#ffffff,stroke:#d1d5db,stroke-width:1px,color:#1f2937,rx:4px
classDef meshNode fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,font-weight:bold,color:#0f172a,rx:8px

%% ---------- NODES ----------
A[" <b>Grid Nodes</b><br/>(atmospheric data points<br/>at lat/lon positions)"]:::dataNode

B["<b>g2m Encoder</b><br/>(grid → mesh<br/>KD-tree nearest)"]:::modelNode

subgraph ENGINE ["Neural Processing Engine"]
    %% This direction strictly controls the loop, forcing a clean, tight curve
    direction TB
    
    C[" <b>Mesh Nodes</b><br/>(processor graph<br/>m2m message passing)"]:::meshNode
    
    %% Standard, safe syntax for the loop
    C -- "m2m × K rounds" --> C
end

D["<b>m2g Decoder</b><br/>(mesh → grid<br/>KD-tree nearest)"]:::modelNode

E[" <b>Predictions</b><br/>(forecast at each<br/>grid point)"]:::dataNode

%% ---------- CONNECTIONS ----------
A --> B
B --> C
C --> D
D --> E

%% ---------- STYLING ----------
style ENGINE fill:#f8fafc,stroke:#cbd5e1,stroke-dasharray: 5 5
```

The mesh topology **directly determines** what the model can learn. The number of message-passing steps needed for information to traverse the domain equals the graph diameter. Edge features encode spatial relationships. Yet today, **only regular rectangular grids** are supported, because `create_graph.py` hardcodes `networkx.grid_2d_graph(Nx, Ny)`.

### 4.2 The Two-Step Mesh Architecture (My PR #81)

My core architectural contribution in PR #81 separates mesh creation into two independent, composable steps:

```mermaid
flowchart LR
    %% Define clean, professional styling
    classDef default fill:#ffffff,stroke:#d1d5db,stroke-width:1px,color:#374151,rx:4px,ry:4px;
    classDef core fill:#e1f5fe,stroke:#0288d1,stroke-width:1.5px,color:#0f172a,rx:4px,ry:4px;

    %% Input Nodes
    A1["rectilinear<br/>(current default)"]
    A2["triangular<br/>(Delaunay PR#92)"]
    A3["prebuilt<br/>(ICON/MPAS PR#91)"]
    A4["density_adaptive ★"]
    A5["stretched ★"]

    %% Core Process Nodes
    S1["Step 1: Coordinate<br/>Creation<br/>mesh_layout parameter"]:::core
    S2["Step 2: Connectivity<br/>Creation<br/>m2m_connectivity parameter"]:::core

    %% Output Nodes
    B1["flat"]
    B2["flat_multiscale"]
    B3["hierarchical"]

    %% Connections (written individually to prevent rendering errors)
    A1 --> S1
    A2 --> S1
    A3 --> S1
    A4 --> S1
    A5 --> S1
    
    S1 --> S2
    
    S2 --> B1
    S2 --> B2
    S2 --> B3
```

Any `mesh_layout` can combine with any `m2m_connectivity` — creating a **combinatorial explosion** of graph topologies from minimal code. This is the foundation upon which ALL other contributions in this proposal are built.

### 4.3 Current Limitations

| # | Problem | Impact | Root Cause |
|---|---------|--------|------------|
| 1 | `create_graph.py` hardcodes `nx.grid_2d_graph` | Only rectangular grids work | `networkx.grid_2d_graph(Nx, Ny)` called directly |
| 2 | 600+ lines duplicated between repos | Maintenance nightmare, divergent behavior | neural-lam reimplements WMG logic |
| 3 | `load_graph()` returns fragile `dict` | 11 raw string keys, no type safety, no validation | No `pyg.HeteroData` adoption |
| 4 | No quality evaluation for meshes | Users can't compare topologies without training | No metrics framework exists |
| 5 | Euclidean distances at high latitudes | Systematic distortion (2× at lat=60°, 5.7× at lat=80°) | No spherical coordinate support |
| 6 | No adaptive mesh refinement | Can't densify mesh in high-error regions | Static mesh construction only |

### 4.4 The Code Duplication Problem

```mermaid
flowchart TD
    %% Professional Class Definitions
    classDef default fill:#ffffff,stroke:#64748b,stroke-width:1.5px,rx:4px,color:#334155
    classDef highlight fill:#f0f9ff,stroke:#0284c7,stroke-width:1.5px,rx:4px,color:#0c4a6e
    classDef danger fill:#fff1f2,stroke:#e11d48,stroke-width:2px,rx:4px,color:#9f1239

    %% Source of Truth Subgraph
    subgraph WMG["weather-model-graphs (source of truth)"]
        W1["create/base.py<br/>create_all_graph_components()"]
        W2["save.py → to_pyg()<br/>Exports .pt files"]
    end

    %% Duplicated Code Subgraph
    subgraph NL["neural-lam (DUPLICATED CODE)"]
        N1["create_graph.py — 614 lines<br/>Reimplements WMG logic<br/>Hardcodes nx.grid_2d_graph"]:::danger
        N2["utils.py → load_graph()<br/>11 fragile string keys"]
    end

    %% Routing / Edges
    W1 -.->|"should be single<br/>source of truth"| N1
    W2 -->|".pt files on disk"| N2
    N1 -->|"❌ NotImplementedError<br/>for irregular data"| X["BLOCKED"]:::danger

    %% Subgraph Visual Styling (Clean, dashed borders)
    style WMG fill:#f8fafc,stroke:#cbd5e1,stroke-width:2px,stroke-dasharray: 5 5,color:#1e293b
    style NL fill:#fcf0f0,stroke:#fecdd3,stroke-width:2px,stroke-dasharray: 5 5,color:#9f1239
```

The **core blocking line** in neural-lam:
```python
# neural-lam/create_graph.py — THE line that blocks all irregular data:
grid_graph = networkx.grid_2d_graph(xy.shape[1], xy.shape[2])  # ← RECTANGULAR ONLY
# ... [remainder builds g2m/m2m/m2g from this rectangular assumption]
```

### 4.5 A Key Opportunity for Enhancement: Quality Guarantees

Once flexible meshes are enabled, a natural next step is providing a mechanism to evaluate whether a generated mesh is well-suited for message-passing. Currently, users hve no qauantitative guidance on questions like:

- Does the mesh have uniform edge-length distribution? (isotropy)
- Does the mesh cover the data domain without gaps? (coverage)
- Is the mesh well-conditioned for message-passing? (spectral gap)
- Does a denser mesh in region X actually improve prediction there? (adaptive value)

This is a natural extension of the flexible graph construction work — as a stretch goal, I propose a **Graph Quality Metrics Framework** that would give users quantitative answers to these questions before committing to expensive model training.

---

## 5. Current Architecture Deep-Dive

### 5.1 Current End-to-End Data Flow

```mermaid
flowchart TD
    %% Professional Class Styling
    classDef default fill:#f8fafc,stroke:#94a3b8,stroke-width:2px,color:#0f172a,rx:8px
    classDef danger fill:#fef2f2,stroke:#ef4444,stroke-width:2px,color:#991b1b,rx:8px
    classDef highlight fill:#f0fdf4,stroke:#22c55e,stroke-width:2px,color:#14532d,rx:8px

    %% Column 1 (Forces nodes to stack vertically on the left)
    subgraph Left_Column [" "]
        direction TB
        DS["<b>DataStore</b><br/>(BaseRegularGrid)<br/>get_xy() → [2, Nx, Ny]"]
        CG["<b>create_graph.py</b><br/>(neural-lam, 614 lines)<br/>DUPLICATES WMG logic"]:::danger
        LG["<b>load_graph()</b><br/>→ dict of 11 tensors<br/>no validation"]
    end

    %% Column 2 (Forces nodes to stack vertically on the right)
    subgraph Right_Column [" "]
        direction TB
        IRR["<b>Irregular Data</b><br/>Stations, satellites,<br/>ship tracks"]:::danger
        PT["<b>.pt files on disk</b><br/>g2m/m2m/m2g edge_index<br/>+ features"]
        MODEL["<b>GraphLAM / HiLAM</b><br/>BaseGraphModel"]:::highlight
    end

    %% Logical Connections (Creates a compact Z-Pattern)
    DS --> CG
    IRR -.->|"❌ Error"| CG
    
    CG --> PT
    
    PT --> LG
    LG --> MODEL

    %% Make the boundary boxes completely invisible
    style Left_Column fill:none,stroke:none
    style Right_Column fill:none,stroke:none
```

### 5.2 Data Source Support Matrix: Before vs After

| Data Type | Before | After | Layout Used |
|-----------|--------|-------|-------------|
| Regular Rectangular Grid | ✅ WORKS | ✅ WORKS | `rectilinear` (default) |
| Regular Hexagonal Grid | ❌ BLOCKED | ✅ WORKS | `triangular` |
| Reduced Gaussian Grid | ❌ BLOCKED | ✅ WORKS | `triangular` |
| ICON Icosahedral Grid | ❌ BLOCKED | ✅ WORKS | `prebuilt` / `triangular` |
| MPAS Unstructured Mesh | ❌ BLOCKED | ✅ WORKS | `prebuilt` |
| Weather Station Network | ❌ BLOCKED | ✅ WORKS | `density_adaptive` ★ |
| Satellite Swath Data | ❌ BLOCKED | ✅ WORKS | `density_adaptive` ★ |
| Ship/Buoy Observations | ❌ BLOCKED | ✅ WORKS | `density_adaptive` ★ |
| Multi-Source Blended | ❌ BLOCKED | ✅ WORKS | `multi_source` + adapt. ★ |

<div style="page-break-after: always;"></div>

## 6. Proposed Solution & Technical Design

### 6.1 Proposed End-to-End Flow (replaces Section 5.1)

```mermaid
flowchart TD
    %% Professional Color Palette
    classDef source fill:#f8fafc,stroke:#94a3b8,stroke-width:1px,rx:6px,color:#334155
    classDef engine fill:#eff6ff,stroke:#3b82f6,stroke-width:2px,rx:6px,color:#1e3a8a,font-weight:bold
    classDef validate fill:#fff7ed,stroke:#f97316,stroke-width:2px,rx:6px,color:#9a3412,font-weight:bold
    classDef runtime fill:#f0fdf4,stroke:#22c55e,stroke-width:2px,rx:6px,color:#14532d,font-weight:bold

    %% Row 1: Ingestion
    subgraph Layer1 ["Data Ingestion & Build"]
        direction LR
        S1["Regular Grid"]:::source
        S2["Irregular Grid"]:::source
        S3["Scattered Obs"]:::source
        S4["Multi-Source"]:::source
        BG["build_graph.py\n(NEW — replaces\ncreate_graph.py)"]:::engine
        
        S1 --> BG
        S2 --> BG
        S3 --> BG
        S4 --> BG
    end

    %% Row 2: Construction & Validation
    subgraph Layer2 ["Graph Construction & Validation"]
        direction LR
        WMG["weather-model-graphs\ncreate_all_graph_components()\nmesh_layout + m2m_connectivity"]:::engine
        VAL["validate() + to_pyg()\nGraphFormatValidator"]:::validate
        
        WMG --> VAL
    end

    %% Row 3: Runtime
    subgraph Layer3 ["Model Runtime"]
        direction LR
        LG["load_graph()\n→ pyg.HeteroData"]:::runtime
        MODEL["GraphLAM / HiLAM\nHeteroData-native"]:::runtime
        
        LG --> MODEL
    end

    %% Inter-Layer Connections
    Layer1 --> Layer2
    Layer2 -->|".pt files"| Layer3

    %% Clean, solid borders for a polished professional look
    style Layer1 fill:#ffffff,stroke:#cbd5e1,stroke-width:2px,rx:8px
    style Layer2 fill:#ffffff,stroke:#cbd5e1,stroke-width:2px,rx:8px
    style Layer3 fill:#ffffff,stroke:#cbd5e1,stroke-width:2px,rx:8px
```

### 6.2 Layer 1: Foundation (My Existing PRs + v0.4.0 Fixes)

**PR #81 — `mesh_layout` parameter:** Decouples coordinate creation from connectivity creation. The `mesh_layout` parameter (`"rectilinear"`, `"triangular"`, `"prebuilt"`) controls WHERE mesh nodes are placed. The `m2m_connectivity` parameter (`"flat"`, `"flat_multiscale"`, `"hierarchical"`) controls HOW they are connected. This two-step architecture is the foundation for all other work.

**PR #92 — Triangular Delaunay mesh:** Adds `mesh_layout="triangular"` using `scipy.spatial.Delaunay` triangulation. This enables non-rectangular meshes for hexagonal, reduced Gaussian, and icosahedral grids.

**PR #91 — Prebuilt mesh pathway:** Adds `mesh_layout="prebuilt"` allowing users to inject arbitrary mesh node positions (e.g., from ICON or MPAS model grids).

**v0.4.0 blockers:**
- **#40 — Convex hull cropping:** `crop_mesh_to_convex_hull()` via `scipy.spatial.ConvexHull` to remove mesh nodes outside the data domain.
- **#42 — G2M assertion:** Detect and auto-fix degree-0 mesh nodes in g2m connections.
- **#45 — Level attributes:** Replace inconsistent `"level"`(int)/`"levels"`(str) with `from_level`/`to_level` (both int).

### 6.3 Layer 2: The Bridge (Issue #384)

The bridge eliminates the 600-line code duplication between repos by making neural-lam call WMG directly.

```mermaid
flowchart LR
    %% Professional Color Palette
    classDef wmg fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1,rx:4px
    classDef contract fill:#F5F5F5,stroke:#424242,stroke-width:2px,color:#212121,rx:4px
    classDef nl fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20,rx:4px

    subgraph WMG ["WMG Side"]
        direction TB
        V["validate_graph_components()<br/>• All nodes have pos, type attrs<br/>• No degree-0 in g2m<br/>• Level attrs consistent"]:::wmg
        TP["to_pyg() enhanced<br/>• Node-ID map preserved (#98)<br/>• Format matches schema"]:::wmg
        
        %% Internal vertical stacking
        V --> TP
    end

    GFV["GraphFormatValidator<br/>(Shared Contract)"]:::contract

    subgraph NL ["neural-lam Side"]
        direction TB
        BG["build_graph.py (NEW)<br/>• Calls WMG directly<br/>• Replaces create_graph.py"]:::nl
        LG["load_graph() enhanced<br/>• Reindexes edge_index<br/>• Rescales features<br/>• Returns HeteroData"]:::nl
        
        %% Internal vertical stacking
        BG --> LG
    end

    %% Connect the outer containers to force a straight horizontal line
    WMG --> GFV
    GFV --> NL

    %% Styling for the outer dashed boxes
    style WMG fill:#FAFAFA,stroke:#BDBDBD,stroke-dasharray: 5 5,color:#424242,rx:8px
    style NL fill:#FAFAFA,stroke:#BDBDBD,stroke-dasharray: 5 5,color:#424242,rx:8px
```

The `GraphFormatValidator` is a shared schema that both repos use to validate `.pt` files. It ensures that every exported graph has the required `edge_index.pt` and `features.pt` for each component (g2m, m2m, m2g), with matching dimensions and valid ranges.

### 6.4 Layer 3: pyg.HeteroData Migration (Issue #385)

**The problem:** `load_graph()` currently returns a `dict` with 11 fragile string keys. No type safety, no schema validation, no named node/edge types.

```mermaid
flowchart TB
    %% Professional styling: Clean monochrome and subtle blues for readability
    classDef default fill:#ffffff,stroke:#94a3b8,stroke-width:2px,rx:6px,color:#1e293b
    classDef highlight fill:#f0f9ff,stroke:#0288d1,stroke-width:2px,rx:8px,color:#0f172a,font-weight:bold

    subgraph OLD ["Current: Dict of Tensors"]
        direction TB
        O1["graph_dict['g2m_edge_index']"]
        O2["graph_dict['m2m_edge_index'] — BufferList"]
        O3["graph_dict['m2g_edge_index']"]
        O4["graph_dict['mesh_static_features']"]
    end

    ARROW["⬇ MIGRATION ⬇"]:::highlight

    subgraph NEW ["Proposed: pyg.HeteroData"]
        direction TB
        N1["data['grid','g2m','mesh'].edge_index"]
        N2["data['mesh','m2m','mesh'].edge_index"]
        N3["data['mesh','m2g','grid'].edge_index"]
        N4["data['mesh'].x"]
    end

    %% Flow connections forcing the 3-row stacked layout
    OLD --> ARROW
    ARROW --> NEW

    %% Subgraph styling for a polished, contained look
    style OLD fill:#f8fafc,stroke:#cbd5e1,stroke-dasharray: 5 5,color:#334155,rx:8px
    style NEW fill:#f8fafc,stroke:#cbd5e1,stroke-dasharray: 5 5,color:#334155,rx:8px
```

**Benefits of HeteroData:**
- **Single `.to(device)` call** moves everything to GPU (instead of 11 individual transfers)
- **Schema validation** built into PyG — wrong shapes fail immediately
- **Typed access** — `data['grid', 'g2m', 'mesh']` is self-documenting
- **Extensible** — adding new node/edge types (e.g., `grid_station`) is trivial

**3-Step incremental migration (each step is non-breaking):**

| Step | What Changes | Backward Compatible? |
|------|-------------|---------------------|
| **A: Adapter** | `graph_dict_to_heterodata()` wraps existing dict output | ✅ Yes — dict still works |
| **B: Internal Refactor** | `BaseGraphModel` accepts HeteroData, feature flag `use_heterodata` | ✅ Yes — flag defaults False |
| **C: Native Loading** | `load_graph_hetero()` reads .pt directly to HeteroData | ✅ Yes — old path preserved |

### 6.5 Multi-Source Data Fusion (Layer 4)

Construct heterogeneous graphs from **multiple data sources** with different spatial densities. This is where `pyg.HeteroData` truly shines — different source types become different node types:

```python
# HeteroData naturally represents multi-source graphs:
data['grid_nwp', 'g2m', 'mesh'].edge_index       # dense NWP grid → mesh
data['grid_station', 'g2m', 'mesh'].edge_index    # sparse stations → mesh
data['mesh', 'm2g', 'grid_nwp'].edge_index        # mesh → NWP predictions
# ... [each source type is a separate node type with its own features]
```

<div style="page-break-after: always;"></div>

## 7. Advanced Research Contributions (Layers 4 & 5)

> **Strategic Note:** Layers 1–3 are core deliverables. The following mechanisms represent **modular stretch goals** that directly advance the state-of-the-art in graph-based neural weather prediction.

### 7.1 Graph Quality Metrics Framework (Layer 4)

Provides a `GraphQualityReport` to quantify mesh quality *before* expensive model training across four dimensions:
1. **Isotropy (edge-length uniformity):** Lower coefficient of variation (CV) ensures edges carry proportionate influence.
2. **Coverage (Voronoi ratio):** Measures how completely the mesh covers the data domain.
3. **Spectral Gap (Fiedler value):** Larger values mean faster mixing and better message-passing efficiency.
4. **G2M Balance:** Measures grid node distribution across mesh nodes to prevent orphaned nodes.

### 7.2 Density-Adaptive Mesh Generation (Layer 4)

Uniform meshes fail over irregular data. This algorithm builds adaptive spacings:

```text
DENSITY-ADAPTIVE MESH NODE PLACEMENT
--------------------------------------------------
Methods fail with uniform grids over oceans:
Data:       Mesh (uniform):
. . . .     O  O  O  O  <- Too many nodes over 
. . . .     O  O  O  O      ocean (sparse data)
. . . .     O  O  O  O
  .         O  O  O  O
      .     O  O  O  O  <- Too few nodes over
  .         O  O  O  O      land (dense data)

New method uses Voronoi cell areas to scale mesh:
Mesh (density-adaptive):
O O O O O O    <- Dense mesh where data is dense
O O O O O O 
    O          <- Sparse mesh where data is sparse
        O
```

**Algorithm:**
1. Compute Voronoi areas for local density mapping.
2. Scale local spacing functionally based on `base_mesh_distance`.
3. Apply variable-radius Poisson disk sampling & Delaunay triangulation.

### 7.3 Adaptive Mesh Refinement — AMR (Layer 4)

A machine-learning feedback loop where mesh structures adapt locally to minimize error:
1. Train with the initial mesh.
2. Map per-grid-point prediction errors.
3. Define spacing computationally via Gaussian KDE: `spacing(x) = base / (1 + factor * kde(x))`.
4. Render new mesh and retrain. 

### 7.4 Stretched-Grid & Topology Benchmarks (Layer 5)

* **Topology Benchmarking Suite (`wmg_benchmark.py`):** Ranks constructed domains dynamically against Information Propagation Diameter (IPD), Effective Receptive Field (ERF), and Edge Efficiency Ratios. 
* **Stretched Grids:** Support for regional resolutions tapering outward—matching the ECMWF AIFS blueprint natively within our `create_all_graph_components()` workflow.

### 7.5 State-of-the-Art Sub-Components (Layer 5)

These modules represent cutting-edge research targets to be explored upon successful integration of the core architectural roadmap. 

* **Spherical-Aware Graph Construction:** WMG currently leverages Euclidean distances, systematically distorting calculations at global scales (e.g., 2× distortion at lat=60°). A native `CoordinateSystem` abstraction (Haversine/Vincenty formulas) will correct Kd-tree generation and edge features.
* **Learned Mesh Coarsening for Hierarchical Graphs:** Replacing rigid N-stride node selection with **Weighted Farthest Point Sampling (wFPS)**. High-gradient geographic regions (mountains, coastlines) selectively retain resolution compared to flat terrain.
* **Weather-State-Aware Dynamic Graphs:** Implementing a `DynamicEdgeAttention` mechanism within Neural-LAM. WMG builds a superset of possible structural edges; Neural-LAM learns to dynamically thin/weight them via attention based on the *active* atmospheric state (e.g., routing density dynamically around a moving cyclone).
* **Graph Visualization & Analysis Dashboard:** Extending `plot_2d.py` into a fully analytic module (`GraphAnalysisPlot`) mapping Receptive Fields, Laplacian Eigenvalue Spectrum, and G2M connection densities.
* **Self-Describing `xr.DataTree` Output Format:** Replacing the loosely associated `.pt` dicts with a unified `graph.zarr/` tree, appending complete generation metadata and quality metrics for straightforward scientific reproduction.

<div style="page-break-after: always;"></div>

## 8. Development Timeline & Core Deliverables

This roadmap distills execution into 5 structured phases over 12 weeks, ensuring **Layers 1–3** (the "Core Deliverables") are solidly completed by Midterm, while safely sandboxing **Layers 4 & 5** (the modular "Alpha/Stretch Research Goals") for the latter half.

> **Note on Process:** All architecture will undergo rigorous Test-Driven Development (E2E Integration, Property Invariants, Backwards-Compatibility Regression).

### ☀️ Community Bonding (May 8 – June 1)
| Focus Area | Output Artifacts |
| :------- | :------- |
| **Design Prep** | • Merge/Address [PR #81](https://github.com/mllam/weather-model-graphs/pull/81) & [PR #258](https://github.com/mllam/neural-lam/pull/258).<br>• Prototype notebook for PyG `HeteroData` translation. |

### 🚀 Phase 1: Core Foundation (Weeks 1–3 / Layer 1 & 2)
| Focus Area | Output Artifacts |
| :------- | :------- |
| **WMG Stabilizers** | • **D1:** Convex hull auto-cropping for non-rectangular data ([#40](https://github.com/mllam/weather-model-graphs/issues/40)).<br>• **D2:** G2M isolation assertion/fixes ([#42](https://github.com/mllam/weather-model-graphs/issues/42)).<br>• **D3:** Enforce hierarchical edge integrity (`from_level`/`to_level`) ([#45](https://github.com/mllam/weather-model-graphs/issues/45)).<br>• **D4:** GraphFormatValidator schema integration ([#384](https://github.com/mllam/neural-lam/issues/384)). |

### 🌉 Phase 2: Building the Bridge (Weeks 4–6 / Layer 2 & 3)
| Focus Area | Output Artifacts |
| :------- | :------- |
| **The PyG Bridge** | • **D5:** Neural-LAM: Build native `build_graph.py` pipeline (retiring obsolete `create_graph.py`).<br>• **D6:** Neural-LAM: PyG `HeteroData` unified structure for non-rectangular data ingestion natively ([#385](https://github.com/mllam/neural-lam/issues/385)). |

> **<<< Midterm Evaluation Checkpoint >>>**<br>
> *Success Criterion: Irregular, non-rectangular topological data successfully routes End-to-End through WMG → Neural-LAM → forward model pass.*

### 🔬 Phase 3: Advanced Architectures (Weeks 7–9 / Layer 4 Stretch Work)
| Focus Area | Output Artifacts |
| :------- | :------- |
| **Custom Mesh Types**| • **D7:** Implementation of Delaunay Triangular (PR #92), Pre-built (PR #91), and Hexagonal grid routing.<br>• **D8:** Quality metrics calculation suite (Coverage, Isotropy, G2M Balance, Spectral Gap).<br>• **D9:** Density-adaptive mapping & Multi-source blending functionality. |

### 🔭 Phase 4: SOTA Implementation (Weeks 10–11 / Layer 5 Stretch Work)
| Focus Area | Output Artifacts |
| :------- | :------- |
| **Research SOTA** | • **D10:** WMG Spherical Haversine implementation for accurate global projections.<br>• **D11:** Neural-LAM: `DynamicEdgeAttention` mechanism for storm-system responsive graphing.<br>• **D12:** Topology analytical plotting toolkit (A/B testing ERF vs Coarsening efficiencies). |

### 📑 Phase 5: Final Polish (Week 12)
| Focus Area | Output Artifacts |
| :------- | :------- |
| **Handover** | • **D13:** Comprehensive Tutorial Jupyter Notebooks for E2E integration.<br>• **D14:** Full docstring verification and final branch rebasing.<br>• **D15:** Submission compilation. |

> **<<< Final Evaluation Checkpoint >>>**

---

## 11. Risk Mitigation

| Risk | Prob. | Impact | Mitigation |
|------|-------|--------|------------|
| PR #81 review delayed | Med | High | Start #40/#42/#45 in parallel — independent |
| HeteroData breaks models | Med | High | Feature flag `use_heterodata=True/False`; baseline comparison tests |
| Spectral computation slow | Low | Med | Power iteration for N>50k; cache eigenvalues |
| AMR doesn't converge | Med | Low | AMR is stretch/optional; core deliverables unaffected |
| Backward compat break | Low | High | `mesh_layout` required param (per leifdenby); full regression suite |
| Merge conflicts upstream | Med | Med | Weekly rebase; coordinate via Slack + weekly mentor sync |

---

## 12. References

### Key Issues & PRs

| Reference | Repo | Role |
|-----------|------|------|
| [**#384**](https://github.com/mllam/neural-lam/issues/384) — Tensor-on-disk | neural-lam | **Core: Layer 2** |
| [**#385**](https://github.com/mllam/neural-lam/issues/385) — pyg.HeteroData | neural-lam | **Core: Layer 3** |
| [**PR #81**](https://github.com/mllam/weather-model-graphs/pull/81) — mesh_layout | WMG | **My work: Layer 1** |
| [**PR #258**](https://github.com/mllam/neural-lam/pull/258) — Area weights | neural-lam | **My work** |
| [PR #91](https://github.com/mllam/weather-model-graphs/pull/91), [#92](https://github.com/mllam/weather-model-graphs/pull/92) | WMG | My work: Layer 1 |
| [#40](https://github.com/mllam/weather-model-graphs/issues/40), [#42](https://github.com/mllam/weather-model-graphs/issues/42), [#45](https://github.com/mllam/weather-model-graphs/issues/45) | WMG | Layer 1: v0.4.0 |
| [PR #47](https://github.com/mllam/weather-model-graphs/pull/47) — xr.DataTree | WMG | Layer 4 |

### Academic Papers

| Paper | Relevance |
|-------|-----------|
| Keisler (2022), "Forecasting Global Weather with GNNs" | Flat architecture → `create_keisler_graph()` |
| Lam et al. (2023), "GraphCast" | Multi-scale icosahedral mesh; encoder-processor-decoder |
| Oskarsson et al. (2023), "Graph-based Neural Weather for LAM" | Hierarchical graph → core neural-lam |
| G-Adaptivity (2024), GNN mesh movement for CFD | AMR research basis |
| M4GN (2025), Mesh-based multi-segment hierarchical GNN | Learned mesh coarsening basis |
| RTEC / TAEGCN (2024) | Dynamic edge construction research basis |
| Bridson (2007), "Fast Poisson disk sampling" | Node placement algorithm for density-adaptive |

---

## 13. Other Commitments

- [List exams, classes, holidays, jobs, internships here]
- Available 30–35 hours/week during the coding period
- Timezone: UTC+5:30 — overlap with European mentors ~10:00–18:00 CEST

| Channel | Frequency | Purpose |
|---------|-----------|---------|
| MLLAM Slack (#gsoc-project1) | Daily | Quick questions, async updates |
| Weekly written update | Monday | Summary + blockers + next steps |
| Video sync with mentors | Bi-weekly (30 min) | Design review, feedback |
| GitHub PRs | Per-week | Incremental code review |

---

## Summary

This proposal addresses [GSoC Idea #1: Flexible Graph Construction](https://github.com/mllam/neural-lam/wiki/GSoC-ideas#1-flexible-graph-construction) through a **five-layer architecture** with **30 deliverables**:

1. **Layer 1 (Foundation):** PRs #81, #91, #92, #258 + v0.4.0 blockers #40, #42, #45
2. **Layer 2 (Bridge — #384):** `build_graph.py` + `GraphFormatValidator` eliminates 600-line duplication
3. **Layer 3 (Architecture — #385):** Incremental `pyg.HeteroData` migration: adapter → refactor → native
4. **Layer 4 (Advanced Research):** Quality Metrics · Density-Adaptive Mesh · AMR · xr.DataTree · Multi-Source Fusion
5. **Layer 5 (Cutting-Edge):** Spherical CoordinateSystem · Topology Benchmark Suite · Stretched-Grid · Learned Coarsening · Dynamic Edge Attention · Analysis Dashboard

**What makes this proposal extraordinary:**
- **No one else has proposed** a topology benchmarking suite that ranks meshes WITHOUT training
- **No one else has proposed** spherical-aware graph construction that fixes systematic polar distortion
- **No one else has proposed** dynamic weather-state-aware edge selection backed by RTEC/TAEGCN (2024)
- **No one else has proposed** learned spectral coarsening preserving the graph's Fiedler value

Every deliverable maps to an open issue, mentor priority, or novel research contribution. Full backward compatibility while enabling graph construction from **any spatial data distribution** — from regular rectangular grids to scattered ship observations.

---

*Proposal prepared by: Prajwal [Your Last Name]*
*Last updated: March 15, 2026*
*GitHub: [github.com/prajwal-tech07](https://github.com/prajwal-tech07)*
