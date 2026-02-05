# Hyperpathway: Network Visualization Tool


## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Windows](#windows)
  - [macOS](#macos)
  - [Linux](#linux)
- [Usage](#usage)
  - [Web Application (Streamlit)](#web-application-streamlit)
  - [Command Line Interface](#command-line-interface)
- [Input File Formats](#input-file-formats)
  - [Pathway Enrichment Analysis Table](#pathway-enrichment-analysis-table)
  - [Bipartite Network Format](#bipartite-network-format)
- [Visualization Modes](#visualization-modes)
  - [Pathway Significance Mode](#pathway-significance-mode)
  - [Gradient Coloring Modes](#gradient-coloring-modes)
- [Command Line Arguments](#command-line-arguments)
- [Output Files](#output-files)
- [Algorithm Overview](#algorithm-overview)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## Features

- 🧬 **Multi-Omics Compatibility**: Support for **lipidomics, genomics, metabolomics**, and other omics data
- 🔬 **Pathway Enrichment Visualization**: Convert PEA tables into bipartite networks
- 🌐 **Hyperbolic Embedding**: Uses coalescent embedding with RA1 weighting and ISOMAP dimension reduction
- 📊 **Statistical Support**: Handles multiple p-value correction methods (Bonferroni, Benjamini-Hochberg, etc.)
- 🎨 **Multiple Coloring Schemes**: 
  - Pathway significance (color-coded by statistical significance)
  - Hierarchy-based (color gradient by node degree)
  - Similarity-based (color gradient by angular position)
  - Label-based (custom categorical coloring)
- 🎯 **Subnetwork Extraction**: Focus on specific pathways or molecules of interest
- 📈 **High-Quality Output**: PNG plots with customizable resolution and Excel data export of coordinates and edge lists
- 🔧 **Flexible Input**: Supports CSV, XLS, XLSX and TSV formats with column name or index specification
- 🖥️ **Multiple Interfaces**: Both **web UI** [https://hyperpathways.org/](https://hyperpathways.org/) and **command-line** interfaces 

---

## Installation

### Prerequisites

- **Python 3.11 or higher**
- **pip** (Python package manager)
- **Git** (for cloning the repository)

### Windows

1. **Install Python**
   - Download from [python.org](https://www.python.org/downloads/)
   - ✅ **Important:** Check "Add Python to PATH" during installation
   - Verify installation:
     ```cmd
     python --version
     pip --version
     ```

2. **Clone the repository**
   ```cmd
   git clone https://github.com/biomedical-cybernetics/Hyperpathway.git
   cd hyperpathway
   ```

3. **Create virtual environment** (recommended)
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

4. **Install dependencies**
   ```cmd
   pip install -r requirements.txt
   ```

### macOS

1. **Install Python** (if not already installed)
   ```bash
   # Using Homebrew (recommended)
   brew install python@3.11
   
   # Verify installation
   python3 --version
   pip3 --version
   ```

2. **Clone the repository**
   ```bash
   git clone https://github.com/biomedical-cybernetics/Hyperpathway.git
   cd hyperpathway
   ```

3. **Create virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Linux

1. **Install Python** (Ubuntu/Debian example)
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv git
   
   # Verify installation
   python3 --version
   pip3 --version
   ```

   For **Fedora/RHEL**:
   ```bash
   sudo dnf install python3 python3-pip git
   ```

2. **Clone the repository**
   ```bash
   git clone https://github.com/biomedical-cybernetics/Hyperpathway.git
   cd hyperpathway
   ```

3. **Create virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Web Application (Streamlit)

The web interface provides an intuitive point-and-click environment for:
- File upload with drag-and-drop support
- Interactive column mapping
- Real-time parameter adjustment
- Visualization preview
- One-click download of results

Visit [https://hyperpathways.org/](https://hyperpathways.org/) for the online version, or run locally:

**Activate virtual environment:**
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

**Launch the app:**
```bash
streamlit run hyperpathway_launcher_web_app.py
```
  
### Command-Line Interface

The CLI provides advanced control and scriptability for batch processing and integration into analysis pipelines.

#### Mode 1: Pathway Enrichment Analysis (PEA)

Process pathway enrichment tables with statistical significance coloring:

```bash
python run_hyperpathway_main_command_tool.py \
  --mode pea \
  -i enrichment_results.csv \
  --pathway-col "Pathway" \
  --molecules-col "Molecules" \
  --pval-col "P-value" \
  --corr1-col "BH_adjusted" \
  --corr2-col "Bonferroni" \
  -o hyperpathway_output.png
```

#### Mode 2: Bipartite Network (Pre-computed adjacency list)

Visualize pre-constructed bipartite networks:

```bash
python run_hyperpathway_main_command_tool.py \
  --mode bipartite \
  --adjacency-file edges.csv \
  --node-file nodes.csv \
  --coloring hierarchy \
  -o network_visualization.png
```

#### Using Column Indices

If your file lacks headers or you prefer numeric indices (0-based):

```bash
python run_hyperpathway_main_command_tool.py \
  --mode pea \
  -i data.xlsx \
  --pathway-col 0 \
  --molecules-col 1 \
  --pval-col 2 \
  --corr1-col 3
```

#### Custom Significance Thresholds

Adjust p-value thresholds for different stringency levels:

```bash
python run_hyperpathway_main_command_tool.py \
  --mode pea \
  -i data.csv \
  --pathway-col "Pathway" \
  --molecules-col "Genes" \
  --pval-col "P-value" \
  --pval-threshold 0.01 \
  --corr1-col "BH_adjusted" \
  --corr1-name "Benjamini-Hochberg FDR" \
  --corr2-col "Bonferroni" \
  --corr2-name "Bonferroni Correction"
```

---

## Input File Formats

### Pathway Enrichment Analysis Table

Your input file should contain at minimum:
1. **Pathway names** (pathway identifiers)
2. **Enriched molecules** (semicolon, comma, or pipe-separated)

Optionally include p-value columns for significance visualization:

| Pathway Name | Enriched Molecules | Raw P-value | P-value corrected #1<br>(e.g. Benjamini-Hochberg) | P-value corrected #2<br>(e.g. Bonferroni) |
|--------------|-------------------|---------|-------------|------------|
| Glycolysis | GLC;PYR;ATP | 0.001 | 0.01 | 0.05 |
| TCA Cycle | CIT;AKG;SUC | 0.005 | 0.02 | 0.10 |

**Supported formats:** `.csv`, `.xls`, `.xlsx`, `.tsv`

**Molecule separators:** semicolon (`;`), comma (`,`), or pipe (`|`)

### Bipartite Network Format

Alternatively, you can provide a pre-computed adjacency list:

| Node1 | Node2 | Color (optional) |
|-------|-------|------------------|
| Pathway1 | Molecule1 | #FF0000 |
| Pathway1 | Molecule2 | #FF0000 |
| Pathway2 | Molecule3 | #FFA500 |

---

## Visualization Modes

### Pathway Significance Mode

When using p-value columns, pathways are automatically colored by statistical significance:

**Legend:**
- 🔴 **Red**: Pathway is statistically significant in **all** provided p-value tests
  - When only ONE p-value column is provided, red indicates significance in that single test
  - When MULTIPLE p-value columns are provided, red indicates significance in ALL tests
- 🟠 **Orange**: Pathway is statistically significant in **some but not all** p-value tests
  - Only appears when multiple p-value columns are provided
  - Indicates partial significance across different correction methods
- ⚪ **Gray**: Pathway is statistically significant only in the **uncorrected** p-value test
  - Fails significance after multiple testing correction
- 🔘 **No display**: Pathway is not statistically significant in any test (filtered out)

**Examples:**

*Single p-value column (only FDR):*
```bash
# Red = significant in FDR test
python run_hyperpathway_main_command_tool.py --mode pea -i data.csv \
  --pathway-col "Pathway" --molecules-col "Molecules" \
  --corr1-col "FDR" --pval-threshold 0.05
```

*Two corrected p-value columns:*
```bash
# Red = significant in both FDR AND Bonferroni
# Orange = significant in FDR OR Bonferroni (but not both)
python run_hyperpathway_main_command_tool.py --mode pea -i data.csv \
  --pathway-col "Pathway" --molecules-col "Molecules" \
  --corr1-col "FDR" --corr2-col "Bonferroni" --pval-threshold 0.05
```

*All three p-value types:*
```bash
# Red = significant in uncorrected AND FDR AND Bonferroni
# Orange = significant in uncorrected + one correction method
# Gray = significant only in uncorrected
python run_hyperpathway_main_command_tool.py --mode pea -i data.csv \
  --pathway-col "Pathway" --molecules-col "Molecules" \
  --pval-col "Raw_P" --corr1-col "FDR" --corr2-col "Bonferroni"
```

### Gradient Coloring Modes

Alternative visualization schemes for exploring network topology:

#### 1. Hierarchy Mode (`--coloring hierarchy`)
Colors nodes by their degree (number of connections):
- Dark blue = low degree (peripheral nodes)
- Yellow = high degree (hub nodes)

**Use case:** Identify central pathways and key molecules

```bash
python run_hyperpathway_main_command_tool.py --mode pea -i data.csv \
  --pathway-col 0 --molecules-col 1 --coloring hierarchy
```

#### 2. Similarity Mode (`--coloring similarity`)
Colors nodes by their angular position in hyperbolic space:
- Similar colors = topologically similar nodes
- Different colors = topologically distant nodes

**Use case:** Discover functional modules and pathway clusters

```bash
python run_hyperpathway_main_command_tool.py --mode pea -i data.csv \
  --pathway-col 0 --molecules-col 1 --coloring similarity
```

#### 3. Labels Mode (`--coloring labels`)
Colors nodes by custom categorical labels (e.g., biological process, disease association):

**Use case:** Overlay external annotations onto the network

```bash
python run_hyperpathway_main_command_tool.py --mode pea -i data.csv \
  --pathway-col "Pathway" --molecules-col "Molecules" \
  --coloring labels --labels-col "Biological_Process"
```

---

## Command-Line Arguments

### Core Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--mode` | choice | Yes | Analysis mode: `pea` or `bipartite` |
| `-i, --input` | path | Yes (PEA) | Input file path (CSV, XLS, XLSX, TSV) |
| `--pathway-col` | str/int | Yes (PEA) | Pathway column name or 0-based index |
| `--molecules-col` | str/int | Yes (PEA) | Molecules column name or 0-based index |

### P-value Columns (PEA Mode)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--pval-col` | str/int | None | Non-corrected p-value column |
| `--corr1-col` | str/int | None | First corrected p-value column |
| `--corr2-col` | str/int | None | Second corrected p-value column |
| `--pval-threshold` | float | 0.05 | Significance threshold for all p-values |
| `--corr1-name` | str | "Correction #1" | Display name for first correction method |
| `--corr2-name` | str | "Correction #2" | Display name for second correction method |

### Bipartite Mode

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--adjacency-file` | path | Yes (bipartite) | Adjacency list file |
| `--node-file` | path | Yes (bipartite) | Node list file |

### Visualization Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--coloring` | choice | similarity | Coloring scheme: `default`, `hierarchy`, `similarity`, or `labels` |
| `--labels-col` | str/int | None | Column for labels mode |
| `--show-labels` | flag | False | Display node labels on plot |
| `--hide-edges` | flag | False | Hide edges (show only nodes) |
| `--max-edges` | int | 20000 | Maximum edges to render |
| `--edge-opacity` | float | 0.3 | Edge transparency (0-1) |
| `--node-size-scale` | float | 1.0 | Node size multiplier |

### Output Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-o, --output` | path | hyperpathway_plot.png | Output plot filename |
| `--dpi` | int | 300 | Plot resolution (dots per inch) |
| `--figsize` | float float | 12 12 | Figure size in inches (width height) |
| `--excel-output` | path | network_data.xlsx | Excel file for coordinates and edges |

### Subnetwork Extraction

| Argument | Type | Description |
|----------|------|-------------|
| `--subnetwork-nodes` | str | Comma-separated list of seed nodes |
| `--subnetwork-file` | path | File containing seed nodes (one per line) |
| `--subnetwork-output` | path | Subnetwork plot filename |
| `--subnetwork-coloring` | choice | Coloring for subnetwork |
| `--subnetwork-show-labels` | flag | Show labels in subnetwork |
| `--subnetwork-hide-edges` | flag | Hide edges in subnetwork |

---

## Output Files

### Visualization Plot

High-resolution publication-ready figure featuring:

**Node types:**
- 🔶 **Diamonds**: Pathways
- 🔵 **Circles**: Molecules

**In Pathway Significance Mode:**
- Color indicates statistical significance level
- Size correlates with node degree (connectivity)

**In Gradient Coloring Modes:**
- Color gradient reflects hierarchy, similarity, or custom labels

### Excel Data File

Contains two sheets:
1. **Node coordinates**: x, y coordinates and node labels for all nodes
2. **Edges**: List of pathway-molecule interactions

## Algorithm Overview

Hyperpathway uses the following steps:

1. **Network Construction**: Build bipartite network from PEA table
2. **Significance Filtering**: Remove non-significant pathways based on thresholds
3. **Coalescent Embedding**
    1. **RA1 Weighting**: Apply Resource Allocation index weighting to edges
    2. **Non-Centered ISOMAP**: Perform dimensionality reduction for angular coordinates
    3. **Radial Positioning**: Set radial coordinates based on degree distribution
4. **Hyperbolic Visualization**: Hyperbolic disk representation

## Examples

### Example 1: Basic Pathway Analysis

```bash
python run_hyperpathway_main_command_tool.py \
  -i metabolomics_results.csv \
  --pathway-col "Pathway" \
  --molecules-col "Metabolites" \
  --pval-col "P.value" \
  --corr1-col "FDR" \
  -o metabolomics_hyperpathway.png
```

### Example 2: High-Resolution Output

```bash
python run_hyperpathway_main_command_tool.py \
  -i proteomics_data.xlsx \
  --pathway-col 0 \
  --molecules-col 1 \
  --pval-col 2 \
  --dpi 600 \
  --figsize 16 16 \
  -o proteomics_highres.png
```

### Example 3: With Labels and Limited Edges

```bash
python run_hyperpathway_main_command_tool.py \
  -i enrichment_results.csv \
  --pathway-col "pathway_name" \
  --molecules-col "genes" \
  --show-labels \
  --max-edges 5000 \
  -o labeled_network.png
```

## Troubleshooting

### Common Issues

**Issue**: "Column 'X' not found in dataframe"
- **Solution**: Check your column names match exactly (case-sensitive), or use column indices instead

**Issue**: Visualization is too cluttered
- **Solution**: Use `--hide-edges` to show only nodes, or increase `--figsize` for a larger canvas

## Citation

If you use Hyperpathway in your research, please cite:

```
[Your citation information here]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Dr. Ilyes Abdelhamid **(I.A. - first author)** - [ilyes.abdelhamid1@gmail.com]  
Prof. Carlo Vittorio Cannistraci **(C.V.C. - corresponding author)** - [kalokagathos.agon@gmail.com ]  
Project Link: [https://github.com/biomedical-cybernetics/Hyperpathway](https://github.com/biomedical-cybernetics/Hyperpathway)    
Hyperpathway Web Application Link - [https://hyperpathways.org/](https://hyperpathways.org/)    
Center of Complex Network Intelligence **(the lab)** - [https://brain.tsinghua.edu.cn/en/Research1/Research_Centers/Complex_Network_Intelligence_Center.htm](https://brain.tsinghua.edu.cn/en/Research1/Research_Centers/Complex_Network_Intelligence_Center.htm)

## Acknowledgments

This work was supported by the Zhou Yahui Chair Professorship award of Tsinghua University (to C.V.C.), the National High-Level Talent Program of the Ministry of Science and Technology of China (grant number 20241710001, to C.V.C.), and the Shuimu Tsinghua Scholar Program (to I.A.).

---
