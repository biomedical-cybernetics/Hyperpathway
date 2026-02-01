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
- 🎨 **Color-Coded Significance**: Visual differentiation of pathways by statistical significance
- 📈 **High-Quality Output**: PNG plots with customizable resolution and Excel data export of coordinates and edge lists
- 🔧 **Flexible Input**: Supports CSV, XLS, and XLSX formats with column name or index specification
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
   git clone https://github.com/IlyesAbdelhamid/hyperpathway.git
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
   git clone https://github.com/IlyesAbdelhamid/hyperpathway.git
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
   git clone https://github.com/IlyesAbdelhamid/hyperpathway.git
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

The easiest way to use Hyperpathway is through the interactive web interface:

**Windows:**
```cmd
venv\Scripts\activate
streamlit run hyperpathway_launcher_web_app_test_v35.py
```

**macOS/Linux:**
```bash
source venv/bin/activate
streamlit run hyperpathway_launcher_web_app_test_v35.py
```

The app will open automatically in your browser at `http://localhost:8501`  
Ctrl+C on the keyboard in your terminal to exit the session.  

  

### Command Line Interface

### Basic Usage

Run Hyperpathway with a pathway enrichment analysis file:

```bash
python run_hyperpathway_main_command_tool.py \
  -i your_data.csv \
  --pathway-col "Pathway" \
  --molecules-col "Molecules" \
  --pval-col "P-value" \
  --corr1-col "BH_adjusted" \
  --corr2-col "Bonferroni"
```

### Using Column Indices

If your file doesn't have headers or you prefer using indices (0-based):

```bash
python run_hyperpathway_main_command_tool.py \
  -i your_data.xlsx \
  --pathway-col 0 \
  --molecules-col 1 \
  --pval-col 2 \
  --corr1-col 3 \
  --corr2-col 4
```

### Custom Thresholds

Specify significance thresholds for different correction methods:

```bash
python run_hyperpathway_main_command_tool.py \
  -i data.csv \
  --pathway-col "Pathway" \
  --molecules-col "Molecules" \
  --pval-col "P-value" \
  --pval-threshold 0.01 \
  --corr1-col "BH_adjusted" \
  --corr1-threshold 0.05 \
  --corr2-col "Bonferroni" \
  --corr2-threshold 0.001 \
  --corr1-name "Benjamini-Hochberg" \
  --corr2-name "Bonferroni"
```


## Input File Formats

### Pathway Enrichment Analysis Table

Your input file should contain at least two columns:

| Pathway Name | Enriched Molecules | Raw P-value | P-value corrected #1<br>(e.g. Benjamini-Hochberg) | P-value corrected #2<br>(e.g. Bonferroni) |
|--------------|-------------------|---------|-------------|------------|
| Glycolysis | GLC;PYR;ATP | 0.001 | 0.01 | 0.05 |
| TCA Cycle | CIT;AKG;SUC | 0.005 | 0.02 | 0.10 |

- **Pathway Name**: Name or identifier of the pathway
- **Enriched Molecules**: Semicolon, comma, or pipe-separated list of molecules
- **P-value columns (optional)**: Columns for statistical significance (uncorrected, correction method 1, correction method 2)

Supported formats: `.csv`, `.xls`, `.xlsx`

### Bipartite Network Format

Alternatively, you can provide a pre-computed adjacency list:

| Node1 | Node2 | Color (optional) |
|-------|-------|------------------|
| Pathway1 | Molecule1 | #FF0000 |
| Pathway1 | Molecule2 | #FF0000 |
| Pathway2 | Molecule3 | #FFA500 |


## Command Line Arguments

### Required Arguments

- `-i, --input`: Input file path (CSV, XLS, or XLSX)
- `--pathway-col`: Pathway column name or index (0-based)
- `--molecules-col`: Molecules column name or index (0-based)

### Optional P-value Columns

- `--pval-col`: Non-corrected p-value column
- `--corr1-col`: First corrected p-value column
- `--corr2-col`: Second corrected p-value column

### Threshold Arguments

- `--pval-threshold`: Significance threshold for non-corrected p-values (default: 0.05)
- `--corr1-threshold`: Significance threshold for first correction (default: 0.05)
- `--corr2-threshold`: Significance threshold for second correction (default: 0.05)

### Correction Method Names

- `--corr1-name`: Display name for first correction method (default: "Correction #1")
- `--corr2-name`: Display name for second correction method (default: "Correction #2")

### Output Arguments

- `-o, --output`: Output plot filename (default: `hyperpathway_plot.png`)
- `--dpi`: Output plot resolution (default: 300)
- `--figsize`: Figure size in inches, width and height (default: 12 12)
- `--excel-output`: Output Excel file for coordinates and edges (default: `network_data.xlsx`)

### Visualization Options

- `--show-labels`: Show node labels on the visualization
- `--hide-edges`: Hide edges on the visualization
- `--max-edges`: Maximum number of edges to render (default: 20000)

## Output Files

### Visualization Plot

A high-resolution PNG file showing:
- **Pathways** (diamonds) colored by significance level:
  - 🔴 Red: Significant after first correction
  - 🟠 Orange: Significant after second correction
  - ⚪ Gray: Significant without correction
  - 🔵 Blue: Non-significant
- **Molecules** (circles) in green
- Hyperbolic edges connecting related pathways and molecules

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
Project Link: [https://github.com/IlyesAbdelhamid/Hyperpathway](https://github.com/IlyesAbdelhamid/Hyperpathway)    
Hyperpathway Web Application Link - [https://hyperpathways.org/](https://hyperpathways.org/)    
Center of Complex Network Intelligence **(the lab)** - [https://brain.tsinghua.edu.cn/en/Research1/Research_Centers/Complex_Network_Intelligence_Center.htm](https://brain.tsinghua.edu.cn/en/Research1/Research_Centers/Complex_Network_Intelligence_Center.htm)

## Acknowledgments

This work was supported by the Zhou Yahui Chair Professorship award of Tsinghua University (to C.V.C.), the National High-Level Talent Program of the Ministry of Science and Technology of China (grant number 20241710001, to C.V.C.), and the Shuimu Tsinghua Scholar Program (to I.A.).

---
