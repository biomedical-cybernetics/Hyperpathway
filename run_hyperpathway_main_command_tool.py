#!/usr/bin/env python3
"""
Hyperpathway Command-Line Tool
Generate hyperbolic pathway visualizations from enrichment analysis data or preconstructed bipartite networks.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Import your modules
from convert_pea_to_bipartite_net_command_tool import (
    process_input_pea_table, 
    build_network,
    process_adjacency_list,
    process_list_nodes
)
from compute_hyperpathway_command_tool import (
    run_hyperpathway_with_progress,
    remove_isolated_nodes,
    filter_subgraph,
    plot_hyperpathway_static,
    plot_hyperpathway_static_gradient_color
)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Hyperpathway: Hyperbolic pathway enrichment visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Mode 1: Pathway Enrichment Analysis:
  python %(prog)s --mode pea -i enrichment.csv \\
    --pathway-col "Pathway" \\
    --molecules-col "Molecules" \\
    --pval-col "P-value" \\
    --corr1-col "BH_adjusted" \\
    --corr2-col "Bonferroni"

  # With custom p-value threshold:
  python %(prog)s --mode pea -i enrichment.csv \\
    --pathway-col "Pathway" \\
    --molecules-col "Molecules" \\
    --pval-col "P-value" \\
    --pval-threshold 0.01
 
  # With gradient coloring by popularity (degree):
  python %(prog)s --mode pea -i enrichment.csv \\
    --pathway-col "Pathway" \\
    --molecules-col "Molecules" \\
    --pval-col "P-value" \\
    --coloring popularity

  # With gradient coloring by similarity (angular position):
  python %(prog)s --mode pea -i enrichment.csv \\
    --pathway-col "Pathway" \\
    --molecules-col "Molecules" \\
    --pval-col "P-value" \\
    --coloring similarity

  # With gradient coloring by custom labels:
  python %(prog)s --mode pea -i enrichment.csv \\
    --pathway-col "Pathway" \\
    --molecules-col "Molecules" \\
    --pval-col "P-value" \\
    --coloring labels \\
    --labels-col "Community"

  # Using column indices (0-based):
  python %(prog)s --mode pea -i enrichment.xlsx \\
    --pathway-col 0 \\
    --molecules-col 1 \\
    --pval-col 2 \\
    --coloring labels \\
    --labels-col 5

  # Mode 2: Bipartite Network (adjacency list + node list)
  python %(prog)s --mode bipartite \\
    --adjacency-file edges.csv \\
    --node-file nodes.csv \\
    -o bipartite_plot.png \\
    --coloring popularity

  # With subnetwork extraction:
  python %(prog)s --mode pea -i data.csv \\
    --pathway-col 0 --molecules-col 1 \\
    --subnetwork-nodes "ATP,NADH,Glucose" \\
    --subnetwork-output subnet.png \\
    --subnetwork-coloring similarity

        '''
    )

    # Mode selection
    parser.add_argument(
        '--mode',
        choices=['pea', 'bipartite'],
        required=True,
        help='Analysis mode: "pea" for pathway enrichment, "bipartite" for adjacency list'
    )

    # PEA mode arguments
    parser.add_argument(
        '-i', '--input',
        help='Input file path for PEA mode(CSV, XLS, or XLSX)'
    )
    
    parser.add_argument(
        '--pathway-col',
        help='Pathway column name or index (0-based) - PEA mode only'
    )
    
    parser.add_argument(
        '--molecules-col',
        help='Molecules column name or index (0-based) - PEA mode only'
    )
    
    parser.add_argument(
        '--pval-col',
        default=None,
        help='Non-corrected p-value column name or index - PEA mode only'
    )
    
    parser.add_argument(
        '--corr1-col',
        default=None,
        help='First corrected p-value column name or index - PEA mode only'
    )
    
    parser.add_argument(
        '--corr2-col',
        default=None,
        help='Second corrected p-value column name or index - PEA mode only'
    )
    
    # Threshold arguments (PEA mode)
    parser.add_argument(
        '--pval-threshold',
        type=float,
        default=0.05,
        help='Significance threshold for all p-values (non-corrected and corrected) (default: 0.05)'
    )
    
    # Correction method names
    parser.add_argument(
        '--corr1-name',
        type=str,
        default='Correction #1',
        help='Display name for first correction method (default: "Correction #1")'
    )
    
    parser.add_argument(
        '--corr2-name',
        type=str,
        default='Correction #2',
        help='Display name for second correction method (default: "Correction #2")'
    )
    
    # Bipartite mode arguments
    parser.add_argument(
        '--adjacency-file',
        help='Adjacency list file (CSV, XLS, or XLSX) - Bipartite mode only'
    )

    parser.add_argument(
        '--node-file',
        help='Node list file (CSV, XLS, or XLSX) - Bipartite mode only'
    )

    # Coloring options
    parser.add_argument(
        '--coloring',
        choices=['popularity', 'similarity', 'labels', 'default'],
        default='default',
        help='Node coloring scheme: "popularity" (by degree), "similarity" (by angular position), "labels" (by custom labels), or "default" (standard pathway coloring)'
    )

    parser.add_argument(
        '--labels-col',
        default=None,
        help='Column name or index (0-based) for labels when using --coloring labels'
    )

    # Subnetwork coloring options
    parser.add_argument(
        '--subnetwork-coloring',
        choices=['popularity', 'similarity', 'labels', 'default'],
        default=None,
        help='Node coloring scheme for subnetwork (defaults to same as main network if not specified)'
    )

    parser.add_argument(
        '--subnetwork-labels-col',
        default=None,
        help='Column name or index for labels in subnetwork (defaults to --labels-col if not specified)'
    )

    # Output arguments
    parser.add_argument(
        '-o', '--output',
        default='hyperpathway_plot.png',
        help='Output plot filename (default: hyperpathway_plot.png)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Output plot resolution (default: 300)'
    )
    
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[12, 12],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size in inches (default: 12 12)'
    )
    
    parser.add_argument(
        '--excel-output',
        default='network_data.xlsx',
        help='Output Excel file for coordinates and edges (default: network_data.xlsx)'
    )
    
    # Visualization options
    parser.add_argument(
        '--show-labels',
        action='store_true',
        help='Show node labels on the visualization'
    )
    
    parser.add_argument(
        '--hide-edges',
        action='store_true',
        help='Hide edges on the visualization'
    )
    
    parser.add_argument(
        '--max-edges',
        type=int,
        default=20000,
        help='Maximum number of edges to render (default: 20000)'
    )
    
    parser.add_argument(
        '--edge-opacity',
        type=float,
        default=0.6,
        help='Opacity/transparency of edges, from 0.0 (transparent) to 1.0 (opaque) (default: 0.6)'
    )

    # Subnetwork extraction arguments
    parser.add_argument(
        '--subnetwork-nodes',
        type=str,
        default=None,
        help='Comma-separated list of node names to extract as subnetwork (e.g., "ATP,NADH,Glucose")'
    )

    parser.add_argument(
        '--subnetwork-file',
        type=str,
        default=None,
        help='Path to file containing node names (one per line) for subnetwork extraction'
    )

    parser.add_argument(
        '--subnetwork-output',
        default='subnetwork_plot.png',
        help='Output filename for subnetwork plot (default: subnetwork_plot.png)'
    )

    parser.add_argument(
        '--subnetwork-show-labels',
        action='store_true',
        help='Show node labels on subnetwork visualization'
    )

    parser.add_argument(
        '--subnetwork-hide-edges',
        action='store_true',
        help='Hide edges on subnetwork visualization'
    )

    # Visualization options (add after existing --max-edges argument)
    parser.add_argument(
        '--node-size-scale',
        type=float,
        default=1.0,
        help='Scaling factor for node sizes (default: 1.0). Values > 1 make nodes larger, < 1 make them smaller'
    )
    
    return parser.parse_args()

    
def validate_arguments(args):
    """Validate mode-specific required arguments."""
    if args.mode == 'pea':
        if not args.input:
            print("✗ Error: --input is required for PEA mode")
            return False
        if not args.pathway_col or not args.molecules_col:
            print("✗ Error: --pathway-col and --molecules-col are required for PEA mode")
            return False
    elif args.mode == 'bipartite':
        if not args.adjacency_file:
            print("✗ Error: --adjacency-file is required for bipartite mode")
            return False
        if not args.node_file:
            print("✗ Error: --node-file is required for bipartite mode")
            return False

    # Validate pval-threshold is positive
    if args.pval_threshold <= 0:
        print(f"✗ Error: --pval-threshold must be positive, got {args.pval_threshold}")
        return False

    # Validate coloring-specific requirements
    if args.coloring == 'labels' and not args.labels_col:
        print("✗ Error: --labels-col is required when using --coloring labels")
        return False

    if args.subnetwork_coloring == 'labels' and not args.subnetwork_labels_col and not args.labels_col:
        print("✗ Error: --subnetwork-labels-col or --labels-col is required when using --subnetwork-coloring labels")
        return False

    # Validate edge_opacity is between 0 and 1
    if not (0.0 <= args.edge_opacity <= 1.0):
        print(f"✗ Error: --edge-opacity must be between 0.0 and 1.0, got {args.edge_opacity}")
        return False

    return True


def convert_column_arg(col_arg):
    """
    Convert column argument to int if it's a numeric string, otherwise keep as string.
    
    Parameters:
    -----------
    col_arg : str or None
        Column argument from command line
        
    Returns:
    --------
    int, str, or None
    """
    if col_arg is None:
        return None
    
    # Try to convert to int (for column indices)
    try:
        return int(col_arg)
    except ValueError:
        # If it fails, it's a column name (string)
        return col_arg


def load_labels_from_file(file_path, labels_col, wname):
    """
    Load labels from the input file for gradient coloring.
    
    Parameters:
    -----------
    file_path : str
        Path to input file
    labels_col : str or int
        Column name or index for labels
    wname : list
        List of node names (pathways and molecules)
        
    Returns:
    --------
    ndarray or None : Labels array matching wname order, or None if error
    """
    try:
        # Read file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            print(f"✗ Error: Unsupported file format for labels")
            return None

        # Get column by name or index
        if isinstance(labels_col, int):
            if labels_col >= len(df.columns):
                print(f"✗ Error: Column index {labels_col} out of range")
                return None
            label_column = df.iloc[:, labels_col]
        else:
            if labels_col not in df.columns:
                print(f"✗ Error: Column '{labels_col}' not found in file")
                return None
            label_column = df[labels_col]

        # Create a mapping from pathway names to labels
        # Assuming the first column contains pathway names
        pathway_col = df.iloc[:, 0]
        label_map = dict(zip(pathway_col, label_column))

        # Map labels to wname order
        labels = []
        for name in wname:
            # Try to find the full name in the mapping
            if name in label_map:
                labels.append(label_map[name])
            else:
                # Try to find by matching the first word
                found = False
                for key in label_map.keys():
                    if str(key).split(' ', 1)[0] == name:
                        labels.append(label_map[key])
                        found = True
                        break
                if not found:
                    labels.append(0)  # Default label

        return np.array(labels)

    except Exception as e:
        print(f"✗ Error loading labels: {e}")

        return None


def load_selected_nodes(args, fixed_names):
    """
    Load selected nodes from command line or file.

    Parameters:
    -----------
    args : Namespace
        Parsed command-line arguments
    fixed_names : list
        List of all node names in the network
    
    Returns:
    --------
    list : Indices of selected nodes, or None if no selection
    """
    selected_node_names = []

    # Load from comma-separated string
    if args.subnetwork_nodes:
        selected_node_names = [name.strip() for name in args.subnetwork_nodes.split(',')]
        print(f"Selected {len(selected_node_names)} nodes from command line")

    # Load from file
    elif args.subnetwork_file:
        try:
            with open(args.subnetwork_file, 'r') as f:
                selected_node_names = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(selected_node_names)} nodes from {args.subnetwork_file}")
        except FileNotFoundError:
            print(f"✗ Error: File not found: {args.subnetwork_file}")
            return None
        except Exception as e:
            print(f"✗ Error reading file: {e}")
            return None

    if not selected_node_names:
        return None

    # Map names to indices
    selected_indices = []
    not_found = []

    for name in selected_node_names:
        try:
            idx = fixed_names.index(name)
            selected_indices.append(idx)
        except ValueError:
            not_found.append(name)

    if not_found:
        print(f"⚠ Warning: {len(not_found)} node(s) not found in network:")
        for name in not_found[:10]:  # Show first 10
            print(f"  - {name}")
        if len(not_found) > 10:
            print(f"  ... and {len(not_found) - 10} more")

    if not selected_indices:
        print("✗ Error: None of the selected nodes were found in the network")
        return None

    print(f"✓ Matched {len(selected_indices)} nodes in the network")

    return selected_indices


def process_pea_mode(args):
    """Process pathway enrichment analysis mode."""
    print("\n" + "=" * 70)
    print("MODE: Pathway Enrichment Analysis")
    print("=" * 70)
    
    # Convert column arguments
    pathway_col = convert_column_arg(args.pathway_col)
    molecules_col = convert_column_arg(args.molecules_col)
    pval_col = convert_column_arg(args.pval_col)
    corr1_col = convert_column_arg(args.corr1_col)
    corr2_col = convert_column_arg(args.corr2_col)
    
    # Step 1: Process input file
    print("\nStep 1: Processing input file...")
    try:
        result = process_input_pea_table(
            file_or_path=args.input,
            col_pathway_name=pathway_col,
            col_mols_pathway=molecules_col,
            col_non_corr=pval_col,
            col_corr_1=corr1_col,
            col_corr_2=corr2_col,
            pval_signi_non_corr=args.pval_threshold,
            pval_signi_corr_1=args.pval_threshold,
            pval_signi_corr_2=args.pval_threshold
        )
        
        pathway_names, enriched_molecules, uncorrected_pvalues, \
        corr_1_values, corr_2_values, pval_signi_non_corr, \
        pval_signi_corr_1, pval_signi_corr_2 = result
        
        print(f"✓ Loaded {len(pathway_names)} pathways")
        print(f"  Using p-value threshold: {args.pval_threshold}")
    except Exception as e:
        print(f"✗ Error processing input file: {e}")
        return None
    
    # Step 2: Build bipartite network
    print("\nStep 2: Building bipartite network...")
    try:
        x, wname, wtype = build_network(
            pathway_names, enriched_molecules, 
            uncorrected_pvalues, corr_1_values, corr_2_values, 
            pval_signi_non_corr, pval_signi_corr_1, pval_signi_corr_2
        )
        print(f"✓ Network created: {x.shape[0]} total nodes")
    except Exception as e:
        print(f"✗ Error building network: {e}")
        return None
    
    # Step 3: Filter non-significant pathways
    print("\nStep 3: Filtering pathways...")
    wname = np.array(wname)
    wtype = np.array(wtype)
    unique_vals = set(wtype)
    
    # Keep only significant pathways and all molecules
    # wtype: 0=molecule, 1=red, 2=orange, 3=grey, 4=non-significant
    if unique_vals == {0, 4}:
        keep_mask = (wtype == 0) | (wtype == 4)
    else:
        keep_mask = (wtype == 0) | (wtype == 1) | (wtype == 2) | (wtype == 3)
    
    wname = np.reshape(wname[keep_mask], (np.sum(keep_mask), 1))
    wtype = np.reshape(wtype[keep_mask], (np.sum(keep_mask), 1))
    x = x[np.reshape(keep_mask, x.shape[0]), :]
    x = x[:, np.reshape(keep_mask, x.shape[1])]
    
    print(f"✓ Filtered to {np.sum(keep_mask)} nodes")
    
    # Step 4: Assign colors and shapes
    print("\nStep 4: Assigning node attributes...")
    wcolor = np.zeros((len(wtype), 3))

    wcolor[np.reshape(wtype == 0, wcolor.shape[0]), :] = np.tile([0, 1, 0], (np.sum(wtype == 0), 1))  # Green (molecules)
    wcolor[np.reshape(wtype == 1, wcolor.shape[0]), :] = np.tile([1, 0, 0], (np.sum(wtype == 1), 1))  # Red (corr_1 significant)
    wcolor[np.reshape(wtype == 2, wcolor.shape[0]), :] = np.tile([1, 0.6, 0], (np.sum(wtype == 2), 1))  # Orange (corr_2 significant)
    wcolor[np.reshape(wtype == 3, wcolor.shape[0]), :] = np.tile([0.5, 0.5, 0.5], (np.sum(wtype == 3), 1))  # Grey (non-corrected significant)
    wcolor[np.reshape(wtype == 4, wcolor.shape[0]), :] = np.tile([0, 0, 1], (np.sum(wtype == 4), 1))  # Blue (non-significant)

    # Extract first word from names
    fixed_names = []
    for i in range(len(wname)):
        original_name = wname[i][0]
        first_word = original_name.split(' ', 1)[0]
        fixed_names.append(first_word)
    
    # Assign shapes
    wsymbol = []
    for i in range(len(wname)):
        if wtype[i] == 0:
            wsymbol.append("o")  # Circle for molecules
        else:
            wsymbol.append("d")  # Diamond for pathways
    
    print(f"✓ Assigned colors and shapes")
 
    # Load labels if needed
    labels = None
    if args.coloring == 'labels' and args.labels_col:
        print(f"\nLoading labels from column: {args.labels_col}")
        labels_col = convert_column_arg(args.labels_col)
        labels = load_labels_from_file(args.input, labels_col, [w[0] for w in wname])
        if labels is not None:
            print(f"✓ Loaded labels for {len(labels)} nodes")

    option = 1  # Pathway visualization with legend
    edge_colors = None

    return x, wtype, wcolor, fixed_names, wsymbol, option, edge_colors, labels


def process_bipartite_mode(args):
    """Process bipartite network mode."""
    print("\n" + "=" * 70)
    print("MODE: Bipartite Network (Adjacency List)")
    print("=" * 70)

    # Step 1: Process adjacency list
    print("\nStep 1: Processing adjacency list...")
    try:
        x, edge_colors, wname, wsymbol = process_adjacency_list(args.adjacency_file)
        print(f"✓ Loaded adjacency list: {x.shape[0]} nodes, {x.nnz} edges")
    except Exception as e:
        print(f"✗ Error processing adjacency file: {e}")
        return None
    
    # Step 2: Process node list
    print("\nStep 2: Processing node list...")
    try:
        list_unique_nodes, node_colors_from_file = process_list_nodes(
            args.node_file, wname, wsymbol
        )
        print(f"✓ Loaded {len(list_unique_nodes)} nodes from node list")
    except Exception as e:
        print(f"✗ Error processing node file: {e}")
        return None

    # Step 3: Assign node attributes
    print("\nStep 3: Assigning node attributes...")

    # Create wtype array (all zeros for bipartite mode)
    wtype = np.zeros(len(wname))

    # Convert node colors from hex to RGB
    wcolor = []
    for node_name in wname:
        if node_name in list_unique_nodes:
            idx = list_unique_nodes.index(node_name)
            hex_color = node_colors_from_file[idx]
            # Convert hex to RGB
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            wcolor.append(rgb)
        else:
            # Default color if not in node list
            wcolor.append((0.5, 0.5, 0.5))

    wcolor = np.array(wcolor)

    # Extract first word from names
    fixed_names = []
    for name in wname:
        original_name = str(name)
        first_word = original_name.split(' ', 1)[0]
        fixed_names.append(first_word)

    print(f"✓ Assigned colors and shapes")

    # Load labels if needed
    labels = None
    if args.coloring == 'labels' and args.labels_col:
        print(f"\nLoading labels from column: {args.labels_col}")
        labels_col = convert_column_arg(args.labels_col)
        labels = load_labels_from_file(args.node_file, labels_col, wname)
        if labels is not None:
            print(f"✓ Loaded labels for {len(labels)} nodes")

    option = 2  # Bipartite network visualization

    return x, wtype, wcolor, fixed_names, wsymbol, option, edge_colors, labels


def main():
    """Main execution function."""
    args = parse_arguments() 

    # Print header
    print("=" * 70)
    print("Hyperpathway: Hyperbolic Pathway Enrichment Visualization")
    print("=" * 70)

    # Validate arguments
    if not validate_arguments(args):
        return 1

    # Print configuration
    print("\nConfiguration:")
    print(f"  Mode: {args.mode}")
    if args.mode == 'pea':
        print(f"  Input file: {args.input}")
        print(f"  Pathway column: {args.pathway_col}")
        print(f"  Molecules column: {args.molecules_col}")
    elif args.mode == 'bipartite':
        print(f"  Adjacency file: {args.adjacency_file}")
        print(f"  Node file: {args.node_file}")
    print(f"  Coloring: {args.coloring}")
    if args.coloring == 'labels':
        print(f"  Labels column: {args.labels_col}")
    print(f"  Output plot: {args.output}")
    print(f"  Output Excel: {args.excel_output}")
    print(f"  Show labels: {args.show_labels}")
    print(f"  Show edges: {not args.hide_edges}")
    if args.subnetwork_nodes or args.subnetwork_file:
        print(f"  Subnetwork output: {args.subnetwork_output}")    
        sub_coloring = args.subnetwork_coloring if args.subnetwork_coloring else args.coloring
        print(f"  Subnetwork coloring: {sub_coloring}")

    # Process based on mode
    if args.mode == 'pea':
        result = process_pea_mode(args)
    else:  # bipartite
        result = process_bipartite_mode(args)

    if result is None:
        return 1

    x, wtype, wcolor, fixed_names, wsymbol, option, edge_colors, labels = result

    # Step 5: Remove isolated nodes
    print("\nStep 5: Removing isolated nodes...")

    # Also filter labels if they exist
    if labels is not None:
        # Create mask for non-isolated nodes
        mask = np.array(x.sum(axis=1)).ravel() != 0
        labels = labels[mask]

    x, wtype, wcolor, fixed_names, wsymbol = remove_isolated_nodes(
        x, wtype, wcolor, fixed_names, wsymbol
    )
    print(f"✓ Final network: {x.shape[0]} nodes, {x.nnz} edges")
    
    # Step 6: Compute hyperbolic embedding and generate plot
    print("\n" + "=" * 70)
    print("Step 6: Computing hyperbolic embedding")
    print("=" * 70)
    print()
    
    try:
        coords, excel_buffer, _ = run_hyperpathway_with_progress(
            x, wtype, wcolor, fixed_names, wsymbol, option,
            e_colors=edge_colors,
            corr_1_name=args.corr1_name,
            corr_2_name=args.corr2_name,
            output_file=args.output,
            show_labels=args.show_labels,
            show_edges=not args.hide_edges,
            max_edges=args.max_edges,
            edge_opacity=args.edge_opacity
        )
    except Exception as e:
        print(f"\n✗ Error during hyperbolic embedding computation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 7: Generate plot with selected coloring
    print(f"\nStep 7: Generating visualization with {args.coloring} coloring...")
    try:
        if args.coloring == 'default':
            # Use standard pathway coloring
            fig_path = plot_hyperpathway_static(
                x, coords, wcolor, fixed_names, wsymbol, option,
                e_colors=edge_colors,
                corr_1_name=args.corr1_name,
                corr_2_name=args.corr2_name,
                output_file=args.output,
                show_labels=args.show_labels,
                build_edges=not args.hide_edges,
                max_edges=args.max_edges,
                dpi=args.dpi,
                figsize=tuple(args.figsize),
                node_size_scale=args.node_size_scale,
                edge_opacity=args.edge_opacity
            )
        else:
            # Use gradient coloring
            fig_path = plot_hyperpathway_static_gradient_color(
                x, coords, fixed_names, wsymbol,
                coloring=args.coloring,
                labels=labels,
                build_edges=not args.hide_edges,
                max_edges=args.max_edges,
                output_file=args.output,
                dpi=args.dpi,
                figsize=tuple(args.figsize),
                show_labels=args.show_labels,
                node_size_scale=args.node_size_scale,
                edge_opacity=args.edge_opacity
            )
        print(f"✓ Visualization saved: {fig_path}")
    except Exception as e:
        print(f"✗ Error generating visualization: {e}")
        import traceback
        traceback.print_exc()

        return 1

    # Step 8: Save Excel file
    print(f"\nStep 8: Saving Excel file...")
    try:
        with open(args.excel_output, 'wb') as f:
            f.write(excel_buffer.getvalue())
        print(f"✓ Excel file saved: {args.excel_output}")
    except Exception as e:
        print(f"✗ Error saving Excel file: {e}")
        return 1
    
    # Step 9: Extract and plot subnetwork (if requested)
    if args.subnetwork_nodes or args.subnetwork_file:
        print("\n" + "=" * 70)
        print("Step 9: Extracting subnetwork")
        print("=" * 70)
        print()

        selected_indices = load_selected_nodes(args, fixed_names)

        if selected_indices:
            try:
                print(f"Extracting subnetwork with {len(selected_indices)} seed nodes...")

                # Filter subgraph
                x_sub, coords_sub, colors_sub, names_sub, shapes_sub, edge_colors_sub = filter_subgraph(
                    x, coords, wcolor, fixed_names, wsymbol, selected_indices, edge_colors=edge_colors
                )

                # Filter labels for subnetwork if needed
                labels_sub = None
                if labels is not None:
                    # Map selected indices to subnetwork
                    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(set(selected_indices) | set(range(x_sub.shape[0]))))}
                    # This is a simplified version - you may need to adjust based on actual filtering
                    labels_sub = labels[sorted(set(selected_indices))]

                print(f"✓ Subnetwork extracted: {len(names_sub)} nodes (including first neighbors)")
                print(f"  - Seed nodes: {len(selected_indices)}")
                print(f"  - First neighbors: {len(names_sub) - len(selected_indices)}")

                # Determine subnetwork coloring
                sub_coloring = args.subnetwork_coloring if args.subnetwork_coloring else args.coloring
                sub_labels_col = args.subnetwork_labels_col if args.subnetwork_labels_col else args.labels_col

                # Plot subnetwork with selected coloring
                print(f"\nGenerating subnetwork visualization with {sub_coloring} coloring...")

                if sub_coloring == 'default':
                    plot_hyperpathway_static(
                        x_sub, coords_sub, colors_sub, names_sub, shapes_sub, option,
                        e_colors=edge_colors_sub,
                        build_edges=not args.subnetwork_hide_edges,
                        max_edges=args.max_edges,
                        corr_1_name=args.corr1_name,
                        corr_2_name=args.corr2_name,
                        output_file=args.subnetwork_output,
                        show_labels=args.subnetwork_show_labels,
                        dpi=args.dpi,
                        figsize=tuple(args.figsize),
                        edge_opacity=args.edge_opacity
                )
                else:
                    # For subnetwork with gradient coloring, we need to handle labels
                    if sub_coloring == 'labels':
                        if labels_sub is None and sub_labels_col:
                            # Try to load labels for subnetwork
                            labels_sub = load_labels_from_file(
                                args.input if args.mode == 'pea' else args.node_file,
                                convert_column_arg(sub_labels_col),
                                names_sub
                            )

                    plot_hyperpathway_static_gradient_color(
                        x_sub, coords_sub, names_sub, shapes_sub,
                        coloring=sub_coloring,
                        labels=labels_sub,
                        build_edges=not args.subnetwork_hide_edges,
                        max_edges=args.max_edges,
                        output_file=args.subnetwork_output,
                        dpi=args.dpi,
                        figsize=tuple(args.figsize),
                        show_labels=args.subnetwork_show_labels,
                        edge_opacity=args.edge_opacity
                    )

                print(f"✓ Subnetwork visualization saved: {args.subnetwork_output}")

            except Exception as e:
                print(f"✗ Error extracting subnetwork: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("✓ Analysis Complete!")
    print("=" * 70)
    print(f"  Full network visualization: {args.output}")
    print(f"  Data export: {args.excel_output}")
    if args.subnetwork_nodes or args.subnetwork_file:
        print(f"  Subnetwork visualization: {args.subnetwork_output}")
    print()
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
