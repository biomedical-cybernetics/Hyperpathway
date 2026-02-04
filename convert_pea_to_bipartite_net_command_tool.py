import os
import re
import io
import ast
import requests
import pandas as pd
import numpy as np
import networkx as nx
from scipy.io import savemat
from scipy.sparse import lil_matrix
from scipy.sparse import lil_matrix, csr_matrix # TO REMOVE AFTER TEST



def process_input_pea_table(file_or_path, 
                            col_pathway_name=None,
                            col_mols_pathway=None,
                            col_non_corr=None,
                            col_corr_1=None,
                            col_corr_2=None,
                            pval_signi_non_corr=0.05,
                            pval_signi_corr_1=0.05,
                            pval_signi_corr_2=0.05):
    """
    Process pathway enrichment analysis table from file or URL.
    
    Parameters:
    -----------
    file_or_path : str or file-like object
        Path to file, URL, or file object
    col_pathway_name : str or int
        Column name or column index (0-based) for pathway names
    col_mols_pathway : str or int
        Column name or column index (0-based) for enriched molecules
    col_non_corr : str or int, optional
        Column name or column index (0-based) for non-corrected p-values
    col_corr_1 : str or int, optional
        Column name or column index (0-based) for first corrected p-values
    col_corr_2 : str or int, optional
        Column name or column index (0-based) for second corrected p-values
    pval_signi_non_corr : float, default=0.05
        Significance threshold for non-corrected p-values
    pval_signi_corr_1 : float, default=0.05
        Significance threshold for first corrected p-values
    pval_signi_corr_2 : float, default=0.05
        Significance threshold for second corrected p-values
    
    Returns:
    --------
    tuple : (pathway_names, enriched_molecules, uncorrected_pvalues, 
             corr_1_values, corr_2_values, pval_signi_non_corr, 
             pval_signi_corr_1, pval_signi_corr_2)
    """
    
    def resolve_column(col_spec, df):
        """
        Resolve column specification to actual column name.
        
        Parameters:
        -----------
        col_spec : str, int, or None
            Column name or column index (0-based)
        df : DataFrame
            The dataframe to resolve column from
            
        Returns:
        --------
        str or None : Resolved column name or None if col_spec is None
        """
        if col_spec is None:
            return None
        
        if isinstance(col_spec, int):
            if col_spec < 0 or col_spec >= len(df.columns):
                raise ValueError(
                    f"Column index {col_spec} is out of range. "
                    f"Valid range: 0-{len(df.columns)-1}"
                )
            return df.columns[col_spec]
        elif isinstance(col_spec, str):
            if col_spec not in df.columns:
                raise ValueError(
                    f"Column '{col_spec}' not found in dataframe. "
                    f"Available columns: {list(df.columns)}"
                )
            return col_spec
        else:
            raise TypeError(
                f"Column specification must be str (column name) or int (column index), "
                f"got {type(col_spec).__name__}"
            )
    
    # Read Excel file robustly based on extension
    if hasattr(file_or_path, 'read'):
        filename = file_or_path.name
        ext = os.path.splitext(filename)[1].lower()

        if ext == '.xlsx':
            df = pd.read_excel(file_or_path, engine='openpyxl')
        elif ext == '.xls':
            df = pd.read_excel(file_or_path, engine='xlrd')
        elif ext == '.csv':
            df = pd.read_csv(file_or_path)
        elif ext == '.tsv':
            df = pd.read_csv(file_or_path, sep='\t')
        else:
            raise ValueError("Unsupported file format. Please upload .xls, .xlsx, .csv or .tsv")

    elif isinstance(file_or_path, str) and file_or_path.startswith(("http://", "https://")):
        ext = os.path.splitext(file_or_path)[1].lower()
        response = requests.get(file_or_path)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch file from URL: {file_or_path}")
        file_bytes = io.BytesIO(response.content)

        if ext == '.xlsx':          
            df = pd.read_excel(file_bytes, engine='openpyxl')
        elif ext == '.xls':
            df = pd.read_excel(file_bytes, engine='xlrd')
        elif ext == '.csv':
            df = pd.read_csv(file_bytes)
        elif ext == '.tsv':
            df = pd.read_csv(file_bytes, sep='\t')
        else:
            raise ValueError("Unsupported file format at URL. Must be .xls, .xlsx, .csv or .tsv")

    else:
        ext = os.path.splitext(str(file_or_path))[1].lower()
        if ext == '.xlsx':
            df = pd.read_excel(file_or_path, engine='openpyxl')
        elif ext == '.xls':
            df = pd.read_excel(file_or_path, engine='xlrd')
        elif ext == '.csv':
            df = pd.read_csv(file_or_path)
        elif ext == '.tsv':
            df = pd.read_csv(file_or_path, sep='\t')
        else:
            raise ValueError("Unsupported file format. Please upload .xls, .xlsx, .csv or .tsv")

    # Clean and validate dataframe
    num_cols = df.shape[1]
    col_names = list(df.columns)

    if num_cols < 2:
        raise ValueError(
            f"Incorrect file format:\n"
            f"- The table contains {num_cols} columns, but at least 2 columns are required.\n"
            f"Detected columns: {col_names}\n"
            f"Please ensure the file follows the expected format and includes a header with column names."
        )

    # Validate required columns are provided
    if col_pathway_name is None or col_mols_pathway is None:
        raise ValueError(
            f"'col_pathway_name' and 'col_mols_pathway' must be specified.\n"
            f"Available columns: {col_names}"
        )

    # Resolve column specifications (names or indices) to actual column names
    resolved_pathway_name = resolve_column(col_pathway_name, df)
    resolved_mols_pathway = resolve_column(col_mols_pathway, df)
    resolved_non_corr = resolve_column(col_non_corr, df)
    resolved_corr_1 = resolve_column(col_corr_1, df)
    resolved_corr_2 = resolve_column(col_corr_2, df)

    # Collect selected columns
    selected_cols = [c for c in [resolved_pathway_name, resolved_mols_pathway, 
                                  resolved_non_corr, resolved_corr_1, resolved_corr_2] 
                     if c is not None]

    # Validate column mapping
    if len(set(selected_cols)) != len(selected_cols):
        raise ValueError("Each column role must be assigned to a different column.")

    # Extract data using resolved column names
    pathway_names = df[resolved_pathway_name].astype(str).tolist()
    enriched_molecules = df[resolved_mols_pathway].astype(str).tolist()
    
    # Optional p-values: only use if selected
    uncorrected_pvalues = df[resolved_non_corr] if resolved_non_corr else pd.Series([1]*len(df))
    corr_1_values = df[resolved_corr_1] if resolved_corr_1 else pd.Series([1]*len(df))
    corr_2_values = df[resolved_corr_2] if resolved_corr_2 else pd.Series([1]*len(df))

    return (pathway_names, enriched_molecules, uncorrected_pvalues, 
            corr_1_values, corr_2_values, pval_signi_non_corr, 
            pval_signi_corr_1, pval_signi_corr_2)



def build_network(pathway_names, enriched_molecules, uncorrected_pvalues, corr_1_values, corr_2_values, pval_signi_non_corr = 0.05, pval_signi_corr_1 = 0.05, pval_signi_corr_2 = 0.05):

    # Gather all molecules
    all_molecules = []
    for entry in enriched_molecules:
        all_molecules.extend(re.split(r'[;,|]\s*', entry))
    unique_molecules = sorted(set(all_molecules))
    unique_pathways = sorted(set(pathway_names))

    num_pathways = len(unique_pathways)
    num_unique_molecules = len(unique_molecules)
    # Build dictionary comprehension. For example:
    # mol_to_idx = {"H2O": 0, "CO2": 1, "O2": 2}
    mol_to_idx = {mol: j for j, mol in enumerate(unique_molecules)}

    #x = np.zeros((num_pathways + num_unique_molecules, num_pathways + num_unique_molecules))
    x = lil_matrix((num_pathways + num_unique_molecules,
                    num_pathways + num_unique_molecules))
    wtype = np.zeros(num_pathways + num_unique_molecules)
    
    # FIXED: Determine how many corrected p-value columns were actually provided
    # Check if all values are 1 (default, meaning column wasn't provided)
    corr_1_provided = not all(corr_1_values == 1)
    corr_2_provided = not all(corr_2_values == 1)
    num_corr_cols_provided = sum([corr_1_provided, corr_2_provided])
    
    # Make sure that in red we have p-value with smaller mean (only if both corrected p-values provided)
    if num_corr_cols_provided == 2 and corr_1_values.mean() < corr_2_values.mean():
        tmp = corr_2_values
        corr_2_values = corr_1_values
        corr_1_values = tmp

    for i in range(num_pathways):
        idx_p = [p == unique_pathways[i] for p in pathway_names]
        idx_list = [i for i, flag in enumerate(idx_p) if flag]
        # Flatten molecules
        molecules = []
        for j in idx_list:
            molecules.extend(re.split(r'[;,|]\s*', enriched_molecules[j]))
        
        #print(unique_pathways[i])
        #print(molecules)
        if sum(idx_p) == 1:
            corr_1 = corr_1_values[idx_p].iloc[0]
            corr_2 = corr_2_values[idx_p].iloc[0]
            uncorrected = uncorrected_pvalues[idx_p].iloc[0]
        else:
            corr_1 = 1
            corr_2 = 1
            uncorrected = 1
  
        # FIXED: Check the significance of p-values
        # NEW LOGIC: When only one corrected p-value is provided, it should always be red
        if num_corr_cols_provided == 1:
            # Only one corrected p-value column provided
            if corr_1_provided and corr_1 <= pval_signi_corr_1:
                wtype[i] = 1  # red
            elif corr_2_provided and corr_2 <= pval_signi_corr_2:
                wtype[i] = 1  # red (treat as red when it's the only corrected one)
            elif uncorrected <= pval_signi_non_corr:
                wtype[i] = 3  # grey
            else:
                wtype[i] = 4  # non-significant
        else:
            # Original logic: either 0 or 2 corrected p-value columns provided
            if corr_1 <= pval_signi_corr_1:
                wtype[i] = 1  # red
            elif corr_2 <= pval_signi_corr_2:
                wtype[i] = 2  # orange
            elif uncorrected <= pval_signi_non_corr:
                wtype[i] = 3  # grey
            else:
                wtype[i] = 4
        # for mol in molecules:
        #     try:
        #         idx = unique_molecules.index(mol)
        #         x[i, num_pathways + idx] = 1
        #         x[num_pathways + idx, i] = 1
        #     except ValueError:
        #         raise ValueError(f"Impossible to find index for molecule: {mol}")

        for mol in molecules:
            if mol in mol_to_idx:
                idx = mol_to_idx[mol]
                x[i, num_pathways + idx] = 1
                x[num_pathways + idx, i] = 1
            else:
                raise ValueError(f"Impossible to find index for molecule: {mol}")

    x = x.tocsr()
    wname = unique_pathways + unique_molecules

    return x, wname, wtype



def detect_color_format(color_str):
    color_str = color_str.strip().strip('"').strip("'")
    # Check for HEX (e.g., "#FF00AA" or "#ccc")
    if re.fullmatch(r'#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})', color_str):
        return 'hex'
    
    # Space-separated RGB without brackets (e.g., "1 0 0" or "0.5 0.4 0.2") 
    if re.fullmatch(r'\d+(\.\d+)?\s+\d+(\.\d+)?\s+\d+(\.\d+)?', color_str):
        return 'rgb'
    
    return 'unknown'


def parse_rgb_string(color_str):
    color_str = color_str.strip()
    parts = re.split(r'[,\s]+', color_str)
    if len(parts) != 3:
        return f"Error: invalid RGB format → '{color_str}'"
    try:
        return tuple(int(p) for p in parts) 
    except ValueError: 
        return f"Error: non-numeric values in RGB → '{color_str}'"


def rgb_to_hex(rgb):
    """Convert an RGB tuple (each 0–1 or 0–255) to HEX color."""
    # Normalize if 0–1 values
    if all(0 <= val <= 1 for val in rgb):
        rgb = [int(val * 255) for val in rgb]
    return '#{:02X}{:02X}{:02X}'.format(*rgb)


def process_adjacency_list(file_or_path):
    # Read Excel file robustly based on extension
    if hasattr(file_or_path, 'read'):
        filename = file_or_path.name
        ext = os.path.splitext(filename)[1].lower()

        if ext == '.xlsx':
            df = pd.read_excel(file_or_path, engine='openpyxl')
        elif ext == '.xls':
            df = pd.read_excel(file_or_path, engine='xlrd')
        elif ext == '.csv':
            df = pd.read_csv(file_or_path)
        elif ext == '.tsv':
            df = pd.read_csv(file_or_path, sep='\t')
        else:
            raise ValueError("Unsupported file format. Please upload .xls, .xlsx, .csv or .tsv")
    else:
        ext = os.path.splitext(str(file_or_path))[1].lower()
        if ext == '.xlsx':
            df = pd.read_excel(file_or_path, engine='openpyxl')
        elif ext == '.xls':
            df = pd.read_excel(file_or_path, engine='xlrd')
        elif ext == '.csv':
            df = pd.read_csv(file_or_path)
        elif ext == '.tsv':
            df = pd.read_csv(file_or_path, sep='\t')
        else:
            raise ValueError("Unsupported file format. Please upload .xls, .xlsx, .csv or .tsv")

    # Clean and validate dataframe
    if df.shape[1] > 3:
        raise ValueError(f"Unexpected format: dataframe has {df.shape[1]} columns. 3 columns expected. Refer to the file format on the webapp.")
    elif df.shape[1] == 3:
        wcolor = df.iloc[:, 2].tolist()
        color_format = detect_color_format(wcolor[0])
        if color_format == 'rgb':
            for i, c in enumerate(wcolor):
                wcolor[i] = rgb_to_hex(parse_rgb_string(c))
    else:
        wcolor = []


    # Extract node lists and auto-cast to string or float depending on dtype
    node1 = df.iloc[:, 0].tolist()        
    node2 = df.iloc[:, 1].tolist()
    #wcolor = df.iloc[:, 2].tolist()
    # Get unique nodes from both sets
    unique_node1 = sorted(set(node1))
    unique_node2 = sorted(set(node2))

    all_nodes = unique_node1 + unique_node2
    node_index = {node: i for i, node in enumerate(all_nodes)}

    n = len(all_nodes)
    # use a sparse matrix instead of np.zeros 
    x = lil_matrix((n, n), dtype=int)

    # Build edge color dictionary mapping (i, j) -> color
    edge_color_dict = {}
    for idx, (a, b) in enumerate(zip(node1, node2)):
        i, j = node_index[a], node_index[b]
        x[i, j] = 1
        x[j, i] = 1  # Make symmetric
        # Store color for this edge (use canonical ordering: smaller index first)
        if wcolor:
            edge_key = (min(i, j), max(i, j))
            edge_color_dict[edge_key] = wcolor[idx]
    
    # Check bipartiteness
    # Convert adjacency matrix to NetworkX graph
    G = nx.from_scipy_sparse_matrix(x)
    # Build node sets: pathways (0 to num_pathways-1), molecules (num_pathways to end)
    n1 = set(range(len(unique_node1)))
    n2 = set(range(len(unique_node1), len(unique_node1) + len(unique_node2)))
    is_bipartite = nx.algorithms.bipartite.is_bipartite_node_set(G, n1)

    if not is_bipartite:
        raise ValueError("Generated network is not bipartite.")

    # Assign shape
    node_shape = ['d'] * len(unique_node1) + ['o'] * len(unique_node2)

    # Assign node labels
    wname = all_nodes

    x = x.tocsr()

    # Return edge_color_dict instead of wcolor list
    return x, edge_color_dict, wname, node_shape


def process_list_nodes(file_or_path, node_name, node_shape):
    # Read Excel file robustly based on extension
    if hasattr(file_or_path, 'read'):
        filename = file_or_path.name
        ext = os.path.splitext(filename)[1].lower()

        if ext == '.xlsx':
            df = pd.read_excel(file_or_path, engine='openpyxl')
        elif ext == '.xls':
            df = pd.read_excel(file_or_path, engine='xlrd')
        elif ext == '.csv':
            df = pd.read_csv(file_or_path)
        elif ext == '.tsv':
            df = pd.read_csv(file_or_path, sep='\t')
        else:
            raise ValueError("Unsupported file format. Please upload .xls, .xlsx or .csv")
    else:
        ext = os.path.splitext(str(file_or_path))[1].lower()
        if ext == '.xlsx':
            df = pd.read_excel(file_or_path, engine='openpyxl')
        elif ext == '.xls':
            df = pd.read_excel(file_or_path, engine='xlrd')
        elif ext == '.csv':
            df = pd.read_csv(file_or_path)
        elif ext == '.tsv':
            df = pd.read_csv(file_or_path, sep='\t')            
        else:
            raise ValueError("Unsupported file format. Please upload .xls, .xlsx or .csv")

    # Clean and validate dataframe
    if df.shape[1] > 2:
        raise ValueError(f"Unexpected format: dataframe has {df.shape[1]} column. 2 columns expected. Refer to the file format on the webapp.")
    elif df.shape[1] == 2:
        wcolor = df.iloc[:, 1].tolist()
        color_format = detect_color_format(wcolor[0])
        if color_format == 'rgb':
            for i, c in enumerate(wcolor):
                wcolor[i] = rgb_to_hex(parse_rgb_string(c))
    else:
        wcolor = []

    # Extract node lists and auto-cast to string or float depending on dtype
    list_nodes = df.iloc[:, 0].tolist()


    if wcolor:
        # Map each node to its corresponding wcolor (assuming 1-to-1 mapping)
        node_to_color = {node: color for node, color in zip(list_nodes, wcolor)}
    else:
        node_to_color = {}
        for node in list_nodes:
            if node not in node_name:
                raise ValueError(
                    f"Node '{node}' in node list is not found in the adjacency list. "
                    f"Ensure the names match between the node list file and the adjacency list."
                )
            idx = node_name.index(node)
            symbol = node_shape[idx]
            if symbol == 'o':
                node_to_color[node] = '#FF0000'  # Red
            elif symbol == 'd':
                node_to_color[node] = '#000000'  # Black
            else:
                raise ValueError(f"Unexpected symbol '{symbol}' at index {idx} for node '{node}'")

    # Return sorted nodes and their corresponding colors
    unique_nodes = sorted(node_to_color.keys())
    wcolor_unique = [node_to_color[node] for node in unique_nodes]


    return unique_nodes, wcolor_unique


def save_to_mat(x, wname, wtype, out_path="data_final_lipea_python.mat"):
    savemat(out_path, {"x": x, "wname": wname, "wtype": wtype})


