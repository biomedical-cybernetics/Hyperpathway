import os
import re
import io
import ast
import requests
import pandas as pd
import numpy as np
import networkx as nx
import streamlit as st
from scipy.io import savemat
from scipy.sparse import lil_matrix
from scipy.sparse import lil_matrix, csr_matrix # TO REMOVE AFTER TEST



def process_input_pea_table(file_or_path, pval_signi=0.05):
    tag_demo = False   
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
        tag_demo = True
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
            f"❌ Incorrect file format:\n"
            f"- The table contains **{num_cols} columns**, but **at least 2 columns** are required.\n\n"
            f"Detected columns:\n{col_names}\n\n"
            f"Please ensure the file follows the expected format provided on the webapp and that it includes a header with column names."
        )

    if 2 <= num_cols and tag_demo == False:
        st.markdown("### 📌 Select which columns correspond to each colname:")

        # --- Always show all selectboxes ---
        col_pathway_name = st.selectbox("Pathway name:", options=col_names, index=None, key="col_pathway_name")
        col_mols_pathway = st.selectbox("Enriched molecules in pathway:", options=col_names, index=None, key="col_mols_pathway")
        col_non_corr     = st.selectbox("Non-corrected p-value:", options=col_names, index=None, key="col_non_corr")
        col_corr_1       = st.selectbox("Corrected p-value #1:", options=col_names, index=None, key="col_corr_1")
        col_corr_2       = st.selectbox("Corrected p-value #2:", options=col_names, index=None, key="col_corr_2")

        # --- Collect all chosen values ---
        selected_cols = [c for c in [col_pathway_name, col_mols_pathway, col_non_corr, col_corr_1, col_corr_2] if c is not None]

        # --- Validation ---
        if len(selected_cols) < 2:
            st.warning(f"➡️ Please select your 'Pathway name' column, your 'Enriched molecules in pathway' column and at least one p-value if possible.")
            return None


        #if len(selected_cols) < num_cols:
        #    st.warning(f"➡️ Please select {num_cols} distinct columns.")
        #    return None, None, None

        if len(set(selected_cols)) != len(selected_cols):
            st.error("❌ Each column role must be assigned to a different column.")
            return None

        if col_pathway_name is None or col_mols_pathway is None:
            st.error("❌ 'Pathway name' and 'Enriched molecules in pathway' must be assigned.")
            return None

        st.success("✅ Column mapping valid. Building bipartite network...")

        THR_KEYS = ("thr_non_corr", "thr_corr_1", "thr_corr_2")
        MASTER_KEY = "thr_master"
        PENDING_KEY = "thr_sync_pending"
        PREV_KEY = "thr_prev_thresholds"

        default_thr = 0.05  # or 0.005

        # --- init master + bookkeeping ---
        if MASTER_KEY not in st.session_state:
            st.session_state[MASTER_KEY] = float(default_thr)

        if PREV_KEY not in st.session_state:
            st.session_state[PREV_KEY] = {k: float(st.session_state.get(k, st.session_state[MASTER_KEY])) for k in THR_KEYS}

        if PENDING_KEY not in st.session_state:
            st.session_state[PENDING_KEY] = False

        # --- if sync is pending, apply it BEFORE widget creation ---
        if st.session_state[PENDING_KEY]:
            v = float(st.session_state[MASTER_KEY])
            for k in THR_KEYS:
                st.session_state[k] = v
            st.session_state[PENDING_KEY] = False

        # --- ensure widget keys exist BEFORE widget creation (so they don't default to 0) ---
        v0 = float(st.session_state[MASTER_KEY])
        for k in THR_KEYS:
            if k not in st.session_state:
                st.session_state[k] = v0

        # --- build UI widgets (no on_change needed) ---
        if col_non_corr:
            st.number_input(
                f"Significance threshold for {col_non_corr}",
                min_value=0.0, max_value=1.0, value = st.session_state[MASTER_KEY],
                step=0.001,
                format="%.2f",
                key="thr_non_corr",
            )

        if col_corr_1:
            st.number_input(
                f"Significance threshold for {col_corr_1}",
                min_value=0.0, max_value=1.0, value = st.session_state[MASTER_KEY],
                step=0.001,
                format="%.2f",
                key="thr_corr_1",
            )

        if col_corr_2:
            st.number_input(
                f"Significance threshold for {col_corr_2}",
                min_value=0.0, max_value=1.0, value = st.session_state[MASTER_KEY],
                step=0.001,
                format="%.2f",
                key="thr_corr_2",
            )

        # --- detect which widget changed THIS run (compared to previous run) ---
        curr = {k: float(st.session_state.get(k, v0)) for k in THR_KEYS}
        prev = st.session_state[PREV_KEY]

        changed = [k for k in THR_KEYS if abs(curr[k] - prev.get(k, curr[k])) > 1e-12]

        if changed:
            # pick the first changed key (normally only one changes per interaction)
            new_v = curr[changed[0]]
            st.session_state[MASTER_KEY] = float(new_v)
            st.session_state[PENDING_KEY] = True

            # update prev BEFORE rerun to avoid loops
            st.session_state[PREV_KEY] = curr
            st.rerun()
        else:
            st.session_state[PREV_KEY] = curr

        # --- read thresholds for downstream logic ---
        pval_signi_non_corr = float(st.session_state["thr_non_corr"])
        pval_signi_corr_1   = float(st.session_state["thr_corr_1"])
        pval_signi_corr_2   = float(st.session_state["thr_corr_2"])

        def _sig_label(cols):
            return f"Significant in {', '.join(cols)} tests"

        # Pick which p-value columns you want included in the labels:
        corr_cols = [c for c in [col_non_corr, col_corr_1, col_corr_2] if c is not None and c in df.columns]

        # If the threshold is identical for all selected p-value columns, set it once:
        p_thr = pval_signi_non_corr  # or whatever single threshold you use everywhere

        # Compute "significant hits" per column (count of rows <= threshold)
        sig_counts = {}
        valid_counts = {}

        for c in corr_cols:
            s = pd.to_numeric(df[c], errors="coerce")   # non-numeric -> NaN
            valid_counts[c] = int(s.notna().sum())
            sig_counts[c]   = int((s <= p_thr).sum(skipna=True))

        # Choose strictest as the one with fewer significant hits.
        # Tie-breaker (optional): fewer valid rows, then name for determinism.
        strictest_col = None
        if corr_cols:
            strictest_col = min(
                corr_cols,
                key=lambda c: (sig_counts.get(c, 10**18), valid_counts.get(c, 10**18), c)
            )

        # Build display names
        if corr_cols:
            st.session_state["corr_1_display_name"] = "Significant in all p-value tests"

            corr_2_cols = [c for c in corr_cols if c != strictest_col]
            # If removing strictest empties the list (only 1 col selected), keep it non-empty:
            if not corr_2_cols:
                corr_2_cols = corr_cols

            st.session_state["corr_2_display_name"] = _sig_label(corr_2_cols)
        else:
            st.session_state["corr_1_display_name"] = "Significant in all p-value tests"
            st.session_state["corr_2_display_name"] = "Significant in []"

        # Keep / adjust this as you like:
        st.session_state["non_corr_display_name"] = "Significant in non-corrected p-value"

        # --- Use selected column names safely ---
        pathway_names = df[col_pathway_name].astype(str).tolist()
        enriched_molecules = df[col_mols_pathway].astype(str).tolist()
        # Optional p-values: only use if selected
        uncorrected_pvalues = df[col_non_corr] if col_non_corr else pd.Series([1]*len(df))
        corr_1_values           = df[col_corr_1]       if col_corr_1       else pd.Series([1]*len(df))
        corr_2_values         = df[col_corr_2]     if col_corr_2     else pd.Series([1]*len(df))

        return pathway_names, enriched_molecules, uncorrected_pvalues, corr_1_values, corr_2_values, pval_signi_non_corr, pval_signi_corr_1, pval_signi_corr_2


@st.cache_data(show_spinner=False)
def cached_build_network(pathway_names, enriched_molecules, uncorrected_pvalues, corr_1_values, corr_2_values, pval_signi_non_corr = 0.05, pval_signi_corr_1 = 0.05, pval_signi_corr_2 = 0.05):

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
    # Make sure that in red we have p-value with smaller mean
    if corr_1_values.mean() < corr_2_values.mean():
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

        # Check if corrections were selected before assigning colors
        # Determine which corrections are actually being used (not all 1.0)
        has_real_corr_1 = corr_1_values.min() < 1.0
        has_real_corr_2 = corr_2_values.min() < 1.0
  
        # Check the significance of p-values
        if has_real_corr_1 and corr_1 <= pval_signi_corr_1:
            wtype[i] = 1  # RED - first correction
        elif has_real_corr_2 and corr_2 <= pval_signi_corr_2:
            # ⭐ If corr_1 doesn't exist, corr_2 should also be RED (not orange)
            if not has_real_corr_1:
                wtype[i] = 1  # RED - only correction available
            else:
                wtype[i] = 2  # ORANGE - second correction (only if TWO corrections exist)
        elif uncorrected <= pval_signi_non_corr:
            wtype[i] = 3  # GREY - non-corrected
        else:
            wtype[i] = 4  # BLUE - non-significant

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


# @st.cache_resource(show_spinner=False)
# def cached_process_input_pea_table(file_bytes: bytes, filename: str, pval_signi: float = 0.05):
#     """
#     Cached wrapper around process_input_pea_table.
#     file_bytes: raw bytes from uploaded file
#     filename: original filename (used to detect extension)
#     """
#     # Wrap bytes in a file-like object
#     buffer = io.BytesIO(file_bytes)
#     buffer.name = filename  # give BytesIO a .name attribute so the function works
#     return process_input_pea_table(buffer, pval_signi)


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
    node1 = df.iloc[:, 0].astype(str).tolist()     
    node2 = df.iloc[:, 1].astype(str).tolist()
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
    
    # STORE ORIGINAL NODE TABLE FOR LABEL MODE (Option 2)
    st.session_state.node_dataframe = df.copy()
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
    wcolor_unique_hex = [node_to_color[node] for node in unique_nodes]
    
    # Convert HEX colors to RGB arrays (0-1 range)
    wcolor_unique_rgb = []
    for hex_color in wcolor_unique_hex:
        # Remove '#' if present
        hex_color = hex_color.lstrip('#')
        # Convert to RGB (0-1 range)
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        wcolor_unique_rgb.append(np.array([r, g, b]))

    return unique_nodes, wcolor_unique_rgb


def save_to_mat(x, wname, wtype, out_path="data_final_lipea_python.mat"):
    savemat(out_path, {"x": x, "wname": wname, "wtype": wtype})



