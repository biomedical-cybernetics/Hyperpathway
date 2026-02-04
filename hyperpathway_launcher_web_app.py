from os import makedirs
from copy import deepcopy
from os.path import abspath, dirname, exists as path_exists, join as join_path
from scipy.sparse import csgraph
from scipy.sparse import issparse, csr_matrix
from scipy.sparse import triu
from matplotlib.colors import hsv_to_rgb
import os
os.environ.setdefault("SCIPY_USE_PROPACK", "1")
os.environ.setdefault("USE_PROPACK", "1")
from scipy.sparse.linalg import svds
from PIL import Image
from streamlit_plotly_events import plotly_events
from concurrent.futures import ThreadPoolExecutor
from convert_pea_to_bipartite_net import process_input_pea_table, cached_build_network, process_adjacency_list, process_list_nodes
import atexit
import base64
import uuid
import powerlaw
import kaleido
import itertools 
import io
import scipy.io as sio
import scipy as sp
import numpy as np
import time
import re
from datetime import datetime, timedelta 
import threading
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1
import pandas as pd
import plotly.graph_objects as go


_SVD_POOL = None

def scroll_to_element(element_id):
    """Inject JavaScript to scroll to a specific element."""
    st.components.v1.html(
        f"""
        <script>
            window.parent.document.getElementById('{element_id}').scrollIntoView({{
                behavior: 'smooth',
                block: 'center'
            }});
        </script>
        """,
        height=0,
    )


def _auto_download_bytes(data: bytes, filename: str, mime: str):
    """
    Trigger a browser download automatically via a hidden <a download> + JS click.
    Note: some browsers/popup blockers may interfere; if so, we also show a manual link.
    """
    b64 = base64.b64encode(data).decode("utf-8")
    element_id = f"dl_{uuid.uuid4().hex}"
    html = f"""
    <a id="{element_id}" download="{filename}" href="data:{mime};base64,{b64}" style="display:none;">download</a>
    <script>
      const a = document.getElementById("{element_id}");
      if (a) a.click();
    </script>
    <p style="margin:0.25rem 0 0 0;">
      If the download didn't start automatically,
      <a download="{filename}" href="data:{mime};base64,{b64}">click here</a>.
    </p>
    """
    st.components.v1.html(html, height=40)


def add_download_buttons(fig, key_suffix, filename, png_scale=4):
    with st.expander("📥 Download High-Resolution Figure", expanded=False):
        cols = st.columns(3)

        # SVG
        with cols[0]:
            if st.button("📥 SVG", key=f"svg_{key_suffix}"):
                st.info("Preparing file for download, please wait…")
                with st.spinner("Exporting SVG…"):
                    try:
                        svg_bytes = fig.to_image(format="svg")
                        _auto_download_bytes(svg_bytes, f"{filename}.svg", "image/svg+xml")
                    except Exception as e:
                        st.error(f"❌ Export failed: {type(e).__name__}: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

        # PNG
        with cols[1]:
            if st.button("📥 PNG 300DPI", key=f"png_{key_suffix}"):
                st.info("Preparing file for download, please wait…")
                with st.spinner("Exporting PNG…"):
                    try:
                        png_bytes = fig.to_image(format="png", scale=png_scale)
                        _auto_download_bytes(png_bytes, f"{filename}_300dpi.png", "image/png")
                    except Exception as e:
                        st.error(f"❌ Export failed: {type(e).__name__}: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

        # PDF
        with cols[2]:
            if st.button("📥 PDF", key=f"pdf_{key_suffix}"):
                st.info("Preparing file for download, please wait…")
                with st.spinner("Exporting PDF…"):
                    try:
                        pdf_bytes = fig.to_image(format="pdf")
                        _auto_download_bytes(pdf_bytes, f"{filename}.pdf", "application/pdf")
                    except Exception as e:
                        st.error(f"❌ Export failed: {type(e).__name__}: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())


def parse_color_value(color_str):
    """
    Parse color string in various formats:
    - HEX: '#FF0000' or 'FF0000'
    - RGB space-separated: '255 0 0' or '1.0 0 0'
    - RGB list string: '[255, 0, 0]'
    
    Returns: RGB array [0-1 range] or None if invalid
    """
    if not color_str or pd.isna(color_str):
        return None
    
    color_str = str(color_str).strip()
    
    # Handle HEX format
    if color_str.startswith('#'):
        color_str = color_str[1:]
    
    if re.match(r'^[0-9A-Fa-f]{6}$', color_str):
        # Valid HEX color
        r = int(color_str[0:2], 16) / 255.0
        g = int(color_str[2:4], 16) / 255.0
        b = int(color_str[4:6], 16) / 255.0
        return np.array([r, g, b])
    
    # Handle RGB space-separated
    if ' ' in color_str:
        parts = color_str.split()
        if len(parts) == 3:
            try:
                rgb = [float(x) for x in parts]
                # Check if 0-255 range
                if all(0 <= x <= 255 for x in rgb):
                    if any(x > 1 for x in rgb):
                        # Convert 0-255 to 0-1
                        rgb = [x / 255.0 for x in rgb]
                    return np.array(rgb)
            except ValueError:
                pass
    
    # Handle RGB list format '[255, 0, 0]'
    if color_str.startswith('[') and color_str.endswith(']'):
        try:
            rgb = eval(color_str)
            if len(rgb) == 3 and all(isinstance(x, (int, float)) for x in rgb):
                if all(0 <= x <= 255 for x in rgb):
                    if any(x > 1 for x in rgb):
                        rgb = [x / 255.0 for x in rgb]
                    return np.array(rgb)
        except:
            pass
    
    return None


def place_legend_below(fig, *, n_items: int, item_height_px: int = 26, extra_px: int = 20):
    """
    Put Plotly legend below the plot (horizontal) and reserve enough bottom margin
    so it won't overlap or get clipped.
    """
    # Rough but robust: estimate how many legend rows will be needed.
    # Plotly wraps legend items horizontally; we can't know wrap points reliably without
    # pixel measurements, so we conservatively assume ~3 items per row.
    items_per_row = 3
    n_rows = max(1, (n_items + items_per_row - 1) // items_per_row)

    bottom_margin = n_rows * item_height_px + extra_px

    fig.update_layout(
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=0.04, yanchor="top",   # below plotting area (paper coords)
            title_text=None,
            bgcolor="rgba(255,255,255,0)",  # transparent
        ),
        margin=dict(b=bottom_margin)
    )
    return fig


def process_custom_node_colors(uploaded_file, full_names):
    """
    Process user-uploaded node color file.
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        File containing node names and colors
    full_names : list
        List of full node names in the network
    
    Returns:
    --------
    dict : {node_name: RGB array} or None if error
    """
    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.tsv') or uploaded_file.name.endswith('.txt'):
            df = pd.read_csv(uploaded_file, sep='\t') 
        else:
            df = pd.read_excel(uploaded_file)
        
        # Validate columns
        if df.shape[1] < 2:
            st.error("❌ Node color file must have at least 2 columns: Node Name, Color")
            return None
        
        # Build color mapping
        color_map = {}
        unmatched = []
        invalid_colors = []
        
        for idx, row in df.iterrows():
            node_name = str(row.iloc[0]).strip()
            color_str = str(row.iloc[1]).strip()
            
            # Parse color
            rgb = parse_color_value(color_str)
            
            if rgb is not None:
                # Check if node exists in network
                if node_name in full_names:
                    color_map[node_name] = rgb
                else:
                    unmatched.append(node_name)
            else:
                invalid_colors.append((node_name, color_str))
        
        # Show warnings
        if unmatched:
            st.warning(f"⚠️ {len(unmatched)} nodes from color file not found in network (first 5): {unmatched[:5]}")
        
        if invalid_colors:
            st.warning(f"⚠️ {len(invalid_colors)} invalid color formats detected (first 5): {invalid_colors[:5]}")
        
        if not color_map:
            st.error("❌ No valid node colors found. Check file format.")
            return None
        
        st.success(f"✅ Loaded custom colors for {len(color_map)} nodes")
        return color_map
    
    except Exception as e:
        st.error(f"Error processing node color file: {e}")
        return None


def process_custom_edge_colors(uploaded_file, full_names):
    """
    Process user-uploaded edge color file.
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        File containing edge list with colors (node1, node2, color)
    full_names : list
        List of full node names in the network
    
    Returns:
    --------
    dict : {(node1_idx, node2_idx): color_string} or None if error
    """
    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.tsv') or uploaded_file.name.endswith('.txt'):
            df = pd.read_csv(uploaded_file, sep='\t') 
        else:
            df = pd.read_excel(uploaded_file)
        
        # Validate columns
        if df.shape[1] < 3:
            st.error("❌ Edge color file must have 3 columns: Node1, Node2, Color")
            return None
        
        # Build name to index mapping
        name_to_idx = {name: idx for idx, name in enumerate(full_names)}
        
        # Build edge color mapping
        edge_color_map = {}
        unmatched = []
        invalid_colors = []
        
        for idx, row in df.iterrows():
            node1 = str(row.iloc[0]).strip()
            node2 = str(row.iloc[1]).strip()
            color_str = str(row.iloc[2]).strip()
            
            # Check if both nodes exist
            if node1 not in name_to_idx or node2 not in name_to_idx:
                unmatched.append((node1, node2))
                continue
            
            # Parse color
            rgb = parse_color_value(color_str)
            
            if rgb is not None:
                # Get indices
                idx1 = name_to_idx[node1]
                idx2 = name_to_idx[node2]
                
                # Use canonical ordering (min, max)
                edge_key = (min(idx1, idx2), max(idx1, idx2))
                
                # Convert to color string for plotly
                color_hex = f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'
                edge_color_map[edge_key] = color_hex
            else:
                invalid_colors.append((node1, node2, color_str))
        
        # Show warnings
        if unmatched:
            st.warning(f"⚠️ {len(unmatched)} edges not found in network (first 5): {unmatched[:5]}")
        
        if invalid_colors:
            st.warning(f"⚠️ {len(invalid_colors)} invalid edge colors (first 5): {invalid_colors[:5]}")
        
        if not edge_color_map:
            st.error("❌ No valid edge colors found. Check file format.")
            return None
        
        st.success(f"✅ Loaded custom colors for {len(edge_color_map)} edges")
        return edge_color_map
    
    except Exception as e:
        st.error(f"Error processing edge color file: {e}")
        return None


def apply_custom_preference_coloring(x_, coords_native, node_shape, custom_node_colors, full_names):
    """
    Apply custom preference coloring to nodes.
    
    Parameters:
    -----------
    x_ : sparse matrix
        Adjacency matrix
    coords_native : ndarray
        Polar hyperbolic coordinates
    node_shape : list
        Node shapes ('o' for circle, 'd' for diamond)
    custom_node_colors : dict
        {node_name: RGB array}
    full_names : list
        Full node names
    
    Returns:
    --------
    ndarray : Node colors (N x 3 RGB array)
    """
    n_nodes = x_.shape[0]
    node_colors = np.zeros((n_nodes, 3))
    
    # Default gray for nodes without custom colors
    default_color = np.array([0.5, 0.5, 0.5])
    
    diamond_mask = np.array([s == 'd' for s in node_shape])
    circle_mask = np.array([s == 'o' for s in node_shape])
    
    colored_count = 0
    
    # Apply custom colors to nodes
    for i in range(n_nodes):
        node_name = full_names[i]
        
        if node_name in custom_node_colors:
            node_colors[i] = custom_node_colors[node_name]
            colored_count += 1
        else:
            node_colors[i] = default_color
    
    st.info(f"🎨 Applied custom colors to {colored_count}/{n_nodes} nodes. Remaining nodes are gray.")
    
    return node_colors


def __sph2cart(az, el, r):
    # Transform spherical to Cartesian coordinates
    r_cos_theta = r * np.cos(el)
    x_ = r_cos_theta * np.cos(az)
    y_ = r_cos_theta * np.sin(az)
    z_ = r * np.sin(el)
    return x_, y_, z_


def __cart2sph(x_, y_, z_):
    # Transform Cartesian to spherical coordinates
    hxy = np.hypot(x_, y_)
    r = np.hypot(hxy, z_)
    el = np.arctan2(z_, hxy)
    az = np.arctan2(y_, x_)
    return az, el, r


def __cart2pol(x_, y_):
    # Transform Cartesian to polar coordinates
    theta = np.arctan2(y_, x_)
    rho = np.hypot(x_, y_)
    return theta, rho


def __pol2cart(theta, rho):
    # Transform polar to Cartesian coordinate
    x_ = rho * np.cos(theta)
    y_ = rho * np.sin(theta)
    return x_, y_


def __equidistant_adjustment(coords):
    # Sort input coordinates
    idx = np.argsort(coords, kind='mergesort')

    # Assign equidistant angular coordinates in [0, 2pi) according to the sorting
    angles = np.linspace(0, 2 * np.pi, len(coords) + 1)[:-1]
    ang_coords = angles.copy()
    ang_coords[idx] = angles

    return ang_coords


def __kernel_centering(d_):
    # Centering
    n_ = d_.shape[0]
    j_ = np.eye(n_) - (1 / n_) * np.ones((n_, n_))
    d_ = -0.5 * (j_ @ (d_ ** 2) @ j_)

    # Housekeeping
    d_[np.isnan(d_)] = 0
    d_[np.isinf(d_)] = 0

    return d_


def get_svd_pool():
    global _SVD_POOL
    if _SVD_POOL is None:
        _SVD_POOL = ThreadPoolExecutor(max_workers=3)
        atexit.register(_SVD_POOL.shutdown, wait=False)

    return _SVD_POOL


def _svd_worker_partial(kernel, k):
    # Set seed for reproducibility 
    np.random.seed(42)
    # Convert to sparse if dense (PROPACK works better with sparse)
    if not issparse(kernel):
        # Only convert if reasonably sparse
        sparsity  = np.count_nonzero(kernel) / kernel.size
        if sparsity < 0.5:
            kernel = csr_matrix(kernel)

    # svds returns singular values in ascending order
    try:
        # Create deterministic initial vector
        v0 = np.ones(kernel.shape[0])
        u, s, vt = svds(kernel, k=k, which="LM", solver='propack', v0=v0)
    except Exception:
        v0 = np.ones(kernel.shape[0])
        u, s, vt = svds(kernel, k=k, which="LM", solver='arpack', v0=v0)
    # Sort descending (important!)
    idx = np.argsort(s)[::-1]
    s = s[idx]
    u = u[:, idx]

    return u, s


def _chunked_shortest_path(x_, chunk_size=100, heartbeat=None):
    """
    Compute all-pairs shortest path in chunks, sending heartbeats between chunks.
    
    This avoids the need for a separate process since we regain control
    between chunks.
    """
    n = x_.shape[0]
    
    # Pre-allocate result matrix
    dist_matrix = np.empty((n, n), dtype=np.float64)
    
    # Process in chunks
    num_chunks = (n + chunk_size - 1) // chunk_size  # Ceiling division
    
    for i, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        indices = np.arange(start, end)
        
        # Compute shortest paths from this chunk of source nodes
        chunk_result = csgraph.shortest_path(
            x_, 
            directed=False, 
            indices=indices
        )
        
        # Store results
        dist_matrix[start:end, :] = chunk_result
        
        # Send heartbeat between chunks
        if heartbeat:
            heartbeat("shortest_path", i + 1, num_chunks)
    
    return dist_matrix


def __isomap_graph_carlo(x_, n_, centring, heartbeat=None):
    # Initialization
    x_ = x_.maximum(x_.T)

    # Iso-kernel computation
    print('start SP')
    # ----------------------------- 
    # Chunked shortest path (no subprocess needed)
    # ----------------------------- 
    kernel = _chunked_shortest_path(x_, chunk_size=100, heartbeat=heartbeat)
    print('end SP')
    # Kernel centering
    if centring == 'yes':
        kernel = __kernel_centering(kernel)

    # Ensure float64 (required by svds)
    kernel = np.asarray(kernel, dtype=np.float64)

    # -------------------------------------------------
    # Partial SVD (Lanczos / PROPACK-style)
    # -------------------------------------------------
    k = n_  # 3 for 2D embedding

    # ----------------------------- 
    # SVD in a separate process 
    # ----------------------------- 

    # Use persistent pool instead of creating new one
    pool = get_svd_pool()
    future = pool.submit(_svd_worker_partial, kernel, k)

    while not future.done():
        if heartbeat:
            heartbeat("svd", None, None)
        time.sleep(0.3)

    v_, l_ = future.result()

    v_[:, 1] = v_[:, 1] * -1
    v_[:, 2] = v_[:, 2] * -1

    sqrt_l = np.sqrt(l_[:n_])
    v_ = v_[:, :n_]
    s = np.real(v_ * sqrt_l)

    return s


def __set_radial_coordinates(x_):
    n_ = x_.shape[0]
    deg = np.array((x_ > 0).sum(axis=1)).ravel()

    if np.all(deg == deg[0]):
        raise ValueError('All the nodes have the same degree, the degree distribution cannot fit a power-law.')

    # Fit power-law degree distribution
    # NOTE: MATLAB vs Python results are different!
    gamma_range = {'alpha': (1.01, 10.00)}
    small_size_limit = 100
    if len(deg) < small_size_limit:
        fit = powerlaw.Fit(data=deg, discrete=True, parameter_range=gamma_range, verbose=False)
    else:
        fit = powerlaw.Fit(data=deg, parameter_range=gamma_range, verbose=False)
    # DIFF:
    #   dataset: data_final
    #   - MATLAB: 2.5867
    #   - Python: 2.523668642131128
    #   - Delta (difference): 0.06303135786887193
    #   dataset: data_final2
    #   - MATLAB: 2.4589
    #   - Python: 2.400016678698991
    #   - Delta (difference): 0.058883321301008706
    gamma = fit.alpha
    beta = 1 / (gamma - 1)

    # Sort nodes by decreasing degree
    idx = np.argsort(-deg, kind='mergesort')

    # For beta > 1 (gamma < 2), some radial coordinates are negative
    radial_coordinates = np.zeros(n_)
    radial_coordinates[idx] = np.maximum(0, 2 * beta * np.log1p(np.arange(0, n_)) + 2 * (1 - beta) * np.log(n_))

    return radial_coordinates


def __set_angular_coordinates_ncISO_2D(xw, angular_adjustment, heartbeat=None):
    # dimension reduction
    dr_coords = __isomap_graph_carlo(xw, 3, 'no', heartbeat=heartbeat)
    print('SVD over')

    # from cartesian to polar coordinates
    # using dimensions 2 and 3 of embedding
    ang_coords, _ = __cart2pol(dr_coords[:, 1], dr_coords[:, 2])
    # change angular range from [-pi,pi] to [0,2pi]
    ang_coords = np.mod(ang_coords + 2 * np.pi, 2 * np.pi)

    if angular_adjustment == 'EA':
        ang_coords = __equidistant_adjustment(ang_coords)

    return ang_coords


def __ra1_weighting(x_):
    n_ = x_.shape[0]
    cn = x_ @ x_.T
    deg = np.array(x_.sum(axis=1)).ravel()   # 1D degree vector
    deg_matrix = np.outer(deg, np.ones(n_))  # (n_, n_)
    weights = deg_matrix + deg_matrix.T + (deg_matrix * deg_matrix.T)
    weights_sparse = csr_matrix(weights)
    x_ra1 = x_.multiply(weights_sparse) / (1 + cn.toarray())
    
    return x_ra1


def ra1_weighting_sparse(x: csr_matrix, heartbeat=None, heartbeat_every=1) -> csr_matrix:
    x = x.tocsr().copy()
    x.data[:] = 1  # binarize

    deg = np.asarray(x.sum(axis=1)).ravel()
    cn = (x @ x.T).tocsr()  # sparse common-neighbors

    coo = x.tocoo()
    i, j = coo.row, coo.col

    data = np.empty(len(i))
    total = len(i)

    for k in range(total):
        data[k] = (deg[i[k]] + deg[j[k]] + deg[i[k]] * deg[j[k]]) / (1.0 + cn[i[k], j[k]])
        if heartbeat and (k+1) % heartbeat_every == 0:
            heartbeat("ra1", k+1, total)

    # final update
    if heartbeat:
        heartbeat("ra1", total, total)

    return csr_matrix((data, (i, j)), shape=x.shape)


def __coal_embed(x_, pre_weighting, dim_red, angular_adjustment, dims, heartbeat=None):
    # TODO: add MATLAB validations

    xd = x_.copy()
    xd[x_ == 2] = 0
    x_[x_ == 2] = 1

    # pre-weighting
    #xw = __ra1_weighting((x_ > 0).astype(int))
    xw = ra1_weighting_sparse((x_ > 0).astype(int), heartbeat=heartbeat, heartbeat_every=1)
    print('RA1 over')

    # dimension reduction and set of hyperbolic coordinates
    if dims == 2:
        coords = np.zeros((x_.shape[0], 2))
        coords[:, 0] = __set_angular_coordinates_ncISO_2D(xw, angular_adjustment, heartbeat=heartbeat)
        coords[:, 1] = __set_radial_coordinates(xd)
    elif dims == 3:
        # TODO: 3D handling
        raise NotImplementedError("coal_embed 3D not implemented")

    print('coal embed')

    return coords


def __calc_slope(point1, point2):
    # Equation to calculate slope from 2 points
    # point1 and point2 need to be in form [x, y]

    dy = point2[1] - point1[1]
    dx = point2[0] - point1[0]

    if dx == 0:
        raise ValueError('Vertical line, no proper slope defined')
    elif dy == 0:
        raise ValueError('Horizontal line, slope 0')
    else:
        m = dy / dx

    return m


def __line_eq(point1, point2, x_):
    # Line equation from 2 points
    # point1 and point2 need to be in form [x, y]

    if (point2[0] - point1[0]) == 0:
        raise ValueError('Vertical line')
    else:
        try:
            y = __calc_slope(point1, point2) * x_ + (point2[0] * point1[1] - point1[0] * point2[1]) / (
                    point2[0] - point1[0])
        except ValueError:
            y = np.repeat((point2[0] * point1[1] - point1[0] * point2[1]) / (point2[0] - point1[0]), len(x_))

    return y


def __ortho_circle(A, B, r, C):
    A[0] = A[0] - C[0]
    A[1] = A[1] - C[1]
    B[0] = B[0] - C[0]
    B[1] = B[1] - C[1]

    # Calculate center Co of orthogonal circle
    tmp_A = A[0] ** 2 + A[1] ** 2 + r ** 2
    tmp_B = B[0] ** 2 + B[1] ** 2 + r ** 2
    Co_y = (0.5 * tmp_B - B[0] * tmp_A / (2 * A[0])) / (B[1] - B[0] * A[1] / A[0])
    Co_x = (tmp_A - 2 * A[1] * Co_y) / (2 * A[0])

    # Calculate radius ro of orthogonal circle
    ro = np.sqrt(Co_x ** 2 + Co_y ** 2 - r ** 2)

    # Add origin to center of orthogonal circle to shift back the complete construction of circles
    Co = [Co_x + C[0], Co_y + C[1]]

    return Co, ro


def __get_arc(A, B, C, r, which, val):
    # Check if given points lie on circle
    tmp1 = (A[0] - C[0]) ** 2 + (A[1] - C[1]) ** 2 - r ** 2
    tmp2 = (B[0] - C[0]) ** 2 + (B[1] - C[1]) ** 2 - r ** 2

    # Calculate angles of the given points on the given circle
    phi_A = __quadrant_phi(A[0], A[1], C[0], C[1], r)
    phi_B = __quadrant_phi(B[0], B[1], C[0], C[1], r)

    # Calculate shorter arc between the points on the circle
    if abs(phi_A - phi_B) >= np.pi:
        tmp1 = 2 * np.pi - max(phi_A, phi_B) + min(phi_A, phi_B)
        if which == 'step':
            tmp2 = np.linspace(max(phi_A, phi_B), max(phi_A, phi_B) + tmp1, round(tmp1 / (val / r)) + 1)
        elif which == 'n':
            tmp2 = np.linspace(max(phi_A, phi_B), max(phi_A, phi_B) + tmp1, val)
        phi_AB = np.mod(tmp2, 2 * np.pi)
    else:
        if which == 'step':
            phi_AB = np.linspace(min(phi_A, phi_B), max(phi_A, phi_B),
                                 round((max(phi_A, phi_B) - min(phi_A, phi_B)) / (val / r)) + 1)
        elif which == 'n':
            phi_AB = np.linspace(min(phi_A, phi_B), max(phi_A, phi_B), val)

    # Turn to cartesian coordinates
    x_arc = r * np.cos(phi_AB)  # make circle as if it were having its center in the origin
    y_arc = r * np.sin(phi_AB)
    x_arc = x_arc + C[0]  # move center
    y_arc = y_arc + C[1]

    return x_arc, y_arc


def __quadrant_phi(Px, Py, Cx, Cy, r):
    if Py >= Cy and Px >= Cx:  # 1st quadrant
        phi = np.arccos((Px - Cx) / r)
    elif Py >= Cy and Px < Cx:  # 2nd quadrant
        phi = np.arccos((Px - Cx) / r)
    elif Py < Cy and Px <= Cx:  # 3rd quadrant
        phi = 2 * np.pi - np.arccos((Px - Cx) / r)
    elif Py < Cy and Px > Cx:  # 4th quadrant
        phi = 2 * np.pi - np.arccos((Px - Cx) / r)

    return phi


def __hyperbolic_line_lipea(A, B, r, M, which, val):
    # Check if points A and B are lying within the disc
    if (np.sqrt((M[0] - A[0]) ** 2 + (M[1] - A[1]) ** 2) > r + 1e-15) or (
            np.sqrt((M[0] - B[0]) ** 2 + (M[1] - B[1]) ** 2) > r + 1e-15):
        # TODO: transform into warning
        print("At least one of the points is lying outside the disc")

    # Make sure that given points A and B are different
    if (A[0] == B[0]) and (A[1] == B[1]):
        raise ValueError("Please enter two different points")

    # Adjust points that are on the zero coordinates
    if abs(A[0]) < 0.0001:
        A[0] = np.sign(A[0]) * 0.0001
    if abs(A[1]) < 0.0001:
        A[1] = np.sign(A[1]) * 0.0001
    if abs(B[0]) < 0.0001:
        B[0] = np.sign(B[0]) * 0.0001
    if abs(B[1]) < 0.0001:
        B[1] = np.sign(B[1]) * 0.0001

    # Check if A and B are on a line through the center
    if abs(A[0] - B[0]) == 0:  # Are A and B on a vertical line through center?
        tmp = abs(A[0] - M[0]) < 1e-12
    elif abs(A[1] - B[1]) == 0:  # Are A and B on a horizontal line through center?
        tmp = abs(A[1] - M[1]) < 1e-12
    else:
        test = abs(M[1] - __line_eq(A, B, M[0]))  # Catches all lines through center besides vertical/horizontal ones
        tmp = test < 1e-12

    # If A and B are on a line through the center, make a straight line and set M_o and r_o to NaN
    if tmp:
        if which == 'n':
            x_arc = np.linspace(A[0], B[0], val)
            y_arc = __line_eq(A, B, x_arc)
        elif which == 'step':
            x_arc = np.linspace(min(A[0], B[0]), max(B[0], A[0]), round((max(B[0], A[0]) - min(A[0], B[0])) / 0.01) + 1)
            y_arc = __line_eq(A, B, x_arc)
        else:
            raise ValueError("which has to be 'n' or 'step'")

        C_o = [np.nan, np.nan]
        r_o = np.nan

    # In all other cases, make the hyperbolic line
    else:
        # Check if there will be a solution for the equations used below
        check = A[0] * B[1] - B[0] * A[1]
        if check == 0:
            raise ValueError("No solution available")

        # Calculate center M_o and radius r_o of the orthogonal circle
        C_o, r_o = __ortho_circle(A, B, r, M)

        # Get cartesian coordinates of the arc between the points A and B on orthogonal circle
        x_arc, y_arc = __get_arc(A, B, C_o, r_o, which, val)

    return C_o, r_o, x_arc, y_arc


def __compute_plot_coords(coords_native):
    # Transform coordinates
    coords_unit = coords_native.copy()
    coords_unit[:, 1] = np.tanh(coords_native[:, 1] / 2)

    cart_coords_unit = np.zeros_like(coords_unit)
    cart_coords_unit[:, 0], cart_coords_unit[:, 1] = __pol2cart(
        coords_unit[:, 0], coords_unit[:, 1]
    )

    return cart_coords_unit


def colormap_blue_to_red(n):
    """Generate a blue-to-red colormap for hierarchy coloring."""
    colors = np.zeros((n, 3))
    m = np.round(np.linspace(0, n-1, 4)).astype(int)
    
    # Blue to cyan
    colors[0:m[1]+1, 1] = np.linspace(0, 1, m[1]+1)
    colors[0:m[1]+1, 2] = 1
    
    # Cyan to yellow
    colors[m[1]:m[2]+1, 0] = np.linspace(0, 1, m[2]-m[1]+1)
    colors[m[1]:m[2]+1, 1] = 1
    colors[m[1]:m[2]+1, 2] = np.linspace(1, 0, m[2]-m[1]+1)
    
    # Yellow to red
    colors[m[2]:n, 0] = 1
    colors[m[2]:n, 1] = np.linspace(1, 0, n-m[2])
    
    return colors


def apply_gradient_coloring(x_, coords_native, node_shape, coloring='hierarchy',
                            labels=None, custom_node_colors=None, full_names=None):
    """
    Apply gradient coloring scheme to nodes.
    
    Parameters:
    -----------
    custom_node_colors : dict, optional
        Custom color mapping {node_name: RGB array}
    full_names : list, optional
        Full node names (required for preference mode)
    """
    n_nodes = x_.shape[0]
    diamond_mask = np.array([s == 'd' for s in node_shape])
    circle_mask = np.array([s == 'o' for s in node_shape])
    node_colors = np.zeros((n_nodes, 3))
    
    # Color DIAMOND nodes based on coloring scheme
    diamond_indices = np.where(diamond_mask)[0]

    # Add preference mode before existing modes
    if coloring == 'preference':
        if custom_node_colors is not None and full_names is not None:
            return apply_custom_preference_coloring(
                x_, coords_native, node_shape, custom_node_colors, full_names
            )
        else:
            st.warning("⚠️ No custom colors provided. Using default gray coloring.")
            node_colors[:] = np.array([0.5, 0.5, 0.5])
            return node_colors
    
    if coloring == 'hierarchy':
        # Color by degree (hierarchy)
        deg = np.array((x_ > 0).sum(axis=1)).ravel()
        deg_diamonds = deg[diamond_mask]
        
        if len(deg_diamonds) > 0 and np.max(deg_diamonds) != np.min(deg_diamonds):
            deg_normalized = np.round(
                (np.max(deg_diamonds) - 1) * (deg_diamonds - np.min(deg_diamonds)) / 
                (np.max(deg_diamonds) - np.min(deg_diamonds)) + 1
            ).astype(int)
        else:
            deg_normalized = np.ones(len(deg_diamonds), dtype=int)
        
        colors = colormap_blue_to_red(int(np.max(deg_normalized)))
        node_colors[diamond_mask] = colors[deg_normalized - 1, :]
        
    elif coloring == 'similarity':
        # Color by angular coordinate (similarity)
        angles_diamonds = coords_native[diamond_mask, 0]
        angles_normalized = angles_diamonds / (2 * np.pi)
        
        # Create HSV colors
        hsv_colors = np.zeros((len(angles_normalized), 3))
        hsv_colors[:, 0] = angles_normalized
        hsv_colors[:, 1] = 1.0
        hsv_colors[:, 2] = 1.0
        hsv_colors[:, 0] = (hsv_colors[:, 0] + 0.5) % 1.0
        
        node_colors[diamond_mask] = hsv_to_rgb(hsv_colors)
        
    elif coloring == 'labels' and labels is not None: 
        # Validate labels length
        if len(labels) != n_nodes:
            print(f"Warning: labels length ({len(labels)}) doesn't match nodes ({n_nodes})")
            # Fallback to gray
            node_colors[diamond_mask] = np.array([0.5, 0.5, 0.5])
        else:         
            # Color DIAMOND nodes by labels
            labels = np.array(labels)
            labels_diamonds = labels[diamond_mask]
  
            # Filter out "Unknown" labels
            unique_labels = [lab for lab in np.unique(labels_diamonds) 
                           if str(lab).lower() not in ["unknown", "nan", "none", ""]]

            if len(unique_labels) == 0:
                # All labels are unknown, use gray
                node_colors[diamond_mask] = np.array([0.5, 0.5, 0.5])
            else:
                # Create label to color mapping with distinct colors
                label_mapping = {lab: i for i, lab in enumerate(unique_labels)}
  
                # Generate MAXIMALLY distinct HSV colors for labels
                n_colors = len(unique_labels)
                hsv_colors = np.zeros((n_colors, 3))
                # Use golden ratio for maximum perceptual distinction
                golden_ratio = 0.618033988749895
                hsv_colors[:, 0] = np.mod(np.arange(n_colors) * golden_ratio, 1.0)
                hsv_colors[:, 1] = 0.85  # High saturation for vibrant colors
                hsv_colors[:, 2] = 0.9   # Slightly dimmed for better visibility
                colors = hsv_to_rgb(hsv_colors)

                # Apply colors to diamond nodes
                for enum_idx, global_idx in enumerate(diamond_indices):
                    label = labels_diamonds[enum_idx]
                    label_str = str(label).lower()

                    if label_str not in ["unknown", "nan", "none", ""] and label in label_mapping:
                        color_idx = label_mapping[label]
                        node_colors[global_idx] = colors[color_idx, :]
                    else:
                        node_colors[global_idx] = np.array([0.5, 0.5, 0.5])  # Gray for unknown
    
    # Color CIRCLE nodes - SAME for all modes: average of connected diamond neighbors
    circle_indices = np.where(circle_mask)[0]
    for circle_idx in circle_indices:
        neighbors = x_[circle_idx, :].nonzero()[1]
        diamond_neighbors = [n for n in neighbors if diamond_mask[n]]
        
        if len(diamond_neighbors) > 0:
            neighbor_colors = node_colors[diamond_neighbors, :]
            # Filter out gray (unknown) colors when computing average
            non_gray = neighbor_colors[~np.all(np.isclose(neighbor_colors, [0.5, 0.5, 0.5]), axis=1)]

            if len(non_gray) > 0:
                # Simple RGB average (color mixing)
                node_colors[circle_idx] = np.mean(non_gray, axis=0)
            else:
                node_colors[circle_idx] = np.array([0.5, 0.5, 0.5])
        else:
            node_colors[circle_idx] = np.array([0.5, 0.5, 0.5])
    
    return node_colors


# ------------------------------------------------------------------
# Helpers for dynamic plot title + legend (main plot + subnetwork)
# ------------------------------------------------------------------
def _coloring_display_name(coloring_scheme: str) -> str:
    """Human-friendly name for the UI title."""
    if coloring_scheme is None:
        return "Default"
    s = str(coloring_scheme).strip()
    if not s:
        return "Default"
    s_norm = s.lower().replace("_", " ").replace("-", " ").strip()
    if s_norm == "pathway significance":
        return "Pathway Significance"
    return s_norm.title()

def _hyperpathway_plot_title(coloring_scheme: str) -> str:
    return f"Hyperpathway visualization ({_coloring_display_name(coloring_scheme)} coloring)"

def _build_hyperpathway_legend_traces(
    *,
    option: int,
    coloring_scheme: str,
    omics_type: str,
    col_non_corr=None,
    col_corr_1=None,
    col_corr_2=None,
    corr_1_name=None,
    corr_2_name=None,
):
    """Build legend traces consistent with the requested rules."""
    if option != 1:
        return []

    cs_norm = (coloring_scheme or "").strip()
    legend_traces = []

    # Non-pathway-significance modes: empty circle (omics) + empty diamond (Pathway)
    if cs_norm != "Pathway Significance":
        legend_traces.append(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(
                symbol='circle',
                size=12,
                color='rgba(0,0,0,0)',
                line=dict(color='black', width=2)
            ),
            name=str(omics_type),
            showlegend=True
        ))
        legend_traces.append(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(
                symbol='diamond',
                size=12,
                color='rgba(0,0,0,0)',
                line=dict(color='black', width=2)
            ),
            name='Pathway',
            showlegend=True
        ))
        return legend_traces

    # Pathway Significance mode: filled green circle (omics) + diamonds for p-values
    legend_traces.append(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(
            symbol='circle',
            size=12,
            color='green',
            line=dict(color='green', width=1)
        ),
        name=str(omics_type),
        showlegend=True
    ))

    # Non-corrected p-value (optional): gray diamond
    if col_non_corr is not None:
        legend_traces.append(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(
                symbol='diamond',
                size=12,
                color='gray',
                line=dict(color='gray', width=1)
            ),
            name=str(col_non_corr),
            showlegend=True
        ))

    # Corrected p-values: red diamond (corr_1) + orange diamond (corr_2, if present)
    if col_corr_1 is not None:
        legend_traces.append(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(
                symbol='diamond',
                size=12,
                color='red',
                line=dict(color='red', width=1)
            ),
            name=str(corr_1_name or col_corr_1),
            showlegend=True
        ))
    if col_corr_2 is not None:
        legend_traces.append(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(
                symbol='diamond',
                size=12,
                color='orange',
                line=dict(color='orange', width=1)
            ),
            name=str(corr_2_name or col_corr_2),
            showlegend=True
        ))

    return legend_traces


def __plot_hyperlipea_interactive(x_, coords_native, node_colors, names, node_shape, option, omics,
                                e_colors=None, build_edges=True, max_edges=20000,
                                corr_1_name='Correction #1', corr_2_name='Correction #2',
                                coloring_mode='default', show_labels=False, edge_opacity=0.6,
                                coloring_scheme='similarity'):
    # Safety check: if node_colors is None, generate appropriate default colors
    if node_colors is None:
        # If edge colors exist, use black for diamonds and red for circles (matching default mode behavior)
        if e_colors is not None and isinstance(e_colors, dict) and len(e_colors) > 0:
            node_colors = np.zeros((len(node_shape), 3))
            for i, shape in enumerate(node_shape):
                if shape == 'd':  # diamond
                    node_colors[i] = [0, 0, 0]  # black
                else:  # circle
                    node_colors[i] = [1, 0, 0]  # red
        else:
            # Otherwise use gradient coloring
            from scipy.sparse import issparse
            if issparse(x_):
                x_temp = x_.toarray()
            else:
                x_temp = x_
            node_colors = apply_gradient_coloring(
                x_, coords_native, node_shape,
                coloring='similarity'
            )
    
    # Transform coordinates
    cart_coords_native = np.zeros_like(coords_native)
    cart_coords_native[:, 0], cart_coords_native[:, 1] = __pol2cart(coords_native[:, 0], coords_native[:, 1])
    cart_coords_unit = __compute_plot_coords(coords_native)

    # Radius of native coordinates (for circle)
    radius = np.max(coords_native[:, 1])
    circle_theta = np.linspace(0, 2 * np.pi, 500)
    circle_x = radius * np.cos(circle_theta)
    circle_y = radius * np.sin(circle_theta)
    circle_trace = go.Scatter(
        x=circle_x,
        y=circle_y,
        mode='lines',
        line=dict(color='rgba(200,200,200,0.5)', width=1),
        hoverinfo='skip',
        showlegend=False
    )

    # Edges - Modified to handle gradient coloring
    edge_traces = []
    if build_edges: 
        e1, e2 = triu(x_, k=1).nonzero()
        if len(e1) > max_edges:
            # optionally subsample
            idx = np.random.choice(len(e1), size=max_edges, replace=False)
            e1, e2 = e1[idx], e2[idx]

        # Group edges by color for efficient rendering
        # e_colors should be a dict: (i, j) -> color_string or similar structure
        edges_by_color = {}
        default_color = '#CCCCCC'
        
        for edge_idx, (e_start, e_end) in enumerate(zip(e1, e2)):
            start_coords = cart_coords_unit[e_start]
            end_coords = cart_coords_unit[e_end]
            if np.sum(start_coords - end_coords) != 0:
                _, _, cart_arc_1, cart_arc_2 = __hyperbolic_line_lipea(start_coords.copy(), end_coords.copy(), 1, [0, 0], 'step', 0.05)
                pol_arc = np.empty((len(cart_arc_1), 2))
                pol_arc[:] = np.nan
                pol_arc[:, 0], pol_arc[:, 1] = __cart2pol(cart_arc_1, cart_arc_2)
                pol_arc[:, 1] = 2 * np.arctanh(pol_arc[:, 1])
                cart_arc_1, cart_arc_2 = __pol2cart(pol_arc[:, 0], pol_arc[:, 1])
                
                # FIXED: Edge coloring logic
                edge_color = default_color
                # PRIORITY 1: Check for explicit edge colors first
                if e_colors is not None and isinstance(e_colors, dict):
                    # Try canonical ordering (min, max) - used by updated process_adjacency_list
                    canonical_key = (min(e_start, e_end), max(e_start, e_end))
                    if canonical_key in e_colors:
                        edge_color = e_colors[canonical_key]
                    # Fallback: Try (i, j) tuple as-is
                    elif (e_start, e_end) in e_colors:
                        edge_color = e_colors[(e_start, e_end)]
                    elif (e_end, e_start) in e_colors:
                        edge_color = e_colors[(e_end, e_start)]
                # PRIORITY 2: Only use gradient coloring if NO explicit edge colors AND gradient mode
                elif coloring_mode == 'gradient':
                    # PRIORITY 1: Color edge by DIAMOND (pathway) node
                    pathway_node = None

                    if node_shape[e_start] == 'd':
                        pathway_node = e_start
                    elif node_shape[e_end] == 'd':
                        pathway_node = e_end
                    
                    if pathway_node is not None:
                        rgb = node_colors[pathway_node]
                        edge_color = f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'
                    else:
                        # Fallback: blend colors if both are circles (rare case)
                        rgb = (node_colors[e_start] + node_colors[e_end]) / 2
                        edge_color = f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'

                elif e_colors is not None:
                    # Use provided edge colors (existing behavior)
                    if isinstance(e_colors, dict):
                        # Try canonical ordering (min, max) - used by updated process_adjacency_list
                        canonical_key = (min(e_start, e_end), max(e_start, e_end))
                        if canonical_key in e_colors:
                            edge_color = e_colors[canonical_key]
                        # Fallback: Try (i, j) tuple as-is
                        elif (e_start, e_end) in e_colors:
                            edge_color = e_colors[(e_start, e_end)]
                        elif (e_end, e_start) in e_colors:
                            edge_color = e_colors[(e_end, e_start)]

                # Group by color
                if edge_color not in edges_by_color:
                    edges_by_color[edge_color] = {'x': [], 'y': []}
                edges_by_color[edge_color]['x'] += list(cart_arc_1) + [None]
                edges_by_color[edge_color]['y'] += list(cart_arc_2) + [None]
        
        # Create one trace per color group
        for color, coords in edges_by_color.items():
            edge_traces.append(go.Scatter(
                x=coords['x'],
                y=coords['y'],
                mode='lines',
                line=dict(width=1, color=color),
                opacity=edge_opacity,
                hoverinfo='none',
                showlegend=False
            ))

    # Nodes - Convert node_colors to RGB strings for Plotly
    node_x = cart_coords_native[:, 0]
    node_y = cart_coords_native[:, 1]
    
    # Calculate degree for each node
    degrees = np.array(x_.sum(axis=1)).ravel()
    
    # Improved degree-based sizing with better visual separation
    # Use power scaling to make high-degree nodes much more prominent
    # Min size 6, then scale by degree^0.7 for better visual distinction
    min_size = 6
    max_degree = degrees.max() if degrees.max() > 0 else 1
    
    # Normalize degrees to 0-1, then apply power for non-linear scaling
    degree_normalized = degrees / max_degree
    # Use power 0.6 to amplify differences (lower = more dramatic)
    degree_scaled = np.power(degree_normalized, 0.6)
    
    # Scale to size range: min_size to min_size + 30
    node_size = min_size + (degree_scaled * 30)

    # Map node_shape values to Plotly-compatible symbols
    symbol_map = {'o': 'circle', 'd': 'diamond'}
    shape_groups = {'circle': [], 'diamond': []}
    x_groups = {'circle': [], 'diamond': []}
    y_groups = {'circle': [], 'diamond': []}
    text_groups = {'circle': [], 'diamond': []}
    hover_groups = {'circle': [], 'diamond': []}
    size_groups = {'circle': [], 'diamond': []}
    color_groups = {'circle': [], 'diamond': []}

    for i in range(len(node_shape)):
        shape = symbol_map[node_shape[i]]
        shape_groups[shape].append(shape)
        x_groups[shape].append(node_x[i])
        y_groups[shape].append(node_y[i])
        size_groups[shape].append(node_size[i])
        text_groups[shape].append(names[i])
        hover_groups[shape].append(names[i])

        color_val = node_colors[i]
        if isinstance(color_val, str):
            color_groups[shape].append(color_val)  # HEX color
        else:
            # Convert RGB array to string
            color_groups[shape].append(f'rgb({int(color_val[0]*255)},{int(color_val[1]*255)},{int(color_val[2]*255)})')

    node_traces = []
    for shape in shape_groups:
        # Set mode based on show_labels parameter
        mode = 'markers+text' if show_labels else 'markers'
        node_traces.append(go.Scatter(
            x=x_groups[shape],
            y=y_groups[shape],
            mode=mode,
            marker=dict(
                size=size_groups[shape],
                color=color_groups[shape],
                symbol=shape,
                line=dict(width=2, color='white')
            ),
            text=text_groups[shape],
            hovertext=hover_groups[shape],
            textposition='top center',
            selected=dict(marker=dict(size=20)),
            unselected=dict(marker=dict(opacity=0.5)),
            showlegend=False,
            # ⭐ Store original sizes for scaling
            customdata=[[s] for s in size_groups[shape]]
        ))

    # Explicit trace ordering ---
    # Edges MUST come before nodes to render underneath
    legend_traces = []
    all_traces = []
    all_traces.append(circle_trace)       # 1. Background boundary
    all_traces.extend(edge_traces)        # 2. ALL edges (bottom layer)
    all_traces.extend(legend_traces)      # 3. Legend items (middle)
    all_traces.extend(node_traces)        # 4. ALL nodes (TOP layer - renders LAST)

    # ===================================================================
    # LEGEND BUILDING - Works for ALL coloring modes
    # ===================================================================
    legend_traces = []
    
    if option == 1:
        # PATHWAY VISUALIZATION MODE - Always build legend
        
        plot_title = _hyperpathway_plot_title(coloring_scheme)

        # Determine which colors are actually present in node_colors
        rgb_map = {
            'red': np.array([1.0, 0.0, 0.0]),      # Correction 1
            'orange': np.array([1.0, 0.6, 0.0]),   # Correction 2  
            'grey': np.array([0.7, 0.7, 0.7]),     # Non-corrected
            'green': np.array([0.0, 1.0, 0.0]),    # Omics (molecules)
        }
        
        color_exists = {}
        for color_name, rgb_val in rgb_map.items():
            color_exists[color_name] = np.any(np.all(np.isclose(node_colors, rgb_val, atol=0.01), axis=1))
        
        # Determine coloring mode for title
        if coloring_mode == 'gradient':
            # Non-pathway-significance modes: use the unified hollow legend so main plot and subnetworks match
            legend_traces = _build_hyperpathway_legend_traces(
                option=option,
                coloring_scheme=coloring_scheme,
                omics_type=omics
            )
        else:
            # PATHWAY SIGNIFICANCE MODE - Detailed legend with column names
                        
            # Get actual column names from session state
            col_corr_1_name = st.session_state.get('corr_1_display_name', 'Correction #1')
            col_corr_2_name = st.session_state.get('corr_2_display_name', 'Correction #2')
            col_non_corr_name = st.session_state.get('non_corr_display_name', 'Non-corrected p-value')
            
            # Check which p-value columns were actually selected
            has_corr_1 = st.session_state.get('col_corr_1_selected', False)
            has_corr_2 = st.session_state.get('col_corr_2_selected', False)
            has_non_corr = st.session_state.get('col_non_corr_selected', False)
            
            # Always show omics type (green circles) if present
            if color_exists['green']:
                legend_traces.append(
                    go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(symbol='circle', color='#00AA00', size=12, line=dict(width=2, color='white')),
                        name=omics,
                        showlegend=True
                    )
                )
            
            # CASE 1: Only ONE corrected p-value selected
            if (has_corr_1 and not has_corr_2) or (has_corr_2 and not has_corr_1):
                if color_exists['red']:
                    # Use whichever correction was actually selected
                    corr_name = col_corr_1_name if has_corr_1 else col_corr_2_name
                    legend_traces.append(
                        go.Scatter(
                            x=[None], y=[None],
                            mode='markers',
                            marker=dict(symbol='diamond', color='#FF0000', size=12, line=dict(width=2, color='white')),
                            name=f'{corr_name}',
                            showlegend=True
                        )
                    )
            
            # CASE 2: TWO corrected p-values selected
            elif has_corr_1 and has_corr_2:
                if color_exists['red']:
                    legend_traces.append(
                        go.Scatter(
                            x=[None], y=[None],
                            mode='markers',
                            marker=dict(symbol='diamond', color='#FF0000', size=12, line=dict(width=2, color='white')),
                            name=f'{col_corr_1_name}',
                            showlegend=True
                        )
                    )
                if color_exists['orange']:
                    legend_traces.append(
                        go.Scatter(
                            x=[None], y=[None],
                            mode='markers',
                            marker=dict(symbol='diamond', color='#FFA500', size=12, line=dict(width=2, color='white')),
                            name=f'{col_corr_2_name}',
                            showlegend=True
                        )
                    )
            
            # Non-corrected p-value (grey) - only if selected
            if has_non_corr and color_exists['grey']:
                legend_traces.append(
                    go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(symbol='diamond', color='#B2B2B2', size=12, line=dict(width=2, color='white')),
                        name=f'{col_non_corr_name}',
                        showlegend=True
                    )
                )
                        # Explicit trace ordering
        all_traces = []
        all_traces.append(circle_trace)       # 1. Background boundary
        all_traces.extend(edge_traces)        # 2. ALL edges (bottom layer)
        all_traces.extend(legend_traces)      # 3. Legend items (middle)
        all_traces.extend(node_traces)        # 4. ALL nodes (TOP layer)

        # Create figure for pathway mode
        fig = go.Figure(
            data=all_traces,
            layout=go.Layout(
                title=plot_title,
                showlegend=True,  # ✅ ALWAYS show legend for option 1
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=-0.05,
                    xanchor='center',
                    x=0.5
                ),
                hovermode='closest',
                margin=dict(b=60, l=20, r=20, t=40),  # ✅ Increased bottom margin for legend
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='y'),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x'),
                height=800
            )
        )
    else:
        # BIPARTITE NETWORK MODE
        if coloring_mode == 'gradient':
            plot_title = f"Bipartite network visualization (gradient coloring)"
        else:
            plot_title = f"Bipartite network visualization in hyperbolic space..."

        fig = go.Figure(
            data=[circle_trace] + edge_traces + node_traces,
            layout=go.Layout(
                title=plot_title,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=80, l=20, r=20, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='y'),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x'),
                height=800
            )
        )


    fig.update_layout(
        dragmode='select',
        selectdirection='any',
        modebar_add=['select2d', 'lasso2d'],
        clickmode='event+select',
    )

    return fig


def filter_subgraph(x, coords_native, node_colors, names, node_shape, selected_nodes, edge_colors=None):
    n_nodes = coords_native.shape[0]

    # Keep only valid selected indices
    selected_nodes = [i for i in selected_nodes if 0 <= i < n_nodes]

    # Partition sets based on wsymbol/node_shape
    o_nodes = {i for i, s in enumerate(node_shape) if s == "o"}
    d_nodes = {i for i, s in enumerate(node_shape) if s == "d"}

    neighbors = set()
    for node in selected_nodes:
        row = x.getrow(node) # sparse row
        nbrs = row.indices # nonzero column indices
        if node_shape[node] == "o":
            # expand only to "d" neighbors
            neighbors.update([j for j in nbrs if j in d_nodes])
        elif node_shape[node] == "d":
            # expand only to "o" neighbors
            neighbors.update([j for j in nbrs if j in o_nodes])

    # Union of selected + valid neighbors
    sub_nodes = sorted(set(selected_nodes) | neighbors)

    # Final guard
    sub_nodes = [i for i in sub_nodes if 0 <= i < n_nodes]

    # Create mapping from old indices to new indices
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sub_nodes)}

    # Slice adjacency and attributes
    mask = x[sub_nodes, :][:, sub_nodes]
    mask = mask.maximum(mask.T)
    coords_sub = coords_native[sub_nodes]
    colors_sub = [node_colors[i] for i in sub_nodes]
    names_sub = [names[i] for i in sub_nodes]
    shapes_sub = [node_shape[i] for i in sub_nodes]

    # Filter edge colors for the subnetwork
    edge_colors_sub = None
    if edge_colors is not None and isinstance(edge_colors, dict):
        edge_colors_sub = {}
        for old_i, old_j in edge_colors.keys():
            # Check if both endpoints are in the subgraph
            if old_i in old_to_new and old_j in old_to_new:
                new_i = old_to_new[old_i]
                new_j = old_to_new[old_j]
                # Use canonical ordering (min, max)
                new_key = (min(new_i, new_j), max(new_i, new_j))
                old_key = (min(old_i, old_j), max(old_i, old_j))
                edge_colors_sub[new_key] = edge_colors[old_key]

    return mask, coords_sub, colors_sub, names_sub, shapes_sub, edge_colors_sub


def remove_isolated_nodes(x_, w_type, w_color, w_name, w_symbol, w_full_name=None):
    
    """

    Remove isolated nodes and update all node attributes consistently.

    

    Parameters:

    -----------

    w_full_name : list, optional

        List of full node names (before truncation)

    """
    # Compute degree safely for sparse or dense
    mask = np.array(x_.sum(axis=1)).ravel() != 0   # True for non-isolated nodes

    # Apply mask to both rows and columns, keep sparse
    x_new = x_[np.ix_(mask, mask)].tocsr()

    # Apply mask to attributes
    w_type_new   = w_type[mask]
    w_color_new  = np.array(w_color)[mask]
    indices      = np.where(mask)[0]
    w_name_new   = [w_name[i] for i in indices]
    w_symbol_new = [w_symbol[i] for i in indices] 

    # Also filter full names if provided
    if w_full_name is not None:
        w_full_name_new = [w_full_name[i] for i in indices]
        return x_new, w_type_new, w_color_new, w_name_new, w_symbol_new, w_full_name_new

    return x_new, w_type_new, w_color_new, w_name_new, w_symbol_new


def create_slider(key_suffix):
    """Create a slider that acts as a scaling factor for degree-based node sizes"""
    return st.slider(
        "Node size scale", 
        min_value=0.3, 
        max_value=2.5, 
        value=1.0, 
        step=0.1,
        key=f"slider {key_suffix}",
        help="Scale all node sizes while preserving degree-based differences"
    )


#@st.cache_resource(show_spinner=False)
def cached_hyperlipea(_x, wtype, wcolor, full_names, wsymbol, option, omics_type, e_colors=None, build_edges=True, _heartbeat=None, corr_1_name='Correction #1', corr_2_name='Correction #2'):

    coords, excel_buffer = __hyperlipea(_x, wtype, wcolor, full_names, wsymbol, option, omics_type, e_colors, build_edges, heartbeat=_heartbeat, corr_1_name=corr_1_name, corr_2_name=corr_2_name)

    return coords, excel_buffer


def __hyperlipea(x_, w_type, w_color, w_name, w_symbol, option, omics, e_colors=None, build_edges=True, heartbeat=None, corr_1_name='Correction #1', corr_2_name='Correction #2'):
    bins = csgraph.connected_components(x_)[1].T
    ncc = len(np.unique(bins))

    w_layer = (w_type > 0).reshape(1, -1)

    if ncc == 1:
        # Pass heartbeat into __coal_embed
        coords = __coal_embed(x_, 'RA1', 'ncISO', 'EA', 2, heartbeat=heartbeat)
        coords_plot = __compute_plot_coords(coords)
        # --- Prepare download data ---
        # Sheet 1: Node coordinates
        df_coords = pd.DataFrame(coords_plot, columns=['x', 'y'])
        df_coords['node label'] = w_name

        # Sheet 2: Interactions (edges list with labels)
        e1, e2 = triu(x_, k=1).nonzero()
        edges_list = pd.DataFrame({
        'Pathway name': [w_name[i] for i in e1],
        'Molecule name': [w_name[j] for j in e2]
        })

        # Create Excel file in memory
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df_coords.to_excel(writer, sheet_name='Node coordinates', index=False)
            edges_list.to_excel(writer, sheet_name='Edges', index=False)

        return coords, excel_buffer

    else:  # if ncc>1
        mask = np.array(x_.sum(axis=1)).ravel()
        t0 = np.zeros((2, ncc)).astype(int)  # first row pathways, second row lipids

        for ncc_index in range(ncc):
            te1 = bins == ncc_index
            t0[0, ncc_index] = np.argmax(te1 * mask * w_layer == np.max(te1 * mask * w_layer))
            t0[1, ncc_index] = np.argmax(te1 * mask * ~w_layer == np.max(te1 * mask * ~w_layer))

       #list_ = np.empty_like(t0).astype(int)
        list_ = np.empty((ncc, 2), dtype=int)
        for ncc_index in range(ncc):
            l1 = np.arange(0, ncc)
            l1 = np.delete(l1, ncc_index)
            m1 = t0[1, l1].T
            m1 = m1[np.where(mask[m1] == np.max(mask[m1]))]
            list_[ncc_index, 0] = t0[0, ncc_index]
            list_[ncc_index, 1] = m1[0]
            #if len(m1) == 1:
            #    list_[ncc_index, 1] = m1[0]
            #else:
            #   # note: seems like this never happens
            #    raise ValueError('m1 has more than 1 value')

        x2 = x_.tolil()
        for i, j in list_:
            x2[i, j] = 2
            x2[j, i] = 2
        x2 = x2.tocsr()

        # Pass heartbeat into __coal_embed
        coords = __coal_embed(x2, 'RA1', 'ncISO', 'EA', 2, heartbeat=heartbeat)
        coords_plot = __compute_plot_coords(coords)
        # --- Prepare download data ---
        # Sheet 1: Node coordinates
        df_coords = pd.DataFrame(coords_plot, columns=['x', 'y'])
        df_coords['node label'] = w_name
        # Sheet 2: Interactions (edges list with labels)
        e1, e2 = triu(x_, k=1).nonzero()
        edges_list = pd.DataFrame({
        'Pathway name': [w_name[i] for i in e1],
        'Molecule name': [w_name[j] for j in e2]
        })

        # Create Excel file in memory
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df_coords.to_excel(writer, sheet_name='Node coordinates', index=False)
            edges_list.to_excel(writer, sheet_name='Edges', index=False)

        return coords, excel_buffer

 
def run_hyperlipea_with_progress(x, wtype, wcolor, fixed_names, wsymbol, option, omics_type,
                                e_colors=None, corr_1_name='Correction #1', corr_2_name='Correction #2',
                                coloring_scheme='similarity', labels_data=None, edge_opacity=0.6):
    """
    Run hyperlipea computation with real-time progress updates.
    Returns (coords, excel_buffer, fig) or raises an exception.
    """
    import itertools
    import threading
    import time
    
    # UI placeholders
    status = st.empty()
    progress = st.progress(0)
    spinner = itertools.cycle(["⏳", "⌛", "🔄"])

    state = {"phase": None, "done": None, "total": None}
    result = {}
    error = {}

    def heartbeat(phase, done, total):
        state["phase"] = phase
        state["done"] = done
        state["total"] = total

    def run_computation():
        try:
            coords, excel_buffer = cached_hyperlipea(
                x, wtype, wcolor, full_names, wsymbol, option, omics_type,
                e_colors=e_colors,
                _heartbeat=heartbeat, 
                corr_1_name=corr_1_name,
                corr_2_name=corr_2_name
            )
            coords[:, 0] = (coords[:, 0] + np.pi) % (2 * np.pi)  # rotate by pi
            result["coords"] = coords
            result["excel"] = excel_buffer
        except Exception as e:
            error["exception"] = e

    # Run computation in background thread
    t = threading.Thread(target=run_computation)
    t.start()

    # Progress tracking flags
    phase_started = {"ra1": False, "shortest_path": False, "svd": False}

    while t.is_alive():
        phase = state["phase"]
        done = state["done"]
        total = state["total"]

        if phase == "ra1" and isinstance(done, int) and isinstance(total, int):
            phase_started["ra1"] = True
            pct = min(int(done / total * 100), 100)
            progress.progress(pct)
            status.info(f"RA1 weighting: {done:,} / {total:,} edges")

        elif phase == "shortest_path" and isinstance(done, int) and isinstance(total, int):
            phase_started["shortest_path"] = True
            pct = min(int(done / total * 100), 100)
            progress.progress(pct)
            status.info(f"Shortest path: {pct}%")

        elif phase == "svd":
            phase_started["svd"] = True
            status.info(f"SVD running {next(spinner)}")

        else:
            # Show status based on what phase we're waiting for
            if phase_started["svd"]:
                status.info(f"SVD running {next(spinner)}")
            elif phase_started["shortest_path"]:
                status.info(f"Shortest path {next(spinner)}")
            elif phase_started["ra1"]:
                status.info(f"RA1 weighting {next(spinner)}")
            else:
                status.info(f"Starting computation {next(spinner)}")

        time.sleep(0.5)

    t.join()

    if "exception" in error:
        status.error(f"Error: {error['exception']}")
        raise error["exception"]

    # Computation done, now plot
    status.info("✅ Embedding computed, generating plot...")
    progress.progress(100)

    plotting_done = {"flag": False}

    def run_plot():
        try:
            # Initialize edge_colors at the start of the function
            edge_colors_for_plot = e_colors  # Use the parameter passed to parent function
            # Apply correct coloring based on scheme
            if coloring_scheme == 'preference':
                # Get custom colors from session state
                custom_node_colors = st.session_state.get('custom_node_colors_fig1', None)
                custom_edge_colors = st.session_state.get('custom_edge_colors_fig1', None)
                
                # Get full names from session state
                full_names = st.session_state.get('full_names_fig1', None)
                
                plot_colors = apply_gradient_coloring(
                    x, result["coords"], wsymbol,
                    coloring='preference',
                    custom_node_colors=custom_node_colors,
                    full_names=full_names
                )
                coloring_mode = 'gradient'
                
                # Use custom edge colors if provided
                if custom_edge_colors is not None:
                    edge_colors_for_plot = custom_edge_colors
            elif coloring_scheme == 'default':
                # Use the wcolor passed in (could be from file or generated)
                # If wcolor is None, check if edge colors exist
                if wcolor is not None:
                    plot_colors = wcolor
                else:
                    # NEW: If no node colors but edge colors exist, use black for diamonds, red for circles
                    if e_colors is not None and isinstance(e_colors, dict) and len(e_colors) > 0:
                        plot_colors = np.zeros((len(wsymbol), 3))
                        for i, shape in enumerate(wsymbol):
                            if shape == 'd':  # diamond
                                plot_colors[i] = [0, 0, 0]  # black
                            else:  # circle
                                plot_colors[i] = [1, 0, 0]  # red
                    else:
                        # Fallback to gradient if no node colors and no edge colors provided
                        plot_colors = apply_gradient_coloring(
                            x, result["coords"], wsymbol,
                            coloring='similarity'
                        )
                # BUGFIX: When we generate black/red colors, use 'gradient' mode to apply them properly
                # Only use 'default' mode when we have actual node colors from the file
                if wcolor is not None:
                    coloring_mode = 'default'  # Use default mode for file-provided colors
                else:
                    coloring_mode = 'gradient'  # Use gradient mode for generated colors
            elif coloring_scheme != 'Pathway Significance':
                plot_colors = apply_gradient_coloring(
                    x, result["coords"], wsymbol,
                    coloring=coloring_scheme,
                    labels=labels_data
                )
                coloring_mode = 'gradient'
            else:
                plot_colors = wcolor
                coloring_mode = 'default'

            fig = __plot_hyperlipea_interactive(
                x, result["coords"], plot_colors, fixed_names, wsymbol, option, omics_type, 
                e_colors=edge_colors_for_plot, corr_1_name=corr_1_name, corr_2_name=corr_2_name,
                coloring_mode=coloring_mode, edge_opacity=edge_opacity, 
                coloring_scheme=coloring_scheme
            )
            result["fig"] = fig
        finally:
            plotting_done["flag"] = True

    plot_thread = threading.Thread(target=run_plot)
    plot_thread.start()

    while not plotting_done["flag"]:
        status.info(f"Plotting network {next(spinner)}")
        time.sleep(0.5)

    plot_thread.join()
    status.success("✅ Almost there! The final visualization is being prepared — please wait a few seconds...")

    return result["coords"], result["excel"], result["fig"]


script_path = abspath(join_path(dirname(__file__)))
figures_path = join_path(script_path, 'figures')
if not path_exists(figures_path):
    makedirs(figures_path)

# Create plot with adjusted significant only pathway
# This dataset contains a network that is one unique component in Fig.1 and 2 components in Fig.2
#file_path = join_path(script_path, 'data_final_transformed.mat')

#data_content = sio.loadmat(file_path)
# Load Excel file
#df = pd.read_excel("lipea_results.xlsx")
st.set_page_config(
    page_title="Hyperpathways",
    page_icon="./images/Logo_Hyperpathway.png", 
    layout="wide",
)

st.markdown("""
    <style>
        /* Remove top padding from main container */
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            padding-bottom: 4rem !important;   /* add space at the bottom */
        }
        
        /* Remove header spacing */
        header[data-testid="stHeader"] {
            height: 0px !important;
            padding: 0px !important;
            background-color: transparent !important;
        }      
    </style>
""", unsafe_allow_html=True)

## LOGO ##
_, col_left, col_middle, col_right, _ = st.columns([1.2, 2, 6, 2, 1.2], vertical_alignment='center')

def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

logo_b64 = img_to_base64("./images/Logo_Hyperpathway.png")

with col_middle:
    st.markdown(f"""
    <style>
      .hp-header-row {{
        display: flex;
        align-items: center;        /* vertical align logo + text */
        justify-content: center;    /* center the whole row */
        gap: 16px;                  /* space between logo and title */
        margin-top: 0px;
        margin-bottom: 30px;        /* space below logo + title */
      }}
      .hp-header-row img {{
        width: 200px;
        height: auto;
      }}
      .hp-header-row h1 {{
        margin: 0;                  /* remove default h1 margins */
        padding: 0;
      }}
    </style>

    <div class="hp-header-row">
      <img src="data:image/png;base64,{logo_b64}" alt="Hyperpathway logo" />
      <h1>Hyperpathway Web App</h1>
    </div>
    """, unsafe_allow_html=True)

_, col_middle, _ = st.columns([1.2, 12, 1.2], vertical_alignment='center')
with col_middle:
    st.markdown(f"""
    <div style='text-align: center;'>
      <div style='font-size:25px; color: gray; margin-bottom: 25px;'>
        <i><a href='https://www.preprints.org/manuscript/202508.2227' target='_blank'>
          Hyperpathway: visualizing organization of pathway-molecule enriched interactions in omics studies via hyperbolic network embedding
        </a></i><br>
        Ilyes Abdelhamid, Ziheng Liao, Yuchi Liu, Armel Lefebvre, Aldo Acevedo &amp; Carlo Vittorio Cannistraci<br>
        Preprints.org, 2025.
      </div>
    </div>
    """, unsafe_allow_html=True)

_, col_middle, _ = st.columns([1.2, 10, 1.2], vertical_alignment='center')
with col_middle:
    st.markdown(f"""
    <div style='font-size:22px; color: gray; margin-top: 5px; padding-left: 70px;'>
      <a href="https://github.com/biomedical-cybernetics/Hyperpathway">GitHub link</a><br>
      <a href="https://brain.tsinghua.edu.cn/en/Research1/Research_Centers/Complex_Network_Intelligence_Center.htm">Center for Complex Network Intelligence</a><br>
      Contact: Ilyes Abdelhamid <a href="mailto:ilyes.abdelhamid1@gmail.com">(ilyes.abdelhamid1@gmail.com)</a>
      or Carlo Vittorio Cannistraci <a href="mailto:kalokagathos.agon@gmail.com">(kalokagathos.agon@gmail.com)</a>
    </div>
    """, unsafe_allow_html=True)

# Add spacing before the three-column layout
st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)

# Three columns: table_genomic (left), Figure_2 (middle), text (right)
_, col_left, col_middle, col_right, _ = st.columns([1.2, 2, 2, 2, 1.2], vertical_alignment='center')

with col_left:
    # Add vertical spacing to center-align with Figure_2
    st.image("./images/table_genomic.png", width=500)

with col_middle:
    st.image("./images/Figure_2.png", width=700)

with col_right:
    st.markdown("""
        <div style='font-size:18px; padding-left: 6px; text-align: justify;'>
            Example of Hyperpathway visualization of <a href="https://www.pnas.org/doi/10.1073/pnas.142287999?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub++0pubmed">genomic data</a> associated to 403 genes 
            up-regulated in peripheral blood mononuclear cells from four healthy donors 
            exposed to HIV gp120 envelope proteins (Figure 2 of the <a href="https://www.preprints.org/manuscript/202508.2227">article</a>)
        </div>
    """, unsafe_allow_html=True)

## LOGO ##
st.markdown("---")

# ----------------------------
# Omics selection (updated)
# ----------------------------
st.markdown(
    """
    <div style='font-size:20px; font-weight: bold; margin-bottom: -2px; line-height: 1.2;'>
        🔬 Select the type of input data you want to visualize:
    </div>
    """,
    unsafe_allow_html=True
)

if "selection_omics" not in st.session_state:
    st.session_state.selection_omics = "-- Select the type of input data --"

omics_choice = st.selectbox(
    "Select omics type",
    [
        "-- Select the type of input data --",
        "Lipidomics",
        "Genomics",
        "Metabolomics",
        "Other Omics",
        "Simple bipartite network",  # <- replaced "Others"
    ],
    key="selection_omics",
    label_visibility="collapsed",
)

# Clear state when the choice changes
if "previous_omics_type" in st.session_state and st.session_state["previous_omics_type"] != omics_choice:
    keys_to_clear = [
        "fig1", "fig2", "coords_embedding_fig1", "coords_embedding_fig2",
        "excel_buffer_fig1", "excel_buffer_fig2", "selected_nodes_fig1",
        "selected_nodes_fig2", "demo_genomics", "demo_metabolomics",
        "demo_lipidomics", "last_pea_file_id", "last_bipartite_file_id",
        "last_node_file_id"
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)

st.session_state["previous_omics_type"] = omics_choice

# Block until a valid choice is made
if omics_choice == "-- Select the type of input data --":
    st.stop()

# Mode flag: bipartite-only mode
is_bipartite_only = (omics_choice == "Simple bipartite network")

# Map to internal type labels used elsewhere (keep your downstream compatibility)
if is_bipartite_only:
    st.info("Bipartite network visualization selected.")
    omics_type = "N/A"   # safest default for downstream code that expects a string label
elif omics_choice == "Lipidomics":
    st.info("Lipidomics selected.")
    omics_type = "Lipids"
elif omics_choice == "Genomics":
    st.info("Genomics selected.")
    omics_type = "Genes"
elif omics_choice == "Metabolomics":
    st.info("Metabolomics selected.")
    omics_type = "Metabolites"
elif omics_choice == "Other Omics":
    st.info("Other omics selected.")
    omics_type = "Molecules"


# ----------------------------
# Upload section + guide
# ----------------------------

st.markdown("""
<style>
div[data-testid="stExpander"] {
    border: 3px solid #2196F3;
    border-radius: 15px;
    background-color: #e3f2fd;
    margin: 30px 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
div[data-testid="stExpander"] summary {
    font-size: 28px !important;
    font-weight: bold !important;
    color: #0D47A1 !important;
    padding: 25px !important;
    line-height: 1.5 !important;
}
div[data-testid="stExpander"] summary:hover {
    background-color: #bbdefb;
    cursor: pointer;
}
div[data-testid="stExpander"] summary svg {
    width: 32px !important;
    height: 32px !important;
}
</style>
""", unsafe_allow_html=True)

with st.expander("▼ Expand to see file format and coloring guide"):

    if is_bipartite_only:
        # --- ONLY Option 2 guide ---
        st.markdown(
            """
            <div style='font-size:18px; margin-top:0px;'>
                <b>You selected to visualize a bipartite network.</b><br><br>
                <b>📁 File 1 – Bipartite adjacency list:</b><br>
                <ul>
                    <li><b>Column 1</b>: Source node name (e.g., pathway)</li>
                    <li><b>Column 2</b>: Target node name (e.g., molecule)</li>
                    <li><b>Column 3</b> (optional): Edge color</li>
                </ul>
            </div>
            <div style='font-size:18px; margin-top:10px;'>
                <b>📁 File 2 – Node list (optional):</b><br>
                <ul>
                    <li><b>Column 1</b>: Node name</li>
                    <li><b>Column 2</b>: Node color</li>
                </ul>
                ⚠️ <b>Important:</b> All node names in the node list must exactly match those in the bipartite adjacency list. Any mismatch will cause an error.<br><br>
                🎨 <b>Color format:</b> You can provide colors in either of the following formats:
                <ul>
                    <li><b>HEX:</b> <code>#FF5733</code> (must start with <code>#</code> and be 6 characters long)</li>
                    <li><b>RGB:</b> <code>255 87 51</code> (3 spaced integers between 0 and 1 or 0 and 255)</li>
                </ul>
            </div>
            🖼️ <i>Example of expected file format:</i>
            """,
            unsafe_allow_html=True
        )

        st.image("./images/example_option2_file_format.png", width=600)

        st.markdown(
            """
            <div style='font-size:18px;'>
                <b>🎨 Node Coloring Schemes:</b><br>
                <ul>
                    <li><b>Similarity</b>: Angular coordinate-based gradient coloring (HSV). Molecule nodes inherit average of connected pathway colors.</li>
                    <li><b>Hierarchy</b>: Radial coordinate-based coloring (blue-to-red) based on degree centrality. Molecule nodes inherit average of connected pathway colors.</li>
                    <li><b>Default</b>: Uses custom colors from node file; unspecified nodes use defaults (red for circle nodes and black for diamond nodes).</li>
                </ul>
            </div>
            <div style='font-size:18px;'>
                <b>🎨 Edge Coloring Schemes</b>:<br>
                <ul>
                    <li><b>Similarity/Hierarchy</b>: Edges inherit the color of their connected pathway node.</li>
                    <li><b>Default</b>: Uses edge colors from adjacency file or defaults to gray.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        # --- Your original full guide (Option 1 + Option 2) ---
        st.markdown(
            """
            <div style='font-size:18px;'>
                <b>You selected to visualize a pathway enrichment table.</b><br><br>
                <b>The application accepts tables with a minimum of two required columns: (1) pathway name and (2) molecules associated with each pathway.</b><br>
                <b>Statistical significance columns and others extra categories (e.g., non-corrected p-value, p-value corrections, pathway community membership, etc) are optional but enable pathway significance-based visualization and label-based one.</b><br><br>
                <b>Two input formats are supported (see example file below):
                <ul>
                    <li><b>1. one pathway-molecule pair per row</b></li>
                    <li><b>2. multiple molecules per pathway in a single row with molecules separated by semicolons (;), commas (,), or pipe symbols (|) within the same cell</b></li>
                </ul>
                <b> Accepted file formats include XLSX, XLS, CSV, and TSV.</b><br>
                <span style='color:red; font-weight:600;'>
                ⚠️ The input table does NOT need to follow a specific column order or use fixed column names.
                </span><br><br>
                🖼️ <i>Example of expected file format:</i>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.image("./images/example_file_format.png", width=800)

        # (keep your button CSS / demo links exactly as you had them)
        st.markdown("""
        <style>
        div[data-testid="stHorizontalBlock"] button {
            background: linear-gradient(to bottom, #b6fcd5, #90ee90);
            color: green !important;
            border-radius: 8px;
            font-weight: bold;
            border: 1px solid #6cb56c;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.6), 0 4px 6px rgba(0, 0, 0, 0.2);
            text-shadow: 0 1px 1px rgba(255, 255, 255, 0.5);
            transition: all 0.2s ease-in-out;
        }
        div[data-testid="stHorizontalBlock"] button:hover {
            background: linear-gradient(to bottom, #c1ffe0, #a9f5bc);
        }
        div[data-testid="stHorizontalBlock"] button:active {
            transform: translateY(2px);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.3), 0 2px 3px rgba(0, 0, 0, 0.2);
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown(
            "<div style='font-size:18px; margin-top:15px;'><b>📥 Download demo files:</b></div>",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns([0.2, 1])
        with col1:
            st.markdown(
                "<a href='https://raw.githubusercontent.com/IlyesAbdelhamid/Hyperpathway/main/demo_data/demo_genomics.xlsx' "
                "download='demo_genomics.xlsx' style='text-decoration:none;'>Demo Genomics File</a>",
                unsafe_allow_html=True
            )

        col1, col2 = st.columns([0.2, 1])
        with col1:
            st.markdown(
                "<a href='https://raw.githubusercontent.com/IlyesAbdelhamid/Hyperpathway/main/demo_data/demo_metabolomics.csv' "
                "download='demo_metabolomics.csv' style='text-decoration:none;'>Demo Metabolomics File</a>",
                unsafe_allow_html=True
            )

        col1, col2 = st.columns([0.2, 1])
        with col1:
            st.markdown(
                "<a href='https://raw.githubusercontent.com/IlyesAbdelhamid/Hyperpathway/main/demo_data/demo_lipidomics.xls' "
                "download='demo_lipidomics.xls' style='text-decoration:none;'>Demo Lipidomics File</a>",
                unsafe_allow_html=True
            )
        st.markdown(
            """
            <div style='font-size:18px;'>
                <b>🎨 Node Coloring Schemes:</b><br>
                <ul>
                    <li><b>Similarity</b>: Angular coordinate-based gradient coloring (HSV). Molecule nodes inherit average of connected pathway colors.</li>
                    <li><b>Hierarchy</b>: Radial coordinate-based coloring (blue-to-red) based on degree centrality. Molecule nodes inherit average of connected pathway colors.</li>
                    <li><b>Labels</b>: Category-based coloring by pathway annotations (distinct HSV palette).</li>
                    <li><b>Preference</b>: Upload your own colors for individual nodes (HEX or RGB).</li>
                </ul>
            </div>
            <div style='font-size:18px;'>
                <b>🎨 Edge Coloring Schemes</b>:<br>
                <ul>
                    <li><b>Similarity/Hierarchy/Labels</b>: Edges inherit the color of their connected pathway node.</li>
                    <li><b>Preference</b>: Uses edge colors from adjacency file or defaults to gray.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )    

        

# ----------------------------
# Uploaders (updated to hide Option 1 in bipartite-only mode)
# ----------------------------
uploaded_pea_file = None
uploaded_bipartite_file = None

if not is_bipartite_only:
    st.markdown(
        """
        <div style='font-size:20px; font-weight: bold; margin-bottom: -2px; line-height: 1.2;'>
            Upload your pathway enrichment results table (for standard Hyperpathway visualization).
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_pea_file = st.file_uploader(
        "Upload PEA results file",
        type=["xlsx", "xls", "csv", "tsv"],
        key="pea_uploader",
        label_visibility="collapsed"
    )

    if uploaded_pea_file is not None:
        current_file_id = f"{uploaded_pea_file.name}_{uploaded_pea_file.size}"
        if st.session_state.get("last_pea_file_id") != current_file_id:
            keys_to_clear = [
                "fig1", "fig2", "coords_embedding_fig1", "coords_embedding_fig2",
                "excel_buffer_fig1", "excel_buffer_fig2", "selected_nodes_fig1",
                "selected_nodes_fig2", "demo_genomics", "demo_metabolomics", "demo_lipidomics"
            ]
            for key in keys_to_clear:
                st.session_state.pop(key, None)
            st.session_state["last_pea_file_id"] = current_file_id

else:
    # Option 2 uploader is always shown (and the only one shown in bipartite-only mode)
    st.markdown(
        """
        <div style='font-size:20px; font-weight: bold; margin-bottom: -2px; line-height: 1.2;'>
            Upload your bipartite adjacency list file (for general purpose bipartite network visualization).
        </div>
        """,
        unsafe_allow_html=True
    )

    def _clear_session_keys(keys):
        for k in keys:
            st.session_state.pop(k, None)

    OPTION2_KEYS = [
        "option2_fig1", "option2_coords_fig1", "option2_excel_buffer_fig1", "selected_nodes_option2_fig1",
        "option2_fig2", "option2_coords_fig2", "option2_excel_buffer_fig2", "selected_nodes_option2_fig2",
        "wcolor_base_option2_fig1", "wcolor_current_option2_fig1", "edge_colors_option2_fig1",
        "wcolor_base_option2_fig2", "wcolor_current_option2_fig2", "edge_colors_option2_fig2",
        "edges_need_rebuild_option2_fig1", "edges_need_rebuild_option2_fig2",
    ]

    uploaded_bipartite_file = st.file_uploader(
        label="Upload bipartite network file",
        type=["xlsx", "xls", "csv", "tsv"],
        key="bipartite_uploader",
        label_visibility="collapsed"
    )

    last_id = st.session_state.get("last_bipartite_file_id")

    if uploaded_bipartite_file is None:
        if last_id is not None:
            _clear_session_keys(OPTION2_KEYS)
            st.session_state["last_bipartite_file_id"] = None
    else:
        current_id = f"{uploaded_bipartite_file.name}_{uploaded_bipartite_file.size}"
        if last_id != current_id:
            _clear_session_keys(OPTION2_KEYS)
            st.session_state["last_bipartite_file_id"] = current_id

if uploaded_pea_file:
    # Proceed with pathway enrichment workflow
    try:
        option = 1;
        if uploaded_pea_file:
            # Store original DataFrame BEFORE processing
            if uploaded_pea_file.name.endswith('.csv'):
                df_original = pd.read_csv(uploaded_pea_file)
            elif uploaded_pea_file.name.endswith('.tsv') or uploaded_pea_file.name.endswith('.txt'):
                df_original = pd.read_csv(uploaded_pea_file, sep='\t')
            else:
                df_original = pd.read_excel(uploaded_pea_file)
    
            # Store in session state for later use
            st.session_state.pea_dataframe = df_original

            uploaded_pea_file.seek(0)

            result = process_input_pea_table(uploaded_pea_file)
            if result is None:
                # Stop here — user hasn’t finished selection
                st.stop()   # Streamlit helper to halt execution cleanly
            else:
                pathway_names, enriched_molecules, uncorrected_pvalues, corr_1_values, corr_2_values, pval_signi_non_corr, pval_signi_corr_1, pval_signi_corr_2 = result
                    
                # ⭐ DETECT COLUMN SELECTION CHANGES
                # Get current column selections from session state (set by process_input_pea_table)
                current_column_selections = {
                    'col_pathway_name': st.session_state.get('col_pathway_name'),
                    'col_mols_pathway': st.session_state.get('col_mols_pathway'),
                    'col_non_corr': st.session_state.get('col_non_corr'),
                    'col_corr_1': st.session_state.get('col_corr_1'),
                    'col_corr_2': st.session_state.get('col_corr_2')
                }

                # Get previous selections
                previous_column_selections = st.session_state.get('previous_column_selections', None)

                # Check if column selections changed
                columns_changed = False
                if previous_column_selections is not None:
                    if current_column_selections != previous_column_selections:
                        columns_changed = True
                        st.info(f"🔄 Column selections changed. Clearing previous visualizations...")

                # If columns changed, clear existing figures
                if columns_changed:
                    keys_to_clear = ["fig1", "fig2", "coords_embedding_fig1", "coords_embedding_fig2",
                                    "excel_buffer_fig1", "excel_buffer_fig2", "wcolor_current_fig1",
                                    "selected_nodes_fig1", "selected_nodes_fig2",
                                    "previous_pvalue_thresholds"]  # Also clear threshold tracking
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]

                # Store current selections for next comparison
                st.session_state.previous_column_selections = current_column_selections

                # ⭐ DETECT P-VALUE THRESHOLD CHANGES
                # Store current thresholds for comparison
                current_thresholds = {
                    'non_corr': pval_signi_non_corr,
                    'corr_1': pval_signi_corr_1,
                    'corr_2': pval_signi_corr_2
                }
                    
                # Get previous thresholds from session state
                previous_thresholds = st.session_state.get('previous_pvalue_thresholds', None)
                    
                # Check if thresholds changed
                thresholds_changed = False
                if previous_thresholds is not None:
                    if (previous_thresholds['non_corr'] != current_thresholds['non_corr'] or
                        previous_thresholds['corr_1'] != current_thresholds['corr_1'] or
                        previous_thresholds['corr_2'] != current_thresholds['corr_2']):
                        thresholds_changed = True
                        st.info(f"🔄 P-value threshold changed. Clearing previous visualizations...")
                    
                # If thresholds changed, clear existing figures to force recomputation
                if thresholds_changed:
                    keys_to_clear = ["fig1", "fig2", "coords_embedding_fig1", "coords_embedding_fig2",
                                    "excel_buffer_fig1", "excel_buffer_fig2", "wcolor_current_fig1",
                                    "selected_nodes_fig1", "selected_nodes_fig2"]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                # Store current thresholds for next comparison
                st.session_state.previous_pvalue_thresholds = current_thresholds
                    
                x_raw, wname_raw, wtype_raw = cached_build_network(
                    pathway_names, enriched_molecules, uncorrected_pvalues, corr_1_values, corr_2_values, 
                    pval_signi_non_corr, pval_signi_corr_1, pval_signi_corr_2
                )
                if x_raw is None:
                    st.stop() 

        if 'x_raw' not in locals():
            st.warning("⚠️ Please select column mappings and configure settings to proceed.")
            st.stop()

        st.success("Pathway enrichment table uploaded and converted to bipartite network successfully!")
    except Exception as e:
        st.error(f"Error loading Pathway enrichment file: {e}")
        st.stop()

    x = deepcopy(x_raw)
    wname = deepcopy(wname_raw)
    wtype = deepcopy(wtype_raw)

    # remove pathways not significant
    wname = np.array(wname)  
    wtype = np.array(wtype)

    unique_vals = set(wtype)
    # Create a mask that keeps only wtype == 0 or wtype == 4 if no p-value provided.
    if unique_vals == {0, 4}:
        keep_mask = (wtype == 0) | (wtype == 4)
    else:
        keep_mask = (wtype == 0) | (wtype == 1) | (wtype == 2) | (wtype == 3)
    # Apply the mask
    wname = np.reshape(wname[keep_mask], (np.sum(keep_mask), 1))
    wtype = np.reshape(wtype[keep_mask], (np.sum(keep_mask), 1))
    x = x[np.reshape(keep_mask, x.shape[0]), :]
    x = x[:, np.reshape(keep_mask, x.shape[1])]

    # create the color for each type of label
    wcolor = np.zeros((len(wtype), 3))
    wcolor[np.reshape(wtype == 0, wcolor.shape[0]), :] = np.tile([0, 1, 0], (np.sum(wtype == 0), 1))
    wcolor[np.reshape(wtype == 1, wcolor.shape[0]), :] = np.tile([1, 0, 0], (np.sum(wtype == 1), 1))
    wcolor[np.reshape(wtype == 2, wcolor.shape[0]), :] = np.tile([1, 0.6, 0], (np.sum(wtype == 2), 1))
    wcolor[np.reshape(wtype == 3, wcolor.shape[0]), :] = np.tile([0.7, 0.7, 0.7], (np.sum(wtype == 3), 1))
    wcolor[np.reshape(wtype == 4, wcolor.shape[0]), :] = np.tile([0, 0, 1], (np.sum(wtype == 4), 1))

    fixed_names = []  # For display (first word)
    full_names = []   # For matching with DataFrame

    for i in range(len(wname)):
        original_name = wname[i][0]
        first_word = original_name.split(' ', 1)[0]  # first word for display
        fixed_names.append(first_word)
        full_names.append(original_name)  # full name for matching

    wsymbol = []
    for i in range(len(wname)):
        if wtype[i] == 0:
            wsymbol.append("o")
        else:
            wsymbol.append("d")

    # Remove isolated nodes
    x, wtype, wcolor, fixed_names, wsymbol, full_names = remove_isolated_nodes(x, wtype, wcolor, fixed_names, wsymbol, full_names)
    # Store full names for preference coloring
    st.session_state.full_names_fig1 = full_names
    
    o_nodes = [i for i, s in enumerate(wsymbol) if s == "o"]
    d_nodes = [i for i, s in enumerate(wsymbol) if s == "d"]

    # Add an anchor/ID marker right before the compute buttons
    st.markdown('<div id="compute-buttons-section"></div>', unsafe_allow_html=True)

    # Scroll trigger logic
    if st.session_state.get("scroll_to_compute", False):
        scroll_to_element("compute-buttons-section")
        st.session_state.scroll_to_compute = False  # Reset flag

    col1, col2 = st.columns(2)
    edge_colors = None
    with col1:
        # Node Coloring Options
        st.markdown("**🎨 Node Coloring Options:**")
        # Determine available coloring options based on p-value selection
        has_pvalues = st.session_state.get("col_non_corr") is not None or \
                st.session_state.get("col_corr_1") is not None or \
                st.session_state.get("col_corr_2") is not None

        # Build coloring options list
        coloring_options = ['similarity', 'hierarchy', 'labels', 'preference']
        format_dict = {
            'hierarchy': 'Hierarchy (Radial position)',
            'similarity': 'Similarity (Angular position)',
            'labels': 'Labels (Custom categories)',
            'preference': 'Preference (Custom node/edge colors)'
        }

        if has_pvalues:
            coloring_options.append('Pathway Significance')
            format_dict['Pathway Significance'] = 'Pathway significance (requires p-values)'

        coloring_scheme = st.radio(
            "Select coloring scheme:",
            options=coloring_options,
            format_func=lambda x: format_dict[x],
            key="coloring_scheme_fig1",
            horizontal=True
        )

        # Show warning if Pathway Significance is somehow selected without p-values
        if coloring_scheme == 'Pathway Significance' and not has_pvalues:
            st.warning("⚠️ Pathway significance coloring requires p-value data. Switching to similarity coloring.")
            coloring_scheme = 'similarity'

        # ⭐ CRITICAL: Clear custom colors when switching away from preference mode
        previous_coloring_scheme = st.session_state.get('coloring_scheme_stored_fig1', None)
        if previous_coloring_scheme == 'preference' and coloring_scheme != 'preference':
            # User switched away from preference mode - clear custom colors
            if 'custom_node_colors_fig1' in st.session_state:
                del st.session_state['custom_node_colors_fig1']
            if 'custom_edge_colors_fig1' in st.session_state:
                del st.session_state['custom_edge_colors_fig1']
            # Force edge rebuild to apply new coloring
            st.session_state.edges_need_rebuild_fig1 = True

        # Initialize variables
        labels_data = None
        custom_node_color_map = None
        custom_edge_color_map = None

        # Show column selector if 'labels' selected
        if coloring_scheme == 'labels':
            st.info("📋 Select a column from your pathway enrichment table to use as labels")
    
            if 'pea_dataframe' in st.session_state:
                df = st.session_state.pea_dataframe
        
                # Let user select any column from the table
                selected_column = st.selectbox(
                    "Choose column for labels:",
                    options=df.columns.tolist(),
                    key="label_column_selector_fig1"
                )

            # 🔍 DEBUG: Show raw data samples
            #st.write("### 🔍 DEBUG: Raw Data Preview")
            #st.write(f"**Selected Column:** `{selected_column}`")
            #st.write("**First 5 rows of relevant columns:**")
            #selected_col_idx = df.columns.get_loc(selected_column)
            #col_indices = list(set([0, 1, selected_col_idx]))  # Remove duplicates with set()
            #col_indices.sort()  # Keep consistent order
            #st.dataframe(df.iloc[:5, col_indices])

            # Create a mapping dictionary for faster lookups
            # Pathway column (column 0) mapping
            pathway_label_map = {}
            molecule_label_map = {}

            for idx, row in df.iterrows():
                pathway_name = str(row.iloc[0]).strip()
                molecule_name = str(row.iloc[1]).strip()
                label_value = str(row[selected_column]).strip()

                pathway_label_map[pathway_name] = label_value
                if molecule_name not in molecule_label_map:
                    molecule_label_map[molecule_name] = label_value

            # 🔍 DEBUG: Show mapping stats
            #st.write("### 🔍 DEBUG: Mapping Statistics")
            #st.write(f"**Pathway map size:** {len(pathway_label_map)} unique pathways")
            #st.write(f"**Molecule map size:** {len(molecule_label_map)} unique molecules")
        
            # Build labels_data array
            labels_data = []
            unmatched_count = 0

            # 🔍 DEBUG: Track matching details
            pathway_matched = 0
            pathway_unmatched = []
            molecule_matched = 0
            molecule_unmatched = []

            #st.write("### 🔍 DEBUG: Node Matching Details")
            #st.write(f"**Total nodes to match:** {len(fixed_names)}")
        
            for i in range(len(fixed_names)):
                # Use the display name (first word) to reconstruct full name
                display_name = fixed_names[i]
                full_name = full_names[i]
            
                # Determine if this is a pathway (diamond) or molecule (circle)
                is_pathway = wsymbol[i] == 'd'
                     
                # Look up label based on node type using full name
                if is_pathway:
                    label_value = pathway_label_map.get(full_name, "Unknown")
                    if label_value != "Unknown":
                        pathway_matched += 1
                    else:
                        pathway_unmatched.append(display_name)
                else:
                    label_value = molecule_label_map.get(full_name, "Unknown")
                    if label_value != "Unknown":
                        molecule_matched += 1
                    else:
                        molecule_unmatched.append(display_name)

                if label_value == "Unknown":
                    unmatched_count += 1

                labels_data.append(label_value)

            # 🔍 DEBUG: Show detailed matching results
            #st.write(f"**Pathways (diamonds):**")
            #st.write(f"  - Matched: {pathway_matched}")
            #st.write(f"  - Unmatched: {len(pathway_unmatched)}")
        
            #st.write(f"**Molecules (circles):**")
            #st.write(f"  - Matched: {molecule_matched}")
            #st.write(f"  - Unmatched: {len(molecule_unmatched)}")
        
            # Show warnings
            if unmatched_count > 0:
                st.warning(f"⚠️ {unmatched_count} out of {len(fixed_names)} nodes could not be matched to the selected column. They will be colored gray.")
        
            # Validate labels
            unique_labels = set(labels_data) - {"Unknown"}
            if len(unique_labels) == 0:
                st.error("❌ No valid labels found in selected column. All nodes will be gray.")
                labels_data = None
            elif len(unique_labels) == 1:
                st.warning(f"⚠️ Only one unique label found: '{list(unique_labels)[0]}'. Consider selecting a different column for better visualization.")
        
        # Show file uploaders if 'preference' selected 
        elif coloring_scheme == 'preference':
            st.info("🎨 Upload custom color files for nodes and/or edges")
        
            # Node color file uploader
            st.markdown("**Optional: Upload node color file**")
            st.markdown("""
            <div style='font-size:14px; color: gray; margin-bottom: 10px;'>
            Format: 2 columns<br>
            • Column 1: Node name (exact match required)<br>
            • Column 2: Color in HEX (#FF0000), RGB space-separated (255 0 0 or 1.0 0.0 0.0)
            </div>
            """, unsafe_allow_html=True)
        
            uploaded_node_color_file = st.file_uploader(
                "Upload node colors",
                type=["xlsx", "xls", "csv", "tsv"],
                key="node_color_uploader_fig1",
                label_visibility="collapsed"
            )
        
            if uploaded_node_color_file is not None:
                custom_node_color_map = process_custom_node_colors(uploaded_node_color_file, full_names)
        
            # Edge color file uploader
            st.markdown("**Optional: Upload edge color file**")
            st.markdown("""
            <div style='font-size:14px; color: gray; margin-bottom: 10px;'>
            Format: 3 columns<br>
            • Column 1: Node 1 name<br>
            • Column 2: Node 2 name<br>
            • Column 3: Color in HEX or RGB format
            </div>
            """, unsafe_allow_html=True)
        
            uploaded_edge_color_file = st.file_uploader(
                "Upload edge colors",
                type=["xlsx", "xls", "csv", "tsv"],
                key="edge_color_uploader_fig1",
                label_visibility="collapsed"
            )
        
            if uploaded_edge_color_file is not None:
                custom_edge_color_map = process_custom_edge_colors(uploaded_edge_color_file, full_names)
        
            # Store in session state for later use
            if custom_node_color_map is not None:
                st.session_state.custom_node_colors_fig1 = custom_node_color_map
        
            if custom_edge_color_map is not None:
                st.session_state.custom_edge_colors_fig1 = custom_edge_color_map

        show_labels_fig1 = st.checkbox("Show node labels (Fig.1)", value=False, key="show_labels_fig1")
        show_edges_fig1 = st.checkbox("Show edges (Fig.1)", value=False, key="show_edges_fig1")
        edge_opacity_fig1 = st.slider("Edge opacity (Fig.1)", min_value=0.1, max_value=1.0, value=0.6, step=0.1, key="edge_opacity_fig1")
        node_size_slider = create_slider(f"Figure 1")
        st.markdown("""
        <style>
        div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
            background: linear-gradient(to bottom, #b6fcd5, #90ee90);
            color: green;
            border-radius: 8px;
            font-weight: bold;
            border: 1px solid #6cb56c;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.6), 0 4px 6px rgba(0, 0, 0, 0.2);
            text-shadow: 0 1px 1px rgba(255, 255, 255, 0.5);
            transition: all 0.2s ease-in-out;
        }

        div[data-testid="stHorizontalBlock"] > div:nth-child(1) button:hover {
            background: linear-gradient(to bottom, #c1ffe0, #a9f5bc);
        }

        div[data-testid="stHorizontalBlock"] > div:nth-child(1) button:active {
            transform: translateY(2px);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.3), 0 2px 3px rgba(0, 0, 0, 0.2);
        }
        </style>
        """, unsafe_allow_html=True)
        if st.button("Compute Hyperpathway embedding - Visualization #1", key="compute_option1_fig1"):

            try:
                # Show network size info
                n_nodes = x.shape[0]
                n_edges = x.nnz if hasattr(x, 'nnz') else np.count_nonzero(x)
                st.info(f"🔢 Network size: {n_nodes} nodes, {n_edges} edges")

                # Get the stored names
                corr_1_name = st.session_state.get('corr_1_display_name', 'Correction #1')
                corr_2_name = st.session_state.get('corr_2_display_name', 'Correction #2')

                # Initialize edge_colors properly
                edge_colors_to_use = None

                # Retrieve custom colors if preference mode
                if coloring_scheme == 'preference':
                    custom_node_colors = st.session_state.get('custom_node_colors_fig1', None)
                    custom_edge_colors = st.session_state.get('custom_edge_colors_fig1', None)

                    # Use custom edge colors if provided
                    if custom_edge_colors is not None:
                        edge_colors_to_use = custom_edge_colors
                # For other modes, do NOT use custom edge colors from session state
                # (edge_colors_to_use stays None, which means default gray edges)

                coords, excel_buffer, fig = run_hyperlipea_with_progress(
                    x, wtype, wcolor, full_names, wsymbol, option, omics_type,
                    e_colors=edge_colors_to_use, corr_1_name=corr_1_name, corr_2_name=corr_2_name,
                    coloring_scheme=coloring_scheme, labels_data=labels_data,
                    edge_opacity=edge_opacity_fig1
                )

                # Figure is already created with correct colors inside run_hyperlipea_with_progress
                # Just determine which colors were used for storage 
                if coloring_scheme == 'preference':
                    # For preference mode, store the custom colors that were actually used
                    custom_node_colors_used = st.session_state.get('custom_node_colors_fig1', None)
                    if custom_node_colors_used is not None:
                        wcolor_gradient = apply_gradient_coloring(
                            x, coords, wsymbol, 
                            coloring='preference',
                            custom_node_colors=custom_node_colors_used,
                            full_names=full_names
                        )
                    else:
                        # Fallback to default gray if no custom colors provided
                        wcolor_gradient = np.full((len(wsymbol), 3), 0.5)
                elif coloring_scheme != 'Pathway Significance':
                    wcolor_gradient = apply_gradient_coloring(
                        x, coords, wsymbol, 
                        coloring=coloring_scheme, 
                        labels=labels_data
                    )
                else: 
                    wcolor_gradient = wcolor

                # Store in session state
                st.session_state.edge_colors_fig1 = edge_colors_to_use  # Store edge colors used
                st.session_state.fig1 = fig
                st.session_state.coords_embedding_fig1 = coords
                st.session_state.excel_buffer_fig1 = excel_buffer
                st.session_state.wcolor_current_fig1 = wcolor_gradient # Store whichever colors were used
                st.session_state.coloring_scheme_stored_fig1 = coloring_scheme
                
            except Exception as e:
                st.error(f"Error during Hyperpathway computation: {e}")

    # Initialize correction names for display updates
    corr_1_name = st.session_state.get('corr_1_display_name', 'Correction #1')
    corr_2_name = st.session_state.get('corr_2_display_name', 'Correction #2')
    # When displaying existing plots, update the color scheme dynamically:
    if "fig1" in st.session_state:
        # If user changed coloring scheme, regenerate with new colors
        current_scheme = st.session_state.get('coloring_scheme_stored_fig1', 'similarity')
    
        # Only recompute colors, not the entire figure 
        if current_scheme != coloring_scheme or coloring_scheme == 'labels':
            # Clear stale selections
            if 'selected_nodes_fig1' in st.session_state:
                del st.session_state['selected_nodes_fig1']

            # Recompute ONLY colors (fast operation)
            if coloring_scheme != 'Pathway Significance':
                # Also handle preference mode for colors
                if coloring_scheme == 'preference':
                    custom_node_colors_for_recolor = st.session_state.get('custom_node_colors_fig1', None)
                    if custom_node_colors_for_recolor is not None:
                        wcolor_new = apply_gradient_coloring(
                            x, st.session_state.coords_embedding_fig1, wsymbol,
                            coloring='preference',
                            custom_node_colors=custom_node_colors_for_recolor,
                            full_names=full_names
                        )
                    else:
                        wcolor_new = np.full((len(wsymbol), 3), 0.5)
                
                else:
                    wcolor_new = apply_gradient_coloring(
                        x, st.session_state.coords_embedding_fig1, wsymbol,
                        coloring=coloring_scheme, labels=labels_data
                    )
            else:
                wcolor_new = wcolor # Original colors

            # Store new colors
            st.session_state.wcolor_current_fig1 = wcolor_new
            st.session_state.coloring_scheme_stored_fig1 = coloring_scheme
        
            # ⭐ CRITICAL: Force edge rebuild when coloring changes
            st.session_state.edges_need_rebuild_fig1 = True

        # Get current colors
        current_wcolor = st.session_state.get('wcolor_current_fig1', wcolor)
        
        # Check if opacity changed
        current_opacity = st.session_state.get('last_edge_opacity_fig1', 0.6)
        if current_opacity != edge_opacity_fig1:
            st.session_state.edges_need_rebuild_fig1 = True
            st.session_state.last_edge_opacity_fig1 = edge_opacity_fig1

        fig1 = go.Figure(st.session_state.fig1)
    
        # ⭐ DYNAMIC COLOR UPDATE: Modify node traces in-place
        o_nodes = [i for i, s in enumerate(wsymbol) if s == "o"]
        d_nodes = [i for i, s in enumerate(wsymbol) if s == "d"]

        # Update node colors for circle and diamond traces
        for trace_idx, trace in enumerate(fig1.data):
            if hasattr(trace, 'mode') and 'markers' in str(trace.mode) and trace.showlegend is False:
                symbol = trace.marker.symbol
        
                if symbol == 'circle':
                    node_indices = o_nodes
                elif symbol == 'diamond':
                    node_indices = d_nodes
                else:
                    continue
        
                # Build color list for this trace
                trace_colors = []
                for idx in node_indices:
                    if idx < len(current_wcolor):
                        rgb = current_wcolor[idx]
                        if isinstance(rgb, str):
                            trace_colors.append(rgb)
                        else:
                            trace_colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')
                    else:
                        trace_colors.append('rgb(128,128,128)')  # Fallback gray
            
                trace.marker.color = trace_colors

        # ⭐ FIX: Always remove old edges first, then rebuild if needed
        # Remove ALL edge traces
        fig1.data = [trace for trace in fig1.data 
                    if not (hasattr(trace, 'mode') and trace.mode == 'lines' and not trace.showlegend)]
    
        circle_boundary_trace = None
        legend_traces_existing = []
        node_traces_existing = []

        for trace in fig1.data:
            if hasattr(trace, 'mode'):
                if trace.mode == 'lines' and trace.showlegend is False and hasattr(trace, 'hoverinfo') and trace.hoverinfo == 'skip':
                    # This is the circle boundary
                    circle_boundary_trace = trace
                elif 'markers' in str(trace.mode) and trace.showlegend is False:
                    # This is a node trace
                    node_traces_existing.append(trace)
                elif trace.showlegend is True:
                    # This is a legend trace
                    legend_traces_existing.append(trace)

        # ---- Dynamic title + legend update (no recompute) ----
        fig1.update_layout(title=_hyperpathway_plot_title(coloring_scheme))

        # Rebuild legend traces to match current toggles (scheme / p-values)
        col_non_corr_sel = st.session_state.get('col_non_corr', None)
        col_corr_1_sel = st.session_state.get('col_corr_1', None)
        col_corr_2_sel = st.session_state.get('col_corr_2', None)

        corr_1_disp = st.session_state.get('corr_1_display_name', col_corr_1_sel or 'Correction #1')
        corr_2_disp = st.session_state.get('corr_2_display_name', col_corr_2_sel or 'Correction #2')

        legend_traces_existing = _build_hyperpathway_legend_traces(
            option=option,
            coloring_scheme=coloring_scheme,
            omics_type=omics_type,
            col_non_corr=col_non_corr_sel,
            col_corr_1=col_corr_1_sel,
            col_corr_2=col_corr_2_sel,
            corr_1_name=corr_1_disp,
            corr_2_name=corr_2_disp,
        )

        edge_traces_new = []
        needs_rebuild = (show_edges_fig1 or st.session_state.get('edges_need_rebuild_fig1', False))
        # Clear custom edge colors when switching away from preference mode
        if coloring_scheme != 'preference' and 'custom_edge_colors_fig1' in st.session_state:
            # Force rebuild to apply correct edge coloring for new mode
            needs_rebuild = True

        if needs_rebuild:
            # Build edges with current colors
            e1, e2 = triu(x, k=1).nonzero()
            max_edges = 20000
            if len(e1) > max_edges:
                idx = np.random.choice(len(e1), size=max_edges, replace=False)
                e1, e2 = e1[idx], e2[idx]

            coords_unit = __compute_plot_coords(st.session_state.coords_embedding_fig1)
            edges_by_color = {}
            default_color = '#CCCCCC'

            for e_start, e_end in zip(e1, e2):
                start_coords = coords_unit[e_start]
                end_coords = coords_unit[e_end]
    
                if np.sum(start_coords - end_coords) != 0:
                    _, _, cart_arc_1, cart_arc_2 = __hyperbolic_line_lipea(
                        start_coords.copy(), end_coords.copy(), 1, [0, 0], 'step', 0.05
                    )

                    # ⭐ CRITICAL: Transform back to native space (this was missing!)
                    pol_arc = np.empty((len(cart_arc_1), 2))
                    pol_arc[:] = np.nan
                    pol_arc[:, 0], pol_arc[:, 1] = __cart2pol(cart_arc_1, cart_arc_2)
                    pol_arc[:, 1] = 2 * np.arctanh(pol_arc[:, 1])  # Transform radius
                    cart_arc_1, cart_arc_2 = __pol2cart(pol_arc[:, 0], pol_arc[:, 1])

                    # Initialize edge_color with default
                    edge_color = default_color
        
                    if coloring_scheme == 'preference':
                        # PRIORITY 1: Check for explicit custom edge colors first
                        custom_edge_colors = st.session_state.get('custom_edge_colors_fig1', None)
                
                        if custom_edge_colors is not None:
                            canonical_key = (min(e_start, e_end), max(e_start, e_end))
                    
                            if canonical_key in custom_edge_colors:
                                edge_color = custom_edge_colors[canonical_key]
                            elif (e_start, e_end) in custom_edge_colors:
                                edge_color = custom_edge_colors[(e_start, e_end)]
                            elif (e_end, e_start) in custom_edge_colors:
                                edge_color = custom_edge_colors[(e_end, e_start)]
                            # else: edge_color stays as default_color (already set above)
                        # If no custom edge colors, edge_color remains default_color

                    elif coloring_scheme != 'Pathway Significance':
                        # Gradient mode: color by pathway node
                        pathway_node = None
                        if wsymbol[e_start] == 'd':
                            pathway_node = e_start
                        elif wsymbol[e_end] == 'd':
                            pathway_node = e_end
        
                        if pathway_node is not None:
                            rgb = current_wcolor[pathway_node]
                            edge_color = f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'
                        else:
                            rgb = (current_wcolor[e_start] + current_wcolor[e_end]) / 2
                            edge_color = f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'
                    else:
                        edge_color = default_color

                    if edge_color not in edges_by_color:
                        edges_by_color[edge_color] = {'x': [], 'y': []}
                    edges_by_color[edge_color]['x'] += list(cart_arc_1) + [None]
                    edges_by_color[edge_color]['y'] += list(cart_arc_2) + [None]

            # Add edge traces with visibility based on checkbox
            for color, coords in edges_by_color.items():
                edge_traces_new.append(go.Scatter(
                    x=coords['x'], y=coords['y'], mode='lines',
                    line=dict(width=1, color=color),
                    opacity=edge_opacity_fig1,
                    hoverinfo='none', showlegend=False,
                    visible=show_edges_fig1  # ⭐ KEY: Respect checkbox state
                ))
        
            # Clear rebuild flag
            st.session_state.edges_need_rebuild_fig1 = False

        fig1.data = []  # Clear all traces

        # Add traces in CORRECT rendering order:
        if circle_boundary_trace is not None:
            fig1.add_trace(circle_boundary_trace)      # 1. Background
    
        for edge_trace in edge_traces_new:             # 2. Edges (bottom)
            fig1.add_trace(edge_trace)
    
        for legend_trace in legend_traces_existing:    # 3. Legend
            fig1.add_trace(legend_trace)
        # ---- make legend non-overlapping, below the plot ----
        # Count legend entries you created (you can also just use len(fig1.data) filtered by showlegend)
        legend_count = sum(1 for tr in fig1.data if getattr(tr, "showlegend", False))
        place_legend_below(fig1, n_items=legend_count)   
    
        for node_trace in node_traces_existing:        # 4. Nodes (TOP)
            fig1.add_trace(node_trace)

        # Handle label toggle
        for trace in fig1.data:
            if hasattr(trace, 'mode') and 'markers' in str(trace.mode) and trace.showlegend is False:
                if show_labels_fig1:
                    trace.mode = 'markers+text'
                    trace.textposition = 'top center'
                    trace.textfont = dict(size=12)
                else:
                    trace.mode = 'markers'

        # Node size adjustment - scale original degree-based sizes
        for trace in fig1.data:
            if hasattr(trace, 'mode') and 'markers' in str(trace.mode) and trace.showlegend is False:
                # Scale the original sizes stored in customdata
                if hasattr(trace, 'customdata') and trace.customdata is not None:
                    original_sizes = [cd[0] for cd in trace.customdata]
                    trace.marker.size = [s * node_size_slider for s in original_sizes]
                else:
                    # Fallback for traces without customdata
                    trace.marker.size = 12 * node_size_slider
        
        edge_visibility_key = f"edges_visible_{show_edges_fig1}"
        if st.session_state.get("last_edge_state_fig1") != edge_visibility_key:
            st.session_state.last_edge_state_fig1 = edge_visibility_key
        
        # Use plotly_events to maintain interactivity
        with col1:
            # Add download buttons
            st.markdown("**📥 Download High-Resolution:**")
            add_download_buttons(fig1, "fig1_main", "hyperpathway_fig1")
            st.markdown("---")
            selected_points = plotly_events(
                fig1,
                click_event=False,
                select_event=True,
                override_height=800,
                key=f"main_hyperpathway_plot_1_{edge_visibility_key}"
            )

            # Download node coordinates + interactions (Visualization #1)
            if "excel_buffer_fig1" in st.session_state:
                st.download_button(
                    label="Download coordinates + list of interactions",
                    data=st.session_state.excel_buffer_fig1.getvalue(),
                    file_name="network_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with st.expander("Select pathway(s) and/or molecule(s) by name"):
                st.markdown("""
                **You can select pathway(s) and/or molecule(s) in two ways:**
    
                - **Interactive selection**: Click nodes directly on the visualization (hold `Shift` to select multiple nodes). Or use 'Box Select' on top of the visualization to select multiple nodes
                - **Name-based selection**: Use the list below or type a node name to search
                """)
                # Create display options (full name : first word)
                display_options = [f"{full_names[i]} ({fixed_names[i]})" for i in range(len(full_names))]
    
                selected_display = st.multiselect(
                    "Choose nodes..",
                    options=display_options,
                    default=[],
                    key="node_selector_fig1"
                )
    
                # Map back to indices
                selected_nodes_by_name = [i for i, opt in enumerate(display_options) if opt in selected_display]

            # Map plotly points → global indices
            selected_nodes_by_click = []
            if selected_points:
                o_nodes = [i for i, s in enumerate(wsymbol) if s == "o"]
                d_nodes = [i for i, s in enumerate(wsymbol) if s == "d"]
                index_groups = {"circle": o_nodes, "diamond": d_nodes}

                for p in selected_points:
                    curve = p["curveNumber"]
                    point = p["pointIndex"]
    
                    # ⭐ SAFETY: Check if curve index is valid
                    if curve >= len(fig1.data):
                        continue
                
                    trace = fig1.data[curve]
            
                    # ⭐ SKIP if this is not a node trace
                    if not hasattr(trace, 'marker') or not hasattr(trace.marker, 'symbol') or trace.marker.symbol is None:
                        continue
                
                    symbol = trace.marker.symbol

                    # ⭐ SAFETY CHECK: Ensure point index is valid
                    if symbol in index_groups and point < len(index_groups[symbol]):
                        global_index = index_groups[symbol][point]
                        selected_nodes_by_click.append(global_index)

            # Combine both sources, preserving existing selection if no new input
            new_selection = list(set(selected_nodes_by_name + selected_nodes_by_click))

            # Initialize selected_nodes_fig1 if it doesn't exist
            if 'selected_nodes_fig1' not in st.session_state:
                st.session_state.selected_nodes_fig1 = []

            # If plotly returned empty selection but we have name-based selection, use that
            # If both are empty, preserve previous selection (edge toggle scenario)
            if new_selection:
                st.session_state.selected_nodes_fig1 = new_selection
            elif selected_nodes_by_name:
                st.session_state.selected_nodes_fig1 = selected_nodes_by_name
            # else: keep existing selection from session state (don't update)

            # Handle subnetwork extraction
            if st.session_state.selected_nodes_fig1:
                show_labels_subfig1 = st.checkbox("Show node labels (Subfig.1)", value=False, key="show_labels_subfig1")
                show_edges_subfig1 = st.checkbox("Show edges (Subfig.1)", value=True, key="show_edges_subfig1")
                edge_opacity_subfig1 = st.slider("Edge opacity (Subfig.1)", min_value=0.1, max_value=1.0, value=0.6, step=0.1, key="edge_opacity_subfig1")
                node_size_slider_sub = create_slider(f"Subfigure 1")

                # Get current colors being used
                current_wcolor = st.session_state.get('wcolor_current_fig1', wcolor)

                mask, coords_sub, colors_sub, names_sub, shapes_sub, edge_colors_sub = filter_subgraph(
                    x, st.session_state.coords_embedding_fig1, current_wcolor,
                    full_names, wsymbol, st.session_state.selected_nodes_fig1,
                    edge_colors=edge_colors if 'edge_colors' in locals() else None 
                )

                # Use same coloring mode for subnetwork
                current_coloring_mode = 'gradient' if coloring_scheme != 'Pathway Significance' else 'default'

                fig_sub = __plot_hyperlipea_interactive(
                    mask, coords_sub, colors_sub, names_sub, shapes_sub, option, omics_type,
                    e_colors=edge_colors_sub,
                    build_edges=show_edges_subfig1,
                    show_labels=show_labels_subfig1,
                    edge_opacity=edge_opacity_subfig1,
                    corr_1_name=corr_1_name, corr_2_name=corr_2_name,
                    coloring_mode=current_coloring_mode,
                    coloring_scheme=coloring_scheme
                )
                # Determine coloring display name
                coloring_display_map = {
                    'default': 'Default',
                    'similarity': 'Similarity',
                    'hierarchy': 'Hierarchy',
                    'labels': 'Labels',
                    'preference': 'Preference',
                    'Pathway Significance': 'Pathway Significance'
                }
                coloring_display = coloring_display_map.get(coloring_scheme, coloring_scheme.capitalize())
                fig_sub.update_layout(title=f"Subnetwork of selected pathway(s) and/or molecule(s) and first‑neighbors ({coloring_display} coloring)")

                if show_labels_subfig1:
                    fig_sub.update_traces(textposition='top center', textfont_size=12)
                else:
                    fig_sub.update_traces(text='')

                # Update ONLY node sizes (not legend)
                for trace in fig_sub.data:
                    if trace.showlegend is False and hasattr(trace, 'mode') and 'markers' in trace.mode:
                        # Scale the original sizes stored in customdata
                        if hasattr(trace, 'customdata') and trace.customdata is not None:
                            original_sizes = [cd[0] for cd in trace.customdata]
                            trace.marker.size = [s * node_size_slider_sub for s in original_sizes]
                        else:
                            trace.marker.size = 12 * node_size_slider_sub

                # Download buttons for subnetwork
                st.markdown("**📥 Download Subnetwork:**")
                add_download_buttons(fig_sub, "subfig1", "hyperpathway_subfig1")
                # Display subnetwork
                st.plotly_chart(fig_sub, use_container_width=True)

            st.markdown("""
            <style>
            div[data-testid="stDownloadButton"] > button {
                background: linear-gradient(to bottom, #cccccc, #999999) !important;
                color: black !important;
                border-radius: 8px !important;
                font-weight: bold !important;
                border: 1px solid #888 !important;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.5), 0 4px 6px rgba(0, 0, 0, 0.15) !important;
                text-shadow: 0 1px 1px rgba(255,255,255,0.3) !important;
                transition: all 0.2s ease-in-out !important;
            }

            div[data-testid="stDownloadButton"] > button:hover {
                background: linear-gradient(to bottom, #dddddd, #aaaaaa) !important;
            }

            div[data-testid="stDownloadButton"] > button:active {
                transform: translateY(2px) !important;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.3), 0 2px 3px rgba(0, 0, 0, 0.15) !important;
            }
            </style>
            """, unsafe_allow_html=True)        

            # ⬇️ NEW: Place download button right after plot
# dataset 2
    x = deepcopy(x_raw)
    wname = deepcopy(wname_raw)
    wtype = deepcopy(wtype_raw)
    wname = np.array(wname)  
    wtype = np.array(wtype) 

    unique_vals = set(wtype)
    # Create a mask that keeps only wtype == 0 or wtype == 4 if no p-value provided.
    if unique_vals == {0, 4}:
        keep_mask = (wtype == 0) | (wtype == 4)
    else:
        keep_mask = (wtype == 0) | (wtype == 1) | (wtype == 2) | (wtype == 3)
    # Apply the mask
    wname = np.reshape(wname[keep_mask], (np.sum(keep_mask), 1))
    wtype = np.reshape(wtype[keep_mask], (np.sum(keep_mask), 1))
    x = x[np.reshape(keep_mask, x.shape[0]), :]
    x = x[:, np.reshape(keep_mask, x.shape[1])]

    # create the color for each type of label
    wcolor = np.zeros((len(wtype), 3))
    wcolor[np.reshape(wtype == 0, wcolor.shape[0]), :] = np.tile([0, 1, 0], (np.sum(wtype == 0), 1))
    wcolor[np.reshape(wtype == 1, wcolor.shape[0]), :] = np.tile([1, 0, 0], (np.sum(wtype == 1), 1))
    wcolor[np.reshape(wtype == 2, wcolor.shape[0]), :] = np.tile([1, 0.6, 0], (np.sum(wtype == 2), 1))
    wcolor[np.reshape(wtype == 3, wcolor.shape[0]), :] = np.tile([0.7, 0.7, 0.7], (np.sum(wtype == 3), 1))
    wcolor[np.reshape(wtype == 4, wcolor.shape[0]), :] = np.tile([0, 0, 1], (np.sum(wtype == 4), 1))
    
    fixed_names = []  # For display (first word)
    full_names = []   # For matching with DataFrame

    for i in range(len(wname)):
        original_name = wname[i][0]
        first_word = original_name.split(' ', 1)[0]  # first word for display
        fixed_names.append(first_word)
        full_names.append(original_name)  # full name for matching

    wsymbol = []
    for i in range(len(wname)):
        if wtype[i] == 0:
            wsymbol.append("o")
        else:
            wsymbol.append("d")

    # Remove isolated nodes
    x, wtype, wcolor, fixed_names, wsymbol, full_names = remove_isolated_nodes(x, wtype, wcolor, fixed_names, wsymbol, full_names)

    o_nodes = [i for i, s in enumerate(wsymbol) if s == "o"]
    d_nodes = [i for i, s in enumerate(wsymbol) if s == "d"]
    
    with col2:
        # Node Coloring Options
        st.markdown("**🎨 Node Coloring Options:**")
        # Determine available coloring options based on p-value selection
        has_pvalues = st.session_state.get("col_non_corr") is not None or \
                st.session_state.get("col_corr_1") is not None or \
                st.session_state.get("col_corr_2") is not None

        # Build coloring options list
        coloring_options = ['similarity', 'hierarchy', 'labels', 'preference']
        format_dict = {
            'hierarchy': 'Hierarchy (Radial position)',
            'similarity': 'Similarity (Angular position)',
            'labels': 'Labels (Custom categories)',
            'preference': 'Preference (Custom node/edge colors)'
        }

        if has_pvalues:
            coloring_options.append('Pathway Significance')
            format_dict['Pathway Significance'] = 'Pathway significance (requires p-values)'

        coloring_scheme = st.radio(
            "Select coloring scheme:",
            options=coloring_options,
            format_func=lambda x: format_dict[x],
            key="coloring_scheme_fig2",
            horizontal=True
        )

        # Show warning if Pathway Significance is somehow selected without p-values
        if coloring_scheme == 'Pathway Significance' and not has_pvalues:
            st.warning("⚠️ Pathway significance coloring requires p-value data. Switching to similarity coloring.")
            coloring_scheme = 'similarity'

        # Clear custom colors when switching away from preference mode
        previous_coloring_scheme = st.session_state.get('coloring_scheme_stored_fig2', None)
        if previous_coloring_scheme == 'preference' and coloring_scheme != 'preference':
            # User switched away from preference mode - clear custom colors
            if 'custom_node_colors_fig2' in st.session_state:
                del st.session_state['custom_node_colors_fig2']
            if 'custom_edge_colors_fig2' in st.session_state:
                del st.session_state['custom_edge_colors_fig2']
            # Force edge rebuild to apply new coloring
            st.session_state.edges_need_rebuild_fig2 = True

        # Initialize variables
        labels_data = None
        custom_node_color_map = None
        custom_edge_color_map = None

        # Show column selector if 'labels' selected
        if coloring_scheme == 'labels':
            st.info("📋 Select a column from your pathway enrichment table to use as labels")
    
            if 'pea_dataframe' in st.session_state:
                df = st.session_state.pea_dataframe
        
                # Let user select any column from the table
                selected_column = st.selectbox(
                    "Choose column for labels:",
                    options=df.columns.tolist(),
                    key="label_column_selector_fig2"
                )

            # 🔍 DEBUG: Show raw data samples
            #st.write("### 🔍 DEBUG: Raw Data Preview")
            #st.write(f"**Selected Column:** `{selected_column}`")
            #st.write("**First 5 rows of relevant columns:**")
            #selected_col_idx = df.columns.get_loc(selected_column)
            #col_indices = list(set([0, 1, selected_col_idx]))  # Remove duplicates with set()
            #col_indices.sort()  # Keep consistent order
            #st.dataframe(df.iloc[:5, col_indices])

            # Create a mapping dictionary for faster lookups
            # Pathway column (column 0) mapping
            pathway_label_map = {}
            molecule_label_map = {}

            for idx, row in df.iterrows():
                pathway_name = str(row.iloc[0]).strip()
                molecule_name = str(row.iloc[1]).strip()
                label_value = str(row[selected_column]).strip()

                pathway_label_map[pathway_name] = label_value
                if molecule_name not in molecule_label_map:
                    molecule_label_map[molecule_name] = label_value

            # 🔍 DEBUG: Show mapping stats
            #st.write("### 🔍 DEBUG: Mapping Statistics")
            #st.write(f"**Pathway map size:** {len(pathway_label_map)} unique pathways")
            #st.write(f"**Molecule map size:** {len(molecule_label_map)} unique molecules")
        
        
            # Build labels_data array
            labels_data = []
            unmatched_count = 0

            # 🔍 DEBUG: Track matching details
            pathway_matched = 0
            pathway_unmatched = []
            molecule_matched = 0
            molecule_unmatched = []

            #st.write("### 🔍 DEBUG: Node Matching Details")
            #st.write(f"**Total nodes to match:** {len(fixed_names)}")
        
            for i in range(len(fixed_names)):
                # Use the display name (first word) to reconstruct full name
                display_name = fixed_names[i]
                full_name = full_names[i]
            
                # Determine if this is a pathway (diamond) or molecule (circle)
                is_pathway = wsymbol[i] == 'd'
                     
                # Look up label based on node type using full name
                if is_pathway:
                    label_value = pathway_label_map.get(full_name, "Unknown")
                    if label_value != "Unknown":
                        pathway_matched += 1
                    else:
                        pathway_unmatched.append(display_name)
                else:
                    label_value = molecule_label_map.get(full_name, "Unknown")
                    if label_value != "Unknown":
                        molecule_matched += 1
                    else:
                        molecule_unmatched.append(display_name)

                if label_value == "Unknown":
                    unmatched_count += 1

                labels_data.append(label_value)

            # 🔍 DEBUG: Show detailed matching results
            #st.write(f"**Pathways (diamonds):**")
            #st.write(f"  - Matched: {pathway_matched}")
            #st.write(f"  - Unmatched: {len(pathway_unmatched)}")
        
            #st.write(f"**Molecules (circles):**")
            #st.write(f"  - Matched: {molecule_matched}")
            #st.write(f"  - Unmatched: {len(molecule_unmatched)}")
        
            # Show warnings
            if unmatched_count > 0:
                st.warning(f"⚠️ {unmatched_count} out of {len(fixed_names)} nodes could not be matched to the selected column. They will be colored gray.")
        
            # Validate labels
            unique_labels = set(labels_data) - {"Unknown"}
            if len(unique_labels) == 0:
                st.error("❌ No valid labels found in selected column. All nodes will be gray.")
                labels_data = None
            elif len(unique_labels) == 1:
                st.warning(f"⚠️ Only one unique label found: '{list(unique_labels)[0]}'. Consider selecting a different column for better visualization.")
        
        # Show file uploaders if 'preference' selected 
        elif coloring_scheme == 'preference':
            st.info("🎨 Upload custom color files for nodes and/or edges")
        
            # Node color file uploader
            st.markdown("**Optional: Upload node color file**")
            st.markdown("""
            <div style='font-size:14px; color: gray; margin-bottom: 10px;'>
            Format: 2 columns<br>
            • Column 1: Node name (exact match required)<br>
            • Column 2: Color in HEX (#FF0000), RGB space-separated (255 0 0 or 1.0 0.0 0.0)
            </div>
            """, unsafe_allow_html=True)
        
            uploaded_node_color_file = st.file_uploader(
                "Upload node colors",
                type=["xlsx", "xls", "csv", "tsv"],
                key="node_color_uploader_fig2",
                label_visibility="collapsed"
            )
        
            if uploaded_node_color_file is not None:
                custom_node_color_map = process_custom_node_colors(uploaded_node_color_file, full_names)
        
            # Edge color file uploader
            st.markdown("**Optional: Upload edge color file**")
            st.markdown("""
            <div style='font-size:14px; color: gray; margin-bottom: 10px;'>
            Format: 3 columns<br>
            • Column 1: Node 1 name<br>
            • Column 2: Node 2 name<br>
            • Column 3: Color in HEX or RGB format
            </div>
            """, unsafe_allow_html=True)
        
            uploaded_edge_color_file = st.file_uploader(
                "Upload edge colors",
                type=["xlsx", "xls", "csv", "tsv"],
                key="edge_color_uploader_fig2",
                label_visibility="collapsed"
            )
        
            if uploaded_edge_color_file is not None:
                custom_edge_color_map = process_custom_edge_colors(uploaded_edge_color_file, full_names)
        
            # Store in session state for later use
            if custom_node_color_map is not None:
                st.session_state.custom_node_colors_fig2 = custom_node_color_map
        
            if custom_edge_color_map is not None:
                st.session_state.custom_edge_colors_fig2 = custom_edge_color_map

        show_labels_fig2 = st.checkbox("Show node labels (Fig.2)", value=False, key="show_labels_fig2")
        show_edges_fig2 = st.checkbox("Show edges (Fig.2)", value=False, key="show_edges_fig2")
        edge_opacity_fig2 = st.slider("Edge opacity (Fig.2)", min_value=0.1, max_value=1.0, value=0.6, step=0.1, key="edge_opacity_fig2")
        node_size_slider = create_slider(f"Figure 2")
        st.markdown("""
        <style>
        div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
            background: linear-gradient(to bottom, #b6fcd5, #90ee90);
            color: green;
            border-radius: 8px;
            font-weight: bold;
            border: 1px solid #6cb56c;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.6), 0 4px 6px rgba(0, 0, 0, 0.2);
            text-shadow: 0 1px 1px rgba(255, 255, 255, 0.5);
            transition: all 0.2s ease-in-out;
        }

        div[data-testid="stHorizontalBlock"] > div:nth-child(2) button:hover {
            background: linear-gradient(to bottom, #c1ffe0, #a9f5bc);
        }

        div[data-testid="stHorizontalBlock"] > div:nth-child(2) button:active {
            transform: translateY(2px);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.3), 0 2px 3px rgba(0, 0, 0, 0.2);
        }
        </style>
        """, unsafe_allow_html=True)

        if st.button("Compute Hyperpathway embedding - Visualization #2", key="compute_option1_fig2"):

            try:
                # Show network size info
                n_nodes = x.shape[0]
                n_edges = x.nnz if hasattr(x, 'nnz') else np.count_nonzero(x)
                st.info(f"🔢 Network size: {n_nodes} nodes, {n_edges} edges")

                # Get the stored names
                corr_1_name = st.session_state.get('corr_1_display_name', 'Correction #1')
                corr_2_name = st.session_state.get('corr_2_display_name', 'Correction #2')
                
                # Initialize edge_colors properly
                edge_colors_to_use = None

                # Retrieve custom colors if preference mode
                if coloring_scheme == 'preference':
                    custom_node_colors = st.session_state.get('custom_node_colors_fig2', None)
                    custom_edge_colors = st.session_state.get('custom_edge_colors_fig2', None)

                    # Use custom edge colors if provided
                    if custom_edge_colors is not None:
                        edge_colors_to_use = custom_edge_colors
                # For other modes, do NOT use custom edge colors from session state
                # (edge_colors_to_use stays None, which means default gray edges)

                coords, excel_buffer, fig = run_hyperlipea_with_progress(
                    x, wtype, wcolor, full_names, wsymbol, option, omics_type,
                    e_colors=edge_colors_to_use, corr_1_name=corr_1_name, corr_2_name=corr_2_name,
                    coloring_scheme=coloring_scheme, labels_data=labels_data,
                    edge_opacity=edge_opacity_fig2
                )

                # Figure is already created with correct colors inside run_hyperlipea_with_progress
                # Just determine which colors were used for storage 
                if coloring_scheme == 'preference':
                    # For preference mode, store the custom colors that were actually used
                    custom_node_colors_used = st.session_state.get('custom_node_colors_fig2', None)
                    if custom_node_colors_used is not None:
                        wcolor_gradient = apply_gradient_coloring(
                            x, coords, wsymbol, 
                            coloring='preference',
                            custom_node_colors=custom_node_colors_used,
                            full_names=full_names
                        )
                    else:
                        # Fallback to default gray if no custom colors provided
                        wcolor_gradient = np.full((len(wsymbol), 3), 0.5)
                elif coloring_scheme != 'Pathway Significance':
                    wcolor_gradient = apply_gradient_coloring(
                        x, coords, wsymbol, 
                        coloring=coloring_scheme, 
                        labels=labels_data
                    )
                else: 
                    wcolor_gradient = wcolor

                # Store in session state
                st.session_state.edge_colors_fig2 = edge_colors_to_use  # Store edge colors used
                st.session_state.fig2 = fig
                st.session_state.coords_embedding_fig2 = coords
                st.session_state.excel_buffer_fig2 = excel_buffer
                st.session_state.wcolor_current_fig2 = wcolor_gradient # Store whichever colors were used
                st.session_state.coloring_scheme_stored_fig2 = coloring_scheme
            
            except Exception as e:
                st.error(f"Error during Hyperpathway computation: {e}")
   
    # Initialize correction names for display updates
    corr_1_name = st.session_state.get('corr_1_display_name', 'Correction #1')
    corr_2_name = st.session_state.get('corr_2_display_name', 'Correction #2')
    # When displaying existing plots, update the color scheme dynamically:
    if "fig2" in st.session_state:
        # If user changed coloring scheme, regenerate with new colors
        current_scheme = st.session_state.get('coloring_scheme_stored_fig2', 'similarity')
    
        # Only recompute colors, not the entire figure 
        if current_scheme != coloring_scheme or coloring_scheme == 'labels':
            # Clear stale selections
            if 'selected_nodes_fig2' in st.session_state:
                del st.session_state['selected_nodes_fig2']

            # Recompute ONLY colors (fast operation)
            if coloring_scheme != 'Pathway Significance':
                # Also handle preference mode for colors
                if coloring_scheme == 'preference':
                    custom_node_colors_for_recolor = st.session_state.get('custom_node_colors_fig2', None)
                    if custom_node_colors_for_recolor is not None:
                        wcolor_new = apply_gradient_coloring(
                            x, st.session_state.coords_embedding_fig2, wsymbol,
                            coloring='preference',
                            custom_node_colors=custom_node_colors_for_recolor,
                            full_names=full_names
                        )
                    else:
                        wcolor_new = np.full((len(wsymbol), 3), 0.5)
                
                else:
                    wcolor_new = apply_gradient_coloring(
                        x, st.session_state.coords_embedding_fig2, wsymbol,
                        coloring=coloring_scheme, labels=labels_data
                    )
            else:
                wcolor_new = wcolor # Original colors

            # Store new colors
            st.session_state.wcolor_current_fig2 = wcolor_new
            st.session_state.coloring_scheme_stored_fig2 = coloring_scheme
        
            # ⭐ CRITICAL: Force edge rebuild when coloring changes
            st.session_state.edges_need_rebuild_fig2 = True

        # Get current colors
        current_wcolor = st.session_state.get('wcolor_current_fig2', wcolor)
        
        # Check if opacity changed
        current_opacity = st.session_state.get('last_edge_opacity_fig2', 0.6)
        if current_opacity != edge_opacity_fig2:
            st.session_state.edges_need_rebuild_fig2 = True
            st.session_state.last_edge_opacity_fig2 = edge_opacity_fig2

        fig2 = go.Figure(st.session_state.fig2)
    
        # ⭐ DYNAMIC COLOR UPDATE: Modify node traces in-place
        o_nodes = [i for i, s in enumerate(wsymbol) if s == "o"]
        d_nodes = [i for i, s in enumerate(wsymbol) if s == "d"]

        # Update node colors for circle and diamond traces
        for trace_idx, trace in enumerate(fig2.data):
            if hasattr(trace, 'mode') and 'markers' in str(trace.mode) and trace.showlegend is False:
                symbol = trace.marker.symbol
        
                if symbol == 'circle':
                    node_indices = o_nodes
                elif symbol == 'diamond':
                    node_indices = d_nodes
                else:
                    continue
        
                # Build color list for this trace
                trace_colors = []
                for idx in node_indices:
                    if idx < len(current_wcolor):
                        rgb = current_wcolor[idx]
                        if isinstance(rgb, str):
                            trace_colors.append(rgb)
                        else:
                            trace_colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')
                    else:
                        trace_colors.append('rgb(128,128,128)')  # Fallback gray
            
                trace.marker.color = trace_colors

        # ⭐ FIX: Always remove old edges first, then rebuild if needed
        # Remove ALL edge traces
        fig2.data = [trace for trace in fig2.data 
                    if not (hasattr(trace, 'mode') and trace.mode == 'lines' and not trace.showlegend)]
    
        circle_boundary_trace = None
        legend_traces_existing = []
        node_traces_existing = []

        for trace in fig2.data:
            if hasattr(trace, 'mode'):
                if trace.mode == 'lines' and trace.showlegend is False and hasattr(trace, 'hoverinfo') and trace.hoverinfo == 'skip':
                    # This is the circle boundary
                    circle_boundary_trace = trace
                elif 'markers' in str(trace.mode) and trace.showlegend is False:
                    # This is a node trace
                    node_traces_existing.append(trace)
                elif trace.showlegend is True:
                    # This is a legend trace
                    legend_traces_existing.append(trace)

        # ---- Dynamic title + legend update (no recompute) ----
        fig2.update_layout(title=_hyperpathway_plot_title(coloring_scheme))

        # Rebuild legend traces to match current toggles (scheme / p-values)
        col_non_corr_sel = st.session_state.get('col_non_corr', None)
        col_corr_1_sel = st.session_state.get('col_corr_1', None)
        col_corr_2_sel = st.session_state.get('col_corr_2', None)

        corr_1_disp = st.session_state.get('corr_1_display_name', col_corr_1_sel or 'Correction #1')
        corr_2_disp = st.session_state.get('corr_2_display_name', col_corr_2_sel or 'Correction #2')

        legend_traces_existing = _build_hyperpathway_legend_traces(
            option=option,
            coloring_scheme=coloring_scheme,
            omics_type=omics_type,
            col_non_corr=col_non_corr_sel,
            col_corr_1=col_corr_1_sel,
            col_corr_2=col_corr_2_sel,
            corr_1_name=corr_1_disp,
            corr_2_name=corr_2_disp,
        )

        edge_traces_new = []
        needs_rebuild = (show_edges_fig2 or st.session_state.get('edges_need_rebuild_fig2', False))
        # Clear custom edge colors when switching away from preference mode
        if coloring_scheme != 'preference' and 'custom_edge_colors_fig2' in st.session_state:
            # Force rebuild to apply correct edge coloring for new mode
            needs_rebuild = True

        if needs_rebuild:
            # Build edges with current colors
            e1, e2 = triu(x, k=1).nonzero()
            max_edges = 20000
            if len(e1) > max_edges:
                idx = np.random.choice(len(e1), size=max_edges, replace=False)
                e1, e2 = e1[idx], e2[idx]

            coords_unit = __compute_plot_coords(st.session_state.coords_embedding_fig2)
            edges_by_color = {}
            default_color = '#CCCCCC'

            for e_start, e_end in zip(e1, e2):
                start_coords = coords_unit[e_start]
                end_coords = coords_unit[e_end]
    
                if np.sum(start_coords - end_coords) != 0:
                    _, _, cart_arc_1, cart_arc_2 = __hyperbolic_line_lipea(
                        start_coords.copy(), end_coords.copy(), 1, [0, 0], 'step', 0.05
                    )

                    # ⭐ CRITICAL: Transform back to native space (this was missing!)
                    pol_arc = np.empty((len(cart_arc_1), 2))
                    pol_arc[:] = np.nan
                    pol_arc[:, 0], pol_arc[:, 1] = __cart2pol(cart_arc_1, cart_arc_2)
                    pol_arc[:, 1] = 2 * np.arctanh(pol_arc[:, 1])  # Transform radius
                    cart_arc_1, cart_arc_2 = __pol2cart(pol_arc[:, 0], pol_arc[:, 1])

                    # Initialize edge_color with default
                    edge_color = default_color
        
                    if coloring_scheme == 'preference':
                        # PRIORITY 1: Check for explicit custom edge colors first
                        custom_edge_colors = st.session_state.get('custom_edge_colors_fig2', None)
                
                        if custom_edge_colors is not None:
                            canonical_key = (min(e_start, e_end), max(e_start, e_end))
                    
                            if canonical_key in custom_edge_colors:
                                edge_color = custom_edge_colors[canonical_key]
                            elif (e_start, e_end) in custom_edge_colors:
                                edge_color = custom_edge_colors[(e_start, e_end)]
                            elif (e_end, e_start) in custom_edge_colors:
                                edge_color = custom_edge_colors[(e_end, e_start)]
                            # else: edge_color stays as default_color (already set above)
                        # If no custom edge colors, edge_color remains default_color

                    elif coloring_scheme != 'Pathway Significance':
                        # Gradient mode: color by pathway node
                        pathway_node = None
                        if wsymbol[e_start] == 'd':
                            pathway_node = e_start
                        elif wsymbol[e_end] == 'd':
                            pathway_node = e_end
        
                        if pathway_node is not None:
                            rgb = current_wcolor[pathway_node]
                            edge_color = f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'
                        else:
                            rgb = (current_wcolor[e_start] + current_wcolor[e_end]) / 2
                            edge_color = f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'
                    else:
                        edge_color = default_color

                    if edge_color not in edges_by_color:
                        edges_by_color[edge_color] = {'x': [], 'y': []}
                    edges_by_color[edge_color]['x'] += list(cart_arc_1) + [None]
                    edges_by_color[edge_color]['y'] += list(cart_arc_2) + [None]

            # Add edge traces with visibility based on checkbox
            for color, coords in edges_by_color.items():
                edge_traces_new.append(go.Scatter(
                    x=coords['x'], y=coords['y'], mode='lines',
                    line=dict(width=1, color=color),
                    opacity=edge_opacity_fig2,
                    hoverinfo='none', showlegend=False,
                    visible=show_edges_fig2  # ⭐ KEY: Respect checkbox state
                ))
        
            # Clear rebuild flag
            st.session_state.edges_need_rebuild_fig2 = False

        fig2.data = []  # Clear all traces

        # Add traces in CORRECT rendering order:
        if circle_boundary_trace is not None:
            fig2.add_trace(circle_boundary_trace)      # 1. Background
    
        for edge_trace in edge_traces_new:             # 2. Edges (bottom)
            fig2.add_trace(edge_trace)
    
        for legend_trace in legend_traces_existing:    # 3. Legend
            fig2.add_trace(legend_trace)
        # ---- make legend non-overlapping, below the plot ----
        # Count legend entries you created (you can also just use len(fig1.data) filtered by showlegend)
        legend_count = sum(1 for tr in fig2.data if getattr(tr, "showlegend", False))
        place_legend_below(fig2, n_items=legend_count)

        for node_trace in node_traces_existing:        # 4. Nodes (TOP)
            fig2.add_trace(node_trace)

        # Handle label toggle
        for trace in fig2.data:
            if hasattr(trace, 'mode') and 'markers' in str(trace.mode) and trace.showlegend is False:
                if show_labels_fig2:
                    trace.mode = 'markers+text'
                    trace.textposition = 'top center'
                    trace.textfont = dict(size=12)
                else:
                    trace.mode = 'markers'

        # Node size adjustment - scale original degree-based sizes
        for trace in fig2.data:
            if hasattr(trace, 'mode') and 'markers' in str(trace.mode) and trace.showlegend is False:
                # Scale the original sizes stored in customdata
                if hasattr(trace, 'customdata') and trace.customdata is not None:
                    original_sizes = [cd[0] for cd in trace.customdata]
                    trace.marker.size = [s * node_size_slider for s in original_sizes]
                else:
                    # Fallback for traces without customdata
                    trace.marker.size = 12 * node_size_slider
        
        edge_visibility_key = f"edges_visible_{show_edges_fig2}"
        if st.session_state.get("last_edge_state_fig2") != edge_visibility_key:
            st.session_state.last_edge_state_fig2 = edge_visibility_key
        
        with col2:
            # Add download buttons for Figure 2
            st.markdown("**📥 Download High-Resolution:**")
            add_download_buttons(fig2, "fig2_main", "hyperpathway_fig2")
            st.markdown("---")

            selected_points = plotly_events(
                fig2,
                click_event=False,
                select_event=True,
                override_height=800,
                key=f"main_hyperpathway_plot_2_{edge_visibility_key}"
            )

            # Download node coordinates + interactions (Visualization #2)
            if "excel_buffer_fig2" in st.session_state:
                st.download_button(
                    label="Download coordinates + list of interactions",
                    data=st.session_state.excel_buffer_fig2.getvalue(),
                    file_name="extended_network_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with st.expander("Select pathway(s) and/or molecule(s) by name"):
                st.markdown("""
                **You can select pathway(s) and/or molecule(s) in two ways:**
    
                - **Interactive selection**: Click nodes directly on the visualization (hold `Shift` to select multiple nodes). Or use 'Box Select' on top of the visualization to select multiple nodes
                - **Name-based selection**: Use the list below or type a node name to search
                """)
                # Create display options (full name : first word)
                display_options = [f"{full_names[i]} ({fixed_names[i]})" for i in range(len(full_names))]
    
                selected_display = st.multiselect(
                    "Choose nodes..",
                    options=display_options,
                    default=[],
                    key="node_selector_fig2"
                )
    
                # Map back to indices
                selected_nodes_by_name = [i for i, opt in enumerate(display_options) if opt in selected_display]

            # Map plotly points → global indices
            selected_nodes_by_click = []
            if selected_points:
                o_nodes = [i for i, s in enumerate(wsymbol) if s == "o"]
                d_nodes = [i for i, s in enumerate(wsymbol) if s == "d"]
                index_groups = {"circle": o_nodes, "diamond": d_nodes}

                for p in selected_points:
                    curve = p["curveNumber"]
                    point = p["pointIndex"]
    
                    # ⭐ SAFETY: Check if curve index is valid
                    if curve >= len(fig2.data):
                        continue
                
                    trace = fig2.data[curve]
            
                    # ⭐ SKIP if this is not a node trace
                    if not hasattr(trace, 'marker') or not hasattr(trace.marker, 'symbol') or trace.marker.symbol is None:
                        continue
                
                    symbol = trace.marker.symbol

                    # ⭐ SAFETY CHECK: Ensure point index is valid
                    if symbol in index_groups and point < len(index_groups[symbol]):
                        global_index = index_groups[symbol][point]
                        selected_nodes_by_click.append(global_index)

            # Combine both sources, preserving existing selection if no new input
            new_selection = list(set(selected_nodes_by_name + selected_nodes_by_click))

            # Initialize selected_nodes_fig1 if it doesn't exist
            if 'selected_nodes_fig2' not in st.session_state:
                st.session_state.selected_nodes_fig2 = []

            # If plotly returned empty selection but we have name-based selection, use that
            # If both are empty, preserve previous selection (edge toggle scenario)
            if new_selection:
                st.session_state.selected_nodes_fig2 = new_selection
            elif selected_nodes_by_name:
                st.session_state.selected_nodes_fig2 = selected_nodes_by_name
            # else: keep existing selection from session state (don't update)

            # Handle subnetwork extraction
            if st.session_state.selected_nodes_fig2:
                show_labels_subfig2 = st.checkbox("Show node labels (Subfig.2)", value=False, key="show_labels_subfig2")
                show_edges_subfig2 = st.checkbox("Show edges (Subfig.2)", value=True, key="show_edges_subfig2")
                edge_opacity_subfig2 = st.slider("Edge opacity (Subfig.2)", min_value=0.1, max_value=1.0, value=st.session_state.get("edge_opacity_fig2", 0.6), step=0.1, key="edge_opacity_subfig2")
                node_size_slider_sub = create_slider(f"Subfigure 2")

                # Get current colors being used
                current_wcolor = st.session_state.get('wcolor_current_fig2', wcolor)

                mask, coords_sub, colors_sub, names_sub, shapes_sub, edge_colors_sub = filter_subgraph(
                    x, st.session_state.coords_embedding_fig2, current_wcolor,
                    full_names, wsymbol, st.session_state.selected_nodes_fig2,
                    edge_colors=edge_colors if 'edge_colors' in locals() else None 
                )

                # Use same coloring mode for subnetwork
                current_coloring_mode = 'gradient' if coloring_scheme != 'Pathway Significance' else 'default'

                fig_sub = __plot_hyperlipea_interactive(
                    mask, coords_sub, colors_sub, names_sub, shapes_sub, option, omics_type,
                    e_colors=edge_colors_sub,
                    build_edges=show_edges_subfig2,
                    show_labels=show_labels_subfig2,
                    edge_opacity=edge_opacity_subfig2,
                    corr_1_name=corr_1_name, corr_2_name=corr_2_name,
                    coloring_mode=current_coloring_mode,
                    coloring_scheme=coloring_scheme
                )
                # Determine coloring display name
                coloring_display_map = {
                    'default': 'Default',
                    'similarity': 'Similarity',
                    'hierarchy': 'Hierarchy',
                    'labels': 'Labels',
                    'preference': 'Preference',
                    'Pathway Significance': 'Pathway Significance'
                }
                coloring_display = coloring_display_map.get(coloring_scheme, coloring_scheme.capitalize())
                fig_sub.update_layout(title=f"Subnetwork of selected pathway(s) and/or molecule(s) and first‑neighbors ({coloring_display} coloring)")

                if show_labels_subfig2:
                    fig_sub.update_traces(textposition='top center', textfont_size=12)
                else:
                    fig_sub.update_traces(text='')

                # Update ONLY node sizes (not legend)
                for trace in fig_sub.data:
                    if trace.showlegend is False and hasattr(trace, 'mode') and 'markers' in trace.mode:
                        # Scale the original sizes stored in customdata
                        if hasattr(trace, 'customdata') and trace.customdata is not None:
                            original_sizes = [cd[0] for cd in trace.customdata]
                            trace.marker.size = [s * node_size_slider_sub for s in original_sizes]
                        else:
                            trace.marker.size = 12 * node_size_slider_sub

                # Download buttons for subnetwork
                st.markdown("**📥 Download Subnetwork:**")
                add_download_buttons(fig_sub, "subfig2", "hyperpathway_subfig2")
                st.markdown("---")

                # ⭐ Display subnetwork with plotly_chart (still interactive for zooming/panning)
                st.plotly_chart(fig_sub, use_container_width=True)

            st.markdown("""
            <style>
            div[data-testid="stDownloadButton"] > button {
                background: linear-gradient(to bottom, #cccccc, #999999) !important;
                color: black !important;
                border-radius: 8px !important;
                font-weight: bold !important;
                border: 1px solid #888 !important;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.5), 0 4px 6px rgba(0, 0, 0, 0.15) !important;
                text-shadow: 0 1px 1px rgba(255,255,255,0.3) !important;
                transition: all 0.2s ease-in-out !important;
            }

            div[data-testid="stDownloadButton"] > button:hover {
                background: linear-gradient(to bottom, #dddddd, #aaaaaa) !important;
            }

            div[data-testid="stDownloadButton"] > button:active {
                transform: translateY(2px) !important;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.3), 0 2px 3px rgba(0, 0, 0, 0.15) !important;
            }
            </style>
            """, unsafe_allow_html=True)        

            # ⬇️ NEW: Place download button right after plot
# After both visualizations

    col_space1, col_center, col_space2 = st.columns([2, 2, 2])
    with col_center:
        st.markdown("""
        <style>
        /* Target the second column of the last horizontal block */
        div[data-testid="stHorizontalBlock"]:last-of-type > div:nth-child(2) button {
            background: linear-gradient(to bottom, #cccccc, #999999);  /* Very light black gradient */
            color: black;
            border-radius: 8px;
            font-weight: bold;
            border: 1px solid #888;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.5), 0 4px 6px rgba(0, 0, 0, 0.15);
            text-shadow: 0 1px 1px rgba(255,255,255,0.3);
            transition: all 0.2s ease-in-out;
        }

        div[data-testid="stHorizontalBlock"]:last-of-type > div:nth-child(2) button:hover {
            background: linear-gradient(to bottom, #dddddd, #aaaaaa);
        }

        div[data-testid="stHorizontalBlock"]:last-of-type > div:nth-child(2) button:active {
            transform: translateY(2px);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.3), 0 2px 3px rgba(0, 0, 0, 0.15);
        }
        </style>
        """, unsafe_allow_html=True)     
        if "fig1" in st.session_state or "fig2" in st.session_state:

            if st.button("➕ New Visualization", key="new_viz_option1"):
                # Clear session state or redirect to input
                st.session_state.clear()
                #st.session_state.selection_omics = "-- Select omics type --"
                st.rerun()

elif uploaded_bipartite_file:
    # Handle bipartite file loading$
    try:
        option = 2
        x, edge_color, wname, wsymbol = process_adjacency_list(uploaded_bipartite_file)
        st.success("Bipartite network file successfully uploaded!")
    except Exception as e:
        st.error(f"Error loading bipartite network file: {e}")

    # --- THIRD uploader: for list of nodes in bipartite network ---
    st.markdown(
    """
    <div style='font-size:20px; font-weight: bold; margin-bottom: -2px; line-height: 1.2;'>
        Upload your node list file (optional).
    </div>
    """,
    unsafe_allow_html=True
)
    uploaded_node_file = st.file_uploader(
    "Upload node file",
    type=["xlsx", "xls", "csv", "tsv"],
    key="node_uploader",
    label_visibility="collapsed"
    )

    if uploaded_node_file is not None:
        current_file_id = f"{uploaded_node_file.name}_{uploaded_node_file.size}"
        if st.session_state.get("last_node_file_id") != current_file_id:
            # New file detected - clear previous visualizations
            keys_to_clear = ["fig1", "coords_embedding_fig1", "excel_buffer_fig1", "selected_nodes_fig1",
                             "fig2", "option2_coords_fig2", "excel_buffer_fig2", "selected_nodes_option2_fig2"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state["last_node_file_id"] = current_file_id
    else:
        # File was removed - clear all visualizations
        if st.session_state.get("last_node_file_id") is not None:
            keys_to_clear = ["fig1", "coords_embedding_fig1", "excel_buffer_fig1", "selected_nodes_fig1",
                             "fig2", "option2_coords_fig2", "excel_buffer_fig2", "selected_nodes_option2_fig2",
                             "wcolor_base_option2_fig1", "wcolor_base_option2_fig2", "last_node_file_id"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]

    # Initialize wtype (always needed)
    wtype = [0] * len(wname)
    wtype = np.array(wtype)
    
    # Initialize fixed_names and full_names
    fixed_names = wname  # For display (first word)
    full_names = wname   # For matching with DataFrame

    # Initialize node_color to None before conditional block
    node_color = None
    
    if uploaded_node_file:
        try:
            list_unique_nodes, node_color = process_list_nodes(uploaded_node_file, wname, wsymbol)
            st.success("Node list file successfully uploaded!")
            
            if "wcolor_base_option2_fig1" not in st.session_state and node_color is not None:
                st.session_state.wcolor_base_option2_fig1 = node_color
        except Exception as e:
            st.error(f"Error loading node list file: {e}")

        # -----------------------------
    col1, col2 = st.columns(2)
    
    # ==================== COLUMN 1 (Fig.1) ====================
    with col1:
            # Option 2 — Node Coloring Options (same as Option 1)
            # -----------------------------
            st.markdown("**🎨 Node Coloring Options:**")

            # Option 2 usually has no p-values unless you explicitly provide them,
            # so keep pathway significance disabled here.
            has_pvalues = False

            # Detect if colors are present in uploaded files
            has_edge_colors = edge_color and isinstance(edge_color, dict) and len(edge_color) > 0
            has_node_colors = 'wcolor_base_option2_fig1' in st.session_state
            
            # Build coloring options based on what's available
            if has_edge_colors or has_node_colors:
                coloring_options = ['default', 'similarity', 'hierarchy']
                format_dict = {
                    'default': 'Default (Colors from files)',
                    'hierarchy': 'Hierarchy (Radial position)',
                    'similarity': 'Similarity (Angular position)',
                }
                default_scheme = 'default'
            else:
                coloring_options = ['similarity', 'hierarchy']
                format_dict = {
                    'hierarchy': 'Hierarchy (Radial position)',
                    'similarity': 'Similarity (Angular position)',
                }
                default_scheme = 'similarity'

            # Get the default index
            default_index = coloring_options.index(default_scheme)
            
            coloring_scheme = st.radio(
                "Select coloring scheme:",
                options=coloring_options,
                format_func=lambda x: format_dict[x],
                key="coloring_scheme_option2_fig1",
                horizontal=True,
                index=default_index
            )

            labels_data = None
            custom_node_color_map = None
            custom_edge_color_map = None

            if coloring_scheme == 'default':
                st.info("📋 Using default colors from uploaded files (edge and/or node colors)")
            else:
                labels_data = None

            show_labels_option2_fig1 = st.checkbox("Show node labels (Fig.1)", value=False, key="show_labels_option2_fig1")
            show_edges_option2_fig1 = st.checkbox("Show edges (Fig.1)", value=False, key="show_edges_option2_fig1")
            edge_opacity_option2_fig1 = st.slider("Edge opacity (Fig.1)", min_value=0.1, max_value=1.0, value=0.6, step=0.1, key="edge_opacity_option2_fig1")
            node_size_slider = create_slider(f"Figure 1")

            col1, = st.columns(1)
            with col1:
                st.markdown("""
                <style>
                div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
                    background: linear-gradient(to bottom, #b6fcd5, #90ee90);
                    color: green;
                    border-radius: 8px;
                    font-weight: bold;
                    border: 1px solid #6cb56c;
                    box-shadow: inset 0 1px 0 rgba(255,255,255,0.6), 0 4px 6px rgba(0, 0, 0, 0.2);
                    text-shadow: 0 1px 1px rgba(255, 255, 255, 0.5);
                    transition: all 0.2s ease-in-out;
                }

                div[data-testid="stHorizontalBlock"] > div:nth-child(1) button:hover {
                    background: linear-gradient(to bottom, #c1ffe0, #a9f5bc);
                }

                div[data-testid="stHorizontalBlock"] > div:nth-child(1) button:active {
                    transform: translateY(2px);
                    box-shadow: inset 0 1px 0 rgba(255,255,255,0.3), 0 2px 3px rgba(0, 0, 0, 0.2);
                }
                </style>
                """, unsafe_allow_html=True)
                if st.button("Compute hyperbolic embedding", key="compute_embedding_option2_fig1"):
                    
                    try:
                        # Show network size info
                        n_nodes = x.shape[0]
                        n_edges = x.nnz if hasattr(x, 'nnz') else np.count_nonzero(x)
                        st.info(f"🔢 Network size: {n_nodes} nodes, {n_edges} edges")
                        
                        # Initialize edge_colors properly
                        edge_colors_to_use = None 
                        
                        # Handle default mode - use edge colors from adjacency list if available
                        if coloring_scheme == 'default':
                            if has_edge_colors:
                                edge_colors_to_use = edge_color
                        
                        # Determine node_color to use based on coloring scheme - MUST DO THIS BEFORE PLOTTING
                        if coloring_scheme == 'default':
                            # Use colors from uploaded files
                            if has_node_colors:
                                node_color_to_use = st.session_state.wcolor_base_option2_fig1
                            else:
                                # NEW: If no node colors but edge colors exist, use black for diamonds, red for circles
                                if has_edge_colors:
                                    node_color_to_use = np.zeros((len(wsymbol), 3))
                                    for i, shape in enumerate(wsymbol):
                                        if shape == 'd':  # diamond
                                            node_color_to_use[i] = [0, 0, 0]  # black
                                        else:  # circle
                                            node_color_to_use[i] = [1, 0, 0]  # red
                                else:
                                    # Fallback: generate gradient colors if no node colors and no edge colors provided
                                    node_color_to_use = None
                        elif node_color is not None:
                            node_color_to_use = node_color
                        else:
                            # For non-default modes, pass None and let the plotting function handle it
                            node_color_to_use = None
                        
                        coords, excel_buffer, fig = run_hyperlipea_with_progress(
                            x, wtype, node_color_to_use, full_names, wsymbol, option, omics_type,
                            e_colors=edge_colors_to_use, coloring_scheme=coloring_scheme,
                            labels_data=labels_data, edge_opacity=edge_opacity_option2_fig1
                        )

                        # Store the final node colors used for session state
                        if coloring_scheme != 'Pathway Significance' and coloring_scheme != 'default':
                            wcolor_used = apply_gradient_coloring(x, coords, wsymbol, coloring=coloring_scheme, labels=labels_data)
                        elif node_color_to_use is not None:
                            wcolor_used = node_color_to_use
                        else:
                            # Fallback to gradient
                            wcolor_used = apply_gradient_coloring(x, coords, wsymbol, coloring='similarity')

                        st.session_state.edge_colors_option2_fig1 = edge_colors_to_use  # Store edge colors used
                        st.session_state.option2_fig1 = fig
                        st.session_state.option2_coords_fig1 = coords
                        st.session_state.option2_excel_buffer_fig1 = excel_buffer
                        st.session_state.wcolor_current_option2_fig1 = wcolor_used
                        if coloring_scheme == 'default':
                            st.session_state.wcolor_base_option2_fig1 = wcolor_used
                        st.session_state.coloring_scheme_stored_option2_fig1 = coloring_scheme
                    except Exception as e:
                        st.error(f"Error during hyperbolic embedding computation: {e}")

            # If embedding already computed, apply UI-driven updates and display
            if "option2_fig1" in st.session_state:
                # If user changed coloring scheme, regenerate with new colors
                current_scheme = st.session_state.get('coloring_scheme_stored_option2_fig1', 'similarity')
    
                # Only recompute colors, not the entire figure 
                if current_scheme != coloring_scheme or coloring_scheme == 'labels':
                    # Clear stale selections
                    if 'selected_nodes_option2_fig1' in st.session_state:
                        del st.session_state['selected_nodes_option2_fig1']

                    # Handle different coloring modes
                    if coloring_scheme == 'default':
                        # Use colors from uploaded files
                        if has_node_colors:
                            wcolor_new = st.session_state.wcolor_base_option2_fig1
                        else:
                            # NEW: If no node colors but edge colors exist, use black for diamonds, red for circles
                            if has_edge_colors:
                                wcolor_new = np.zeros((len(wsymbol), 3))
                                for i, shape in enumerate(wsymbol):
                                    if shape == 'd':  # diamond
                                        wcolor_new[i] = [0, 0, 0]  # black
                                    else:  # circle
                                        wcolor_new[i] = [1, 0, 0]  # red
                            else:
                                # Fallback to gradient if no node colors and no edge colors provided
                                wcolor_new = apply_gradient_coloring(
                                    x, st.session_state.option2_coords_fig1, wsymbol,
                                    coloring='similarity'
                                )             
                    else:
                        wcolor_new = apply_gradient_coloring(
                            x, st.session_state.option2_coords_fig1, wsymbol,
                            coloring=coloring_scheme, labels=labels_data
                        )

                    # Store new colors
                    st.session_state.wcolor_current_option2_fig1 = wcolor_new
                    st.session_state.coloring_scheme_stored_option2_fig1 = coloring_scheme
        
                    # ⭐ CRITICAL: Force edge rebuild when coloring changes
                    st.session_state.edges_need_rebuild_option2_fig1 = True

                # Get current colors
                base_wcolor = st.session_state.get("wcolor_base_option2_fig1", None)
                current_wcolor = st.session_state.get("wcolor_current_option2_fig1", base_wcolor) 
                
                # If still None, generate default gradient colors
                if current_wcolor is None:
                    current_wcolor = apply_gradient_coloring(
                        x, st.session_state.option2_coords_fig1, wsymbol,
                        coloring='similarity'
                    ) 

                # Check if opacity changed
                current_opacity = st.session_state.get('last_edge_opacity_option2_fig1', 0.6)
                if current_opacity != edge_opacity_option2_fig1:
                    st.session_state.edges_need_rebuild_option2_fig1 = True
                    st.session_state.last_edge_opacity_option2_fig1 = edge_opacity_option2_fig1
            
                # Make a deep copy so we don't mutate the cached object
                fig1 = go.Figure(st.session_state.option2_fig1)

                # ⭐ DYNAMIC COLOR UPDATE: Modify node traces in-place
                o_nodes = [i for i, s in enumerate(wsymbol) if s == "o"]
                d_nodes = [i for i, s in enumerate(wsymbol) if s == "d"]
                
                # Update node colors for circle and diamond traces
                for trace_idx, trace in enumerate(fig1.data):
                    if hasattr(trace, 'mode') and 'markers' in str(trace.mode) and trace.showlegend is False:
                        symbol = trace.marker.symbol
        
                        if symbol == 'circle':
                            node_indices = o_nodes
                        elif symbol == 'diamond':
                            node_indices = d_nodes
                        else:
                            continue
        
                        # Build color list for this trace
                        trace_colors = []
                        for idx in node_indices:
                            if idx < len(current_wcolor):
                                rgb = current_wcolor[idx]
                                if isinstance(rgb, str):
                                    trace_colors.append(rgb)
                                else:
                                    trace_colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')
                            else:
                                trace_colors.append('rgb(128,128,128)')  # Fallback gray
            
                        trace.marker.color = trace_colors
            
                # ⭐ FIX: Always remove old edges first, then rebuild if needed
                # Remove ALL edge traces
                fig1.data = [trace for trace in fig1.data 
                    if not (hasattr(trace, 'mode') and trace.mode == 'lines' and not trace.showlegend)]
    
                circle_boundary_trace = None
                legend_traces_existing = []
                node_traces_existing = []
            
                for trace in fig1.data:
                    if hasattr(trace, 'mode'):
                        if trace.mode == 'lines' and trace.showlegend is False and hasattr(trace, 'hoverinfo') and trace.hoverinfo == 'skip':
                            # This is the circle boundary
                            circle_boundary_trace = trace
                        elif 'markers' in str(trace.mode) and trace.showlegend is False:
                            # This is a node trace
                            node_traces_existing.append(trace)
                        elif trace.showlegend is True:
                            # This is a legend trace
                            legend_traces_existing.append(trace)

                # ---- Dynamic title + legend update (no recompute) ----
                fig1.update_layout(title=_hyperpathway_plot_title(coloring_scheme))

                legend_traces_existing = _build_hyperpathway_legend_traces(
                    option=option,
                    coloring_scheme=coloring_scheme,
                    omics_type=omics_type,
                )

                edge_traces_new = []
                needs_rebuild = (show_edges_option2_fig1 or st.session_state.get('edges_need_rebuild_option2_fig1', False))
                # Clear custom edge colors when switching away from preference mode
                if coloring_scheme != 'preference' and 'custom_edge_colors_option2_fig1' in st.session_state:
                    # Force rebuild to apply correct edge coloring for new mode
                    needs_rebuild = True

                if needs_rebuild:
                    # Build edges with current colors
                    e1, e2 = triu(x, k=1).nonzero()
                    max_edges = 20000
                    if len(e1) > max_edges:
                        idx = np.random.choice(len(e1), size=max_edges, replace=False)
                        e1, e2 = e1[idx], e2[idx]

                    coords_unit = __compute_plot_coords(st.session_state.option2_coords_fig1)
                    edges_by_color = {}
                    default_color = '#CCCCCC'

                    for e_start, e_end in zip(e1, e2):
                        start_coords = coords_unit[e_start]
                        end_coords = coords_unit[e_end]
    
                        if np.sum(start_coords - end_coords) != 0:
                            _, _, cart_arc_1, cart_arc_2 = __hyperbolic_line_lipea(
                                start_coords.copy(), end_coords.copy(), 1, [0, 0], 'step', 0.05
                            )

                            # ⭐ CRITICAL: Transform back to native space (this was missing!)
                            pol_arc = np.empty((len(cart_arc_1), 2))
                            pol_arc[:] = np.nan
                            pol_arc[:, 0], pol_arc[:, 1] = __cart2pol(cart_arc_1, cart_arc_2)
                            pol_arc[:, 1] = 2 * np.arctanh(pol_arc[:, 1])  # Transform radius
                            cart_arc_1, cart_arc_2 = __pol2cart(pol_arc[:, 0], pol_arc[:, 1])

                            # Initialize edge color string with default
                            edge_color_str = default_color
        
                            if coloring_scheme == 'default':
                                # PRIORITY 1: Use edge colors from adjacency list if available
                                if has_edge_colors:
                                    canonical_key = (min(e_start, e_end), max(e_start, e_end))
                                    
                                    if canonical_key in edge_color:
                                        edge_color_str = edge_color[canonical_key]
                                    elif (e_start, e_end) in edge_color:
                                        edge_color_str = edge_color[(e_start, e_end)]
                                    elif (e_end, e_start) in edge_color:
                                        edge_color_str = edge_color[(e_end, e_start)]
                                    # else: edge_color_str stays as default_color
                                else:
                                    # No edge colors in file, keep default gray color
                                    edge_color_str = default_color                                 
                            else:  
                                # Gradient mode: color by pathway node
                                pathway_node = None
                                if wsymbol[e_start] == 'd':
                                    pathway_node = e_start
                                elif wsymbol[e_end] == 'd':
                                    pathway_node = e_end
        
                                if pathway_node is not None:
                                    rgb = current_wcolor[pathway_node]
                                    edge_color_str = f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'
                                else:
                                    rgb = (current_wcolor[e_start] + current_wcolor[e_end]) / 2
                                    edge_color_str = f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'

                            if edge_color_str not in edges_by_color:
                                edges_by_color[edge_color_str] = {'x': [], 'y': []}
                            edges_by_color[edge_color_str]['x'] += list(cart_arc_1) + [None]
                            edges_by_color[edge_color_str]['y'] += list(cart_arc_2) + [None]

                    # Add edge traces with visibility based on checkbox
                    for color, coords in edges_by_color.items():
                        edge_traces_new.append(go.Scatter(
                            x=coords['x'], y=coords['y'], mode='lines',
                            line=dict(width=1, color=color),
                            opacity=edge_opacity_option2_fig1,
                            hoverinfo='none', showlegend=False,
                            visible=show_edges_option2_fig1  # ⭐ KEY: Respect checkbox state
                        ))
        
                    # Clear rebuild flag
                    st.session_state.edges_need_rebuild_option2_fig1 = False

                fig1.data = []  # Clear all traces

                # Add traces in CORRECT rendering order:
                if circle_boundary_trace is not None:
                    fig1.add_trace(circle_boundary_trace)      # 1. Background
    
                for edge_trace in edge_traces_new:             # 2. Edges (bottom)
                    fig1.add_trace(edge_trace)
    
                for legend_trace in legend_traces_existing:    # 3. Legend
                    fig1.add_trace(legend_trace)
                # ---- make legend non-overlapping, below the plot ----
                # Count legend entries you created (you can also just use len(fig1.data) filtered by showlegend)
                legend_count = sum(1 for tr in fig1.data if getattr(tr, "showlegend", False))
                place_legend_below(fig1, n_items=legend_count)
    
                for node_trace in node_traces_existing:        # 4. Nodes (TOP)
                    fig1.add_trace(node_trace)

                # Labels toggle - change mode instead of clearing text
                for trace in fig1.data:
                    if hasattr(trace, 'mode') and 'markers' in str(trace.mode) and trace.showlegend is False:
                        if show_labels_option2_fig1:
                            trace.mode = 'markers+text'
                            trace.textposition = 'top center'
                            trace.textfont = dict(size=12)
                        else:
                            trace.mode = 'markers' # Remove text from mode, keeps markers only
            
                # Node size adjustment - scale original degree-based sizes
                for trace in fig1.data:
                    if hasattr(trace, 'mode') and 'markers' in str(trace.mode) and trace.showlegend is False:
                        # Scale the original sizes stored in customdata
                        if hasattr(trace, 'customdata') and trace.customdata is not None:
                            original_sizes = [cd[0] for cd in trace.customdata]
                            trace.marker.size = [s * node_size_slider for s in original_sizes]
                        else:
                            # Fallback for traces without customdata
                            trace.marker.size = 12 * node_size_slider

                edge_visibility_key = f"edges_visible_{show_edges_option2_fig1}"
                if st.session_state.get("last_edge_state_option2_fig1") != edge_visibility_key:
                    st.session_state.last_edge_state_option2_fig1 = edge_visibility_key

                # Add download buttons for Figure 2
                st.markdown("**📥 Download High-Resolution:**")
                add_download_buttons(fig1, "fig1_main", "hyperpathway_fig1")
                st.markdown("---")

                selected_points = plotly_events(
                    fig1,
                    click_event=False,
                    select_event=True,
                    override_height=800,
                    key=f"col1_option2_plot_{edge_visibility_key}"
                )

                # Download node coordinates + interactions (Visualization #2)
                if "option2_excel_buffer_fig1" in st.session_state:
                    st.download_button(
                        label="Download coordinates + list of interactions",
                        data=st.session_state.option2_excel_buffer_fig1.getvalue(),
                        file_name="network_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_option2_col1_fig1"
                    )
                with st.expander("Select any node(s) by name"):
                    st.markdown("""
                    **You can select any node(s) in two ways:**
    
                    - **Interactive selection**: Click nodes directly on the visualization (hold `Shift` to select multiple nodes). Or use 'Box Select' on top of the visualization to select multiple nodes
                    - **Name-based selection**: Use the list below or type a node name to search
                    """)
                    # Create display options (full name : first word)
                    display_options = [f"{full_names[i]} ({fixed_names[i]})" for i in range(len(full_names))]

                    selected_display = st.multiselect(
                        "Choose nodes..",
                        options=display_options,
                        default=[],
                        key="node_selector_option2_fig1"
                    )

                    # Map back to indices
                    selected_nodes_by_name = [i for i, opt in enumerate(display_options) if opt in selected_display]

                # Map plotly points → global indices
                selected_nodes_by_click = []
                if selected_points:
                    o_nodes = [i for i, s in enumerate(wsymbol) if s == "o"]
                    d_nodes = [i for i, s in enumerate(wsymbol) if s == "d"]
                    index_groups = {"circle": o_nodes, "diamond": d_nodes}

                    for p in selected_points:
                        curve = p["curveNumber"]
                        point = p["pointIndex"]
    
                        # ⭐ SAFETY: Check if curve index is valid
                        if curve >= len(fig1.data):
                            continue
                
                        trace = fig1.data[curve]
            
                        # ⭐ SKIP if this is not a node trace
                        if not hasattr(trace, 'marker') or not hasattr(trace.marker, 'symbol') or trace.marker.symbol is None:
                            continue
                
                        symbol = trace.marker.symbol

                        # ⭐ SAFETY CHECK: Ensure point index is valid
                        if symbol in index_groups and point < len(index_groups[symbol]):
                            global_index = index_groups[symbol][point]
                            selected_nodes_by_click.append(global_index)
            
                # Combine both sources, preserving existing selection if no new input
                new_selection = list(set(selected_nodes_by_name + selected_nodes_by_click))

                # Initialize selected_nodes_fig1 if it doesn't exist
                if 'selected_nodes_option2_fig1' not in st.session_state:
                    st.session_state.selected_nodes_option2_fig1 = []

                # If plotly returned empty selection but we have name-based selection, use that
                # If both are empty, preserve previous selection (edge toggle scenario)
                if new_selection:
                    st.session_state.selected_nodes_option2_fig1 = new_selection
                elif selected_nodes_by_name:
                    st.session_state.selected_nodes_option2_fig1 = selected_nodes_by_name
                # else: keep existing selection from session state (don't update)

                # Handle subnetwork extraction
                if st.session_state.selected_nodes_option2_fig1:
                    show_labels_option2_subfig1 = st.checkbox("Show node labels (Subfig.1)", value=False, key="show_labels_option2_subfig1")
                    show_edges_option2_subfig1 = st.checkbox("Show edges (Subfig.1)", value=True, key="show_edges_option2_subfig1")
                    edge_opacity_option2_subfig1 = st.slider("Edge opacity (Subfig.1)", min_value=0.1, max_value=1.0, value=st.session_state.get("edge_opacity_fig1", 0.6), step=0.1, key="edge_opacity_option2_subfig1")
                    node_size_slider_sub = create_slider(f"Subfigure 1")

                    # Get current colors being used
                    base_wcolor = st.session_state.get("wcolor_base_option2_fig1", None)
                    current_wcolor = st.session_state.get("wcolor_current_option2_fig1", base_wcolor)
                    
                    # If still None, generate default gradient colors
                    if current_wcolor is None:
                        current_wcolor = apply_gradient_coloring(
                            x, st.session_state.option2_coords_fig1, wsymbol,
                            coloring='similarity'
                        )      

                    # Determine which edge colors to pass based on coloring scheme
                    edge_colors_to_pass = None
                    if coloring_scheme == 'default' and has_edge_colors:
                        # Pass edge colors from adjacency file for default mode
                        edge_colors_to_pass = edge_color
                    elif coloring_scheme == 'preference':
                        # Pass custom edge colors if in preference mode
                        edge_colors_to_pass = st.session_state.get('custom_edge_colors_option2_fig1', None)
                    
                    mask, coords_sub, colors_sub, names_sub, shapes_sub, edge_colors_sub = filter_subgraph(
                        x, st.session_state.option2_coords_fig1, current_wcolor,
                        full_names, wsymbol, st.session_state.selected_nodes_option2_fig1,
                        edge_colors=edge_colors_to_pass
                    )

                    # Determine coloring mode for subnetwork based on scheme
                    if coloring_scheme == 'default':
                        # In default mode, use 'default' mode so edges use e_colors dict
                        current_coloring_mode = 'default'
                    else:
                        # For other schemes (similarity, hierarchy, etc.), use gradient
                        current_coloring_mode = 'gradient'

                    fig_sub = __plot_hyperlipea_interactive(
                        mask, coords_sub, colors_sub, names_sub, shapes_sub, option, omics_type,
                        e_colors=edge_colors_sub,
                        build_edges=show_edges_option2_subfig1,
                        show_labels=show_labels_option2_subfig1,
                        edge_opacity=edge_opacity_option2_subfig1,
                        coloring_mode=current_coloring_mode,
                        coloring_scheme=coloring_scheme
                    )

                    # Determine coloring display name
                    coloring_display_map = {
                        'default': 'Default',
                    'similarity': 'Similarity',
                        'hierarchy': 'Hierarchy',
                    }
                    coloring_display = coloring_display_map.get(coloring_scheme, coloring_scheme.capitalize())
                    fig_sub.update_layout(title=f"Subnetwork of selected node(s) and first‑neighbors ({coloring_display} coloring)")
                
                    if show_labels_option2_subfig1:
                        fig_sub.update_traces(textposition='top center', textfont_size=12)
                    else:
                        fig_sub.update_traces(text='')

                    # Update ONLY node sizes (not legend)
                    for trace in fig_sub.data:
                        if trace.showlegend is False and hasattr(trace, 'mode') and 'markers' in trace.mode:
                            # Scale the original sizes stored in customdata
                            if hasattr(trace, 'customdata') and trace.customdata is not None:
                                original_sizes = [cd[0] for cd in trace.customdata]
                                trace.marker.size = [s * node_size_slider_sub for s in original_sizes]
                            else:
                                trace.marker.size = 12 * node_size_slider_sub

                    # Display the subnetwork figure for Column 1
                    st.plotly_chart(fig_sub, use_container_width=True, key="subnetwork_plot_option2_fig1")
                    
                    # Add download buttons for subnetwork
                    add_download_buttons(fig_sub, "option2_subfig1", "subnetwork_option2_fig1")

                st.markdown("""
                <style>
                div[data-testid="stDownloadButton"] > button {
                    background: linear-gradient(to bottom, #cccccc, #999999) !important;
                    color: black !important;
                    border-radius: 8px !important;
                    font-weight: bold !important;
                    border: 1px solid #888 !important;
                    box-shadow: inset 0 1px 0 rgba(255,255,255,0.5), 0 4px 6px rgba(0, 0, 0, 0.15) !important;
                    text-shadow: 0 1px 1px rgba(255,255,255,0.3) !important;
                    transition: all 0.2s ease-in-out !important;
                }

                div[data-testid="stDownloadButton"] > button:hover {
                    background: linear-gradient(to bottom, #dddddd, #aaaaaa) !important;
                }

                div[data-testid="stDownloadButton"] > button:active {
                    transform: translateY(2px) !important;
                    box-shadow: inset 0 1px 0 rgba(255,255,255,0.3), 0 2px 3px rgba(0, 0, 0, 0.15) !important;
                }
                </style>
                """, unsafe_allow_html=True)        

            # Place download button right after plot
            # New Visualization button
            col_space1, col_center, col_space2 = st.columns([2, 2, 2])
            with col_center:
                st.markdown("""
                <style>
                /* Target the second column of the last horizontal block */
                div[data-testid="stHorizontalBlock"]:last-of-type > div:nth-child(2) button {
                    background: linear-gradient(to bottom, #cccccc, #999999);
                    color: black;
                    border-radius: 8px;
                    font-weight: bold;
                    border: 1px solid #888;
                    box-shadow: inset 0 1px 0 rgba(255,255,255,0.5), 0 4px 6px rgba(0, 0, 0, 0.15);
                    text-shadow: 0 1px 1px rgba(255,255,255,0.3);
                    transition: all 0.2s ease-in-out;
                }

                div[data-testid="stHorizontalBlock"]:last-of-type > div:nth-child(2) button:hover {
                    background: linear-gradient(to bottom, #dddddd, #aaaaaa);
                }

                div[data-testid="stHorizontalBlock"]:last-of-type > div:nth-child(2) button:active {
                    transform: translateY(2px);
                    box-shadow: inset 0 1px 0 rgba(255,255,255,0.3), 0 2px 3px rgba(0, 0, 0, 0.15);
                }
                </style>
                """, unsafe_allow_html=True)
            
                if st.button("➕ New Visualization", key="new_viz_option2_fig1"):
                    st.session_state.clear()
                    st.rerun()

    
    # ==================== COLUMN 2 (Fig.2) ====================
    with col2:
            # Option 2 — Node Coloring Options (same as Option 1)
            # -----------------------------
            st.markdown("**🎨 Node Coloring Options:**")

            # Option 2 usually has no p-values unless you explicitly provide them,
            # so keep pathway significance disabled here.
            has_pvalues = False

            # Detect if colors are present in uploaded files (same detection as Fig.1)
            has_edge_colors_fig2 = edge_color and isinstance(edge_color, dict) and len(edge_color) > 0
            has_node_colors_fig2 = 'wcolor_base_option2_fig2' in st.session_state and st.session_state.get('wcolor_base_option2_fig2') is not None
            
            # Build coloring options based on what's available
            if has_edge_colors_fig2 or has_node_colors_fig2:
                coloring_options = ['default', 'similarity', 'hierarchy']
                format_dict = {
                    'default': 'Default (Colors from files)',
                    'hierarchy': 'Hierarchy (Radial position)',
                    'similarity': 'Similarity (Angular position)',
                }
                default_scheme = 'default'
            else:
                coloring_options = ['similarity', 'hierarchy']
                format_dict = {
                    'hierarchy': 'Hierarchy (Radial position)',
                    'similarity': 'Similarity (Angular position)',
                }
                default_scheme = 'similarity'

            # Get the default index
            default_index_fig2 = coloring_options.index(default_scheme)
            
            coloring_scheme = st.radio(
                "Select coloring scheme:",
                options=coloring_options,
                format_func=lambda x: format_dict[x],
                key="coloring_scheme_option2_fig2",
                horizontal=True,
                index=default_index_fig2
            )

            labels_data = None
            custom_node_color_map = None
            custom_edge_color_map = None

            if coloring_scheme == 'default':
                st.info("📋 Using default colors from uploaded files (edge and/or node colors)")
            else:
                labels_data = None

            show_labels_option2_fig2 = st.checkbox("Show node labels (Fig.2)", value=False, key="show_labels_option2_fig2")
            show_edges_option2_fig2 = st.checkbox("Show edges (Fig.2)", value=False, key="show_edges_option2_fig2")
            edge_opacity_option2_fig2 = st.slider("Edge opacity (Fig.2)", min_value=0.1, max_value=1.0, value=0.6, step=0.1, key="edge_opacity_option2_fig2")
            node_size_slider = create_slider(f"Figure 2")

            col1, = st.columns(1)
            with col1:
                st.markdown("""
                <style>
                div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
                    background: linear-gradient(to bottom, #b6fcd5, #90ee90);
                    color: green;
                    border-radius: 8px;
                    font-weight: bold;
                    border: 1px solid #6cb56c;
                    box-shadow: inset 0 1px 0 rgba(255,255,255,0.6), 0 4px 6px rgba(0, 0, 0, 0.2);
                    text-shadow: 0 1px 1px rgba(255, 255, 255, 0.5);
                    transition: all 0.2s ease-in-out;
                }

                div[data-testid="stHorizontalBlock"] > div:nth-child(1) button:hover {
                    background: linear-gradient(to bottom, #c1ffe0, #a9f5bc);
                }

                div[data-testid="stHorizontalBlock"] > div:nth-child(1) button:active {
                    transform: translateY(2px);
                    box-shadow: inset 0 1px 0 rgba(255,255,255,0.3), 0 2px 3px rgba(0, 0, 0, 0.2);
                }
                </style>
                """, unsafe_allow_html=True)
                if st.button("Compute hyperbolic embedding", key="compute_embedding_option2_fig2"):

                    try:
                        # Show network size info
                        n_nodes = x.shape[0]
                        n_edges = x.nnz if hasattr(x, 'nnz') else np.count_nonzero(x)
                        st.info(f"🔢 Network size: {n_nodes} nodes, {n_edges} edges")
                        
                        # Initialize edge_colors properly
                        edge_colors_to_use = None 
                        
                        # Handle default mode - use edge colors from adjacency list if available
                        if coloring_scheme == 'default':
                            if has_edge_colors_fig2:
                                edge_colors_to_use = edge_color
       
                        # Determine node_color to use based on coloring scheme
                        if coloring_scheme == 'default' and has_node_colors_fig2:
                            node_color_to_use = st.session_state.wcolor_base_option2_fig1
                        elif coloring_scheme == 'default' and has_edge_colors_fig2:
                            # NEW: If in default mode with edge colors but no explicit node colors,
                            # generate black for diamonds, red for circles
                            node_color_to_use = np.zeros((len(wsymbol), 3))
                            for i, shape in enumerate(wsymbol):
                                if shape == 'd':  # diamond
                                    node_color_to_use[i] = [0, 0, 0]  # black
                                else:  # circle
                                    node_color_to_use[i] = [1, 0, 0]  # red
                        elif node_color is not None:
                            node_color_to_use = node_color
                        else:
                            # Fallback: generate gradient colors if no node colors available
                            node_color_to_use = None
                        
                        coords, excel_buffer, fig = run_hyperlipea_with_progress(
                            x, wtype, node_color_to_use, full_names, wsymbol, option, omics_type,
                            e_colors=edge_colors_to_use, coloring_scheme=coloring_scheme,
                            labels_data=labels_data, edge_opacity=edge_opacity_option2_fig2
                        )

                        # determine final node colors used
                        if coloring_scheme == 'default':
                            # Use colors from uploaded files
                            if has_node_colors_fig2:
                                wcolor_used = st.session_state.get('wcolor_base_option2_fig2')
                            else:
                                # NEW: If no node colors but edge colors exist, use black for diamonds, red for circles
                                if has_edge_colors_fig2:
                                    wcolor_used = np.zeros((len(wsymbol), 3))
                                    for i, shape in enumerate(wsymbol):
                                        if shape == 'd':  # diamond
                                            wcolor_used[i] = [0, 0, 0]  # black
                                        else:  # circle
                                            wcolor_used[i] = [1, 0, 0]  # red
                                else:
                                    # Fallback to gradient if no node colors and no edge colors provided
                                    wcolor_used = apply_gradient_coloring(x, coords, wsymbol, coloring='similarity')
                        elif coloring_scheme != 'Pathway Significance':
                            wcolor_used = apply_gradient_coloring(x, coords, wsymbol, coloring=coloring_scheme, labels=labels_data)
                        else:
                            # Pathway Significance - use node_color if available, otherwise use gradient
                            if node_color is not None:
                                wcolor_used = node_color
                            else:
                                wcolor_used = apply_gradient_coloring(x, coords, wsymbol, coloring='similarity')

                        st.session_state.edge_colors_option2_fig2 = edge_colors_to_use  # Store edge colors used
                        st.session_state.option2_fig2 = fig
                        st.session_state.option2_coords_fig2 = coords
                        st.session_state.option2_excel_buffer_fig2 = excel_buffer
                        st.session_state.wcolor_base_option2_fig2 = wcolor_used  # Store base colors
                        st.session_state.wcolor_current_option2_fig2 = wcolor_used  # Store current colors
                        st.session_state.coloring_scheme_stored_option2_fig2 = coloring_scheme
                    except Exception as e:
                        st.error(f"Error during hyperbolic embedding computation: {e}")

            # If embedding already computed, apply UI-driven updates and display
            if "option2_fig2" in st.session_state:
                # If user changed coloring scheme, regenerate with new colors
                current_scheme = st.session_state.get('coloring_scheme_stored_option2_fig2', 'similarity')
    
                # Only recompute colors, not the entire figure 
                if current_scheme != coloring_scheme or coloring_scheme == 'labels':
                    # Clear stale selections
                    if 'selected_nodes_option2_fig2' in st.session_state:
                        del st.session_state['selected_nodes_option2_fig2']

                    # Handle different coloring modes
                    if coloring_scheme == 'default':
                        # Use colors from uploaded files
                        if has_node_colors_fig2:
                            wcolor_new = st.session_state.wcolor_base_option2_fig1
                        else:
                            # NEW: If no node colors but edge colors exist, use black for diamonds, red for circles
                            if has_edge_colors_fig2:
                                wcolor_new = np.zeros((len(wsymbol), 3))
                                for i, shape in enumerate(wsymbol):
                                    if shape == 'd':  # diamond
                                        wcolor_new[i] = [0, 0, 0]  # black
                                    else:  # circle
                                        wcolor_new[i] = [1, 0, 0]  # red
                            else:
                                # Fallback to gradient if no node colors and no edge colors provided
                                wcolor_new = apply_gradient_coloring(
                                    x, st.session_state.option2_coords_fig2, wsymbol,
                                    coloring='similarity'
                                )
                    else:
                        wcolor_new = apply_gradient_coloring(
                            x, st.session_state.option2_coords_fig2, wsymbol,
                            coloring=coloring_scheme, labels=labels_data
                        )

                    # Store new colors
                    st.session_state.wcolor_current_option2_fig2 = wcolor_new
                    st.session_state.coloring_scheme_stored_option2_fig2 = coloring_scheme
        
                    # ⭐ CRITICAL: Force edge rebuild when coloring changes
                    st.session_state.edges_need_rebuild_option2_fig2 = True

                # Get current colors
                base_wcolor = st.session_state.get("wcolor_base_option2_fig2", None)
                current_wcolor = st.session_state.get("wcolor_current_option2_fig2", base_wcolor) 
                
                # If still None, generate default gradient colors
                if current_wcolor is None:
                    current_wcolor = apply_gradient_coloring(
                        x, st.session_state.option2_coords_fig2, wsymbol,
                        coloring='similarity'
                    ) 

                # Check if opacity changed
                current_opacity = st.session_state.get('last_edge_opacity_option2_fig2', 0.6)
                if current_opacity != edge_opacity_option2_fig2:
                    st.session_state.edges_need_rebuild_option2_fig2 = True
                    st.session_state.last_edge_opacity_option2_fig2 = edge_opacity_option2_fig2
            
                # Make a deep copy so we don't mutate the cached object
                fig1 = go.Figure(st.session_state.option2_fig2)

                # ⭐ DYNAMIC COLOR UPDATE: Modify node traces in-place
                o_nodes = [i for i, s in enumerate(wsymbol) if s == "o"]
                d_nodes = [i for i, s in enumerate(wsymbol) if s == "d"]
                
                # Update node colors for circle and diamond traces
                for trace_idx, trace in enumerate(fig1.data):
                    if hasattr(trace, 'mode') and 'markers' in str(trace.mode) and trace.showlegend is False:
                        symbol = trace.marker.symbol
        
                        if symbol == 'circle':
                            node_indices = o_nodes
                        elif symbol == 'diamond':
                            node_indices = d_nodes
                        else:
                            continue
        
                        # Build color list for this trace
                        trace_colors = []
                        for idx in node_indices:
                            if idx < len(current_wcolor):
                                rgb = current_wcolor[idx]
                                if isinstance(rgb, str):
                                    trace_colors.append(rgb)
                                else:
                                    trace_colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')
                            else:
                                trace_colors.append('rgb(128,128,128)')  # Fallback gray
            
                        trace.marker.color = trace_colors
            
                # ⭐ FIX: Always remove old edges first, then rebuild if needed
                # Remove ALL edge traces
                fig1.data = [trace for trace in fig1.data 
                    if not (hasattr(trace, 'mode') and trace.mode == 'lines' and not trace.showlegend)]
    
                circle_boundary_trace = None
                legend_traces_existing = []
                node_traces_existing = []
            
                for trace in fig1.data:
                    if hasattr(trace, 'mode'):
                        if trace.mode == 'lines' and trace.showlegend is False and hasattr(trace, 'hoverinfo') and trace.hoverinfo == 'skip':
                            # This is the circle boundary
                            circle_boundary_trace = trace
                        elif 'markers' in str(trace.mode) and trace.showlegend is False:
                            # This is a node trace
                            node_traces_existing.append(trace)
                        elif trace.showlegend is True:
                            # This is a legend trace
                            legend_traces_existing.append(trace)

                # ---- Dynamic title + legend update (no recompute) ----
                fig1.update_layout(title=_hyperpathway_plot_title(coloring_scheme))

                legend_traces_existing = _build_hyperpathway_legend_traces(
                    option=option,
                    coloring_scheme=coloring_scheme,
                    omics_type=omics_type,
                )

                edge_traces_new = []
                needs_rebuild = (show_edges_option2_fig2 or st.session_state.get('edges_need_rebuild_option2_fig2', False))
                # Clear custom edge colors when switching away from preference mode
                if coloring_scheme != 'preference' and 'custom_edge_colors_option2_fig2' in st.session_state:
                    # Force rebuild to apply correct edge coloring for new mode
                    needs_rebuild = True

                if needs_rebuild:
                    # Build edges with current colors
                    e1, e2 = triu(x, k=1).nonzero()
                    max_edges = 20000
                    if len(e1) > max_edges:
                        idx = np.random.choice(len(e1), size=max_edges, replace=False)
                        e1, e2 = e1[idx], e2[idx]

                    coords_unit = __compute_plot_coords(st.session_state.option2_coords_fig2)
                    edges_by_color = {}
                    default_color = '#CCCCCC'

                    for e_start, e_end in zip(e1, e2):
                        start_coords = coords_unit[e_start]
                        end_coords = coords_unit[e_end]
    
                        if np.sum(start_coords - end_coords) != 0:
                            _, _, cart_arc_1, cart_arc_2 = __hyperbolic_line_lipea(
                                start_coords.copy(), end_coords.copy(), 1, [0, 0], 'step', 0.05
                            )

                            # ⭐ CRITICAL: Transform back to native space (this was missing!)
                            pol_arc = np.empty((len(cart_arc_1), 2))
                            pol_arc[:] = np.nan
                            pol_arc[:, 0], pol_arc[:, 1] = __cart2pol(cart_arc_1, cart_arc_2)
                            pol_arc[:, 1] = 2 * np.arctanh(pol_arc[:, 1])  # Transform radius
                            cart_arc_1, cart_arc_2 = __pol2cart(pol_arc[:, 0], pol_arc[:, 1])

                            # Initialize edge color string with default
                            edge_color_str = default_color
        
                            if coloring_scheme == 'default':
                                # PRIORITY 1: Use edge colors from adjacency list if available
                                if has_edge_colors_fig2:
                                    canonical_key = (min(e_start, e_end), max(e_start, e_end))
                                    
                                    if canonical_key in edge_color:
                                        edge_color_str = edge_color[canonical_key]
                                    elif (e_start, e_end) in edge_color:
                                        edge_color_str = edge_color[(e_start, e_end)]
                                    elif (e_end, e_start) in edge_color:
                                        edge_color_str = edge_color[(e_end, e_start)]
                                    # else: edge_color_str stays as default_color
                                else:
                                    # No edge colors in file, keep default gray color
                                    edge_color_str = default_color                                   
                            else:  
                                # Gradient mode: color by pathway node
                                pathway_node = None
                                if wsymbol[e_start] == 'd':
                                    pathway_node = e_start
                                elif wsymbol[e_end] == 'd':
                                    pathway_node = e_end
        
                                if pathway_node is not None:
                                    rgb = current_wcolor[pathway_node]
                                    edge_color_str = f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'
                                else:
                                    rgb = (current_wcolor[e_start] + current_wcolor[e_end]) / 2
                                    edge_color_str = f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'

                            if edge_color_str not in edges_by_color:
                                edges_by_color[edge_color_str] = {'x': [], 'y': []}
                            edges_by_color[edge_color_str]['x'] += list(cart_arc_1) + [None]
                            edges_by_color[edge_color_str]['y'] += list(cart_arc_2) + [None]

                    # Add edge traces with visibility based on checkbox
                    for color, coords in edges_by_color.items():
                        edge_traces_new.append(go.Scatter(
                            x=coords['x'], y=coords['y'], mode='lines',
                            line=dict(width=1, color=color),
                            opacity=edge_opacity_option2_fig2,
                            hoverinfo='none', showlegend=False,
                            visible=show_edges_option2_fig2  # ⭐ KEY: Respect checkbox state
                        ))
        
                    # Clear rebuild flag
                    st.session_state.edges_need_rebuild_option2_fig2 = False

                fig1.data = []  # Clear all traces

                # Add traces in CORRECT rendering order:
                if circle_boundary_trace is not None:
                    fig1.add_trace(circle_boundary_trace)      # 1. Background
    
                for edge_trace in edge_traces_new:             # 2. Edges (bottom)
                    fig1.add_trace(edge_trace)
    
                for legend_trace in legend_traces_existing:    # 3. Legend
                    fig1.add_trace(legend_trace)
                # ---- make legend non-overlapping, below the plot ----
                # Count legend entries you created (you can also just use len(fig1.data) filtered by showlegend)
                legend_count = sum(1 for tr in fig1.data if getattr(tr, "showlegend", False))
                place_legend_below(fig1, n_items=legend_count)
    
                for node_trace in node_traces_existing:        # 4. Nodes (TOP)
                    fig1.add_trace(node_trace)

                # Labels toggle - change mode instead of clearing text
                for trace in fig1.data:
                    if hasattr(trace, 'mode') and 'markers' in str(trace.mode) and trace.showlegend is False:
                        if show_labels_option2_fig2:
                            trace.mode = 'markers+text'
                            trace.textposition = 'top center'
                            trace.textfont = dict(size=12)
                        else:
                            trace.mode = 'markers' # Remove text from mode, keeps markers only
            
                # Node size adjustment - scale original degree-based sizes
                for trace in fig1.data:
                    if hasattr(trace, 'mode') and 'markers' in str(trace.mode) and trace.showlegend is False:
                        # Scale the original sizes stored in customdata
                        if hasattr(trace, 'customdata') and trace.customdata is not None:
                            original_sizes = [cd[0] for cd in trace.customdata]
                            trace.marker.size = [s * node_size_slider for s in original_sizes]
                        else:
                            # Fallback for traces without customdata
                            trace.marker.size = 12 * node_size_slider

                edge_visibility_key = f"edges_visible_{show_edges_option2_fig2}"
                if st.session_state.get("last_edge_state_option2_fig2") != edge_visibility_key:
                    st.session_state.last_edge_state_option2_fig2 = edge_visibility_key

                # Add download buttons for Figure 2
                st.markdown("**📥 Download High-Resolution:**")
                add_download_buttons(fig1, "fig1_main", "hyperpathway_fig1")
                st.markdown("---")

                selected_points = plotly_events(
                    fig1,
                    click_event=False,
                    select_event=True,
                    override_height=800,
                    key=f"col2_option2_plot_{edge_visibility_key}"
                )

                # Download node coordinates + interactions (Visualization #2)
                if "option2_excel_buffer_fig2" in st.session_state:
                    st.download_button(
                        label="Download coordinates + list of interactions",
                        data=st.session_state.option2_excel_buffer_fig2.getvalue(),
                        file_name="network_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_option2_col2_fig2"
                    )
                with st.expander("Select any node(s) by name"):
                    st.markdown("""
                    **You can select any node(s) in two ways:**
    
                    - **Interactive selection**: Click nodes directly on the visualization (hold `Shift` to select multiple nodes). Or use 'Box Select' on top of the visualization to select multiple nodes
                    - **Name-based selection**: Use the list below or type a node name to search
                    """)
                    # Create display options (full name : first word)
                    display_options = [f"{full_names[i]} ({fixed_names[i]})" for i in range(len(full_names))]

                    selected_display = st.multiselect(
                        "Choose nodes..",
                        options=display_options,
                        default=[],
                        key="node_selector_option2_fig2"
                    )

                    # Map back to indices
                    selected_nodes_by_name = [i for i, opt in enumerate(display_options) if opt in selected_display]

                # Map plotly points → global indices
                selected_nodes_by_click = []
                if selected_points:
                    o_nodes = [i for i, s in enumerate(wsymbol) if s == "o"]
                    d_nodes = [i for i, s in enumerate(wsymbol) if s == "d"]
                    index_groups = {"circle": o_nodes, "diamond": d_nodes}

                    for p in selected_points:
                        curve = p["curveNumber"]
                        point = p["pointIndex"]
    
                        # ⭐ SAFETY: Check if curve index is valid
                        if curve >= len(fig1.data):
                            continue
                
                        trace = fig1.data[curve]
            
                        # ⭐ SKIP if this is not a node trace
                        if not hasattr(trace, 'marker') or not hasattr(trace.marker, 'symbol') or trace.marker.symbol is None:
                            continue
                
                        symbol = trace.marker.symbol

                        # ⭐ SAFETY CHECK: Ensure point index is valid
                        if symbol in index_groups and point < len(index_groups[symbol]):
                            global_index = index_groups[symbol][point]
                            selected_nodes_by_click.append(global_index)
            
                # Combine both sources, preserving existing selection if no new input
                new_selection = list(set(selected_nodes_by_name + selected_nodes_by_click))

                # Initialize selected_nodes_fig1 if it doesn't exist
                if 'selected_nodes_option2_fig2' not in st.session_state:
                    st.session_state.selected_nodes_option2_fig2 = []

                # Update selection: prioritize any new selection (click or name)
                if new_selection:
                    st.session_state.selected_nodes_option2_fig2 = new_selection
                # else: keep existing selection from session state (don't update)

                # Handle subnetwork extraction
                if st.session_state.selected_nodes_option2_fig2:         
                    try:
                        show_labels_option2_subfig2 = st.checkbox("Show node labels (Subfig.2)", value=False, key="show_labels_option2_subfig2")
                        show_edges_option2_subfig2 = st.checkbox("Show edges (Subfig.2)", value=True, key="show_edges_option2_subfig2")
                        edge_opacity_option2_subfig2 = st.slider("Edge opacity (Subfig.2)", min_value=0.1, max_value=1.0, value=st.session_state.get("edge_opacity_fig1", 0.6), step=0.1, key="edge_opacity_option2_subfig2")
                        node_size_slider_sub = create_slider(f"Subnetwork Fig 2")

                        # Get current colors being used
                        base_wcolor = st.session_state.get("wcolor_base_option2_fig2", None)
                        current_wcolor = st.session_state.get("wcolor_current_option2_fig2", base_wcolor)
                        
                        # If still None, generate default gradient colors
                        if current_wcolor is None:
                            current_wcolor = apply_gradient_coloring(
                                x, st.session_state.option2_coords_fig2, wsymbol,
                                coloring='similarity'
                            )      

                        # Determine which edge colors to pass based on coloring scheme
                        edge_colors_to_pass = None
                        if coloring_scheme == 'default' and has_edge_colors_fig2:
                            # Pass edge colors from adjacency file for default mode
                            edge_colors_to_pass = edge_color
                        
                        mask, coords_sub, colors_sub, names_sub, shapes_sub, edge_colors_sub = filter_subgraph(
                            x, st.session_state.option2_coords_fig2, current_wcolor,
                            full_names, wsymbol, st.session_state.selected_nodes_option2_fig2,
                            edge_colors=edge_colors_to_pass
                        )
                        
                        # Determine coloring mode for subnetwork based on scheme
                        if coloring_scheme == 'default':
                            # In default mode, use 'default' mode so edges use e_colors dict
                            current_coloring_mode = 'default'
                        else:
                            # For other schemes (similarity, hierarchy, etc.), use gradient
                            current_coloring_mode = 'gradient'

                        fig_sub = __plot_hyperlipea_interactive(
                            mask, coords_sub, colors_sub, names_sub, shapes_sub, option, omics_type,
                            e_colors=edge_colors_sub,
                            build_edges=show_edges_option2_subfig2,
                            show_labels=show_labels_option2_subfig2,
                            edge_opacity=edge_opacity_option2_subfig2,
                            coloring_mode=current_coloring_mode,
                            coloring_scheme=coloring_scheme
                        )                

                        # Determine coloring display name
                        coloring_display_map = {
                            'default': 'Default',
                            'similarity': 'Similarity',
                            'hierarchy': 'Hierarchy',
                        }
                        coloring_display = coloring_display_map.get(coloring_scheme, coloring_scheme.capitalize())
                        fig_sub.update_layout(title=f"Subnetwork of selected node(s) and first‑neighbors ({coloring_display} coloring)")
                    
                        if show_labels_option2_subfig2:
                            fig_sub.update_traces(textposition='top center', textfont_size=12)
                        else:
                            fig_sub.update_traces(text='')

                        # Update ONLY node sizes (not legend)
                        for trace in fig_sub.data:
                            if trace.showlegend is False and hasattr(trace, 'mode') and 'markers' in trace.mode:
                                # Scale the original sizes stored in customdata
                                if hasattr(trace, 'customdata') and trace.customdata is not None:
                                    original_sizes = [cd[0] for cd in trace.customdata]
                                    trace.marker.size = [s * node_size_slider_sub for s in original_sizes]
                                else:
                                    trace.marker.size = 12 * node_size_slider_sub

                        # Display the subnetwork figure
                        st.plotly_chart(fig_sub, use_container_width=True, key="subnetwork_plot_option2_fig2")
                                               
                        # Add download buttons for subnetwork
                        add_download_buttons(fig_sub, "option2_subfig2", "subnetwork_option2")
                        
                    except Exception as e:
                        st.error(f"❌ ERROR in subnetwork extraction: {str(e)}")
                        st.error(f"Error type: {type(e).__name__}")
                        import traceback
                        st.code(traceback.format_exc())

                st.markdown("""
                <style>
                div[data-testid="stDownloadButton"] > button {
                    background: linear-gradient(to bottom, #cccccc, #999999) !important;
                    color: black !important;
                    border-radius: 8px !important;
                    font-weight: bold !important;
                    border: 1px solid #888 !important;
                    box-shadow: inset 0 1px 0 rgba(255,255,255,0.5), 0 4px 6px rgba(0, 0, 0, 0.15) !important;
                    text-shadow: 0 1px 1px rgba(255,255,255,0.3) !important;
                    transition: all 0.2s ease-in-out !important;
                }

                div[data-testid="stDownloadButton"] > button:hover {
                    background: linear-gradient(to bottom, #dddddd, #aaaaaa) !important;
                }

                div[data-testid="stDownloadButton"] > button:active {
                    transform: translateY(2px) !important;
                    box-shadow: inset 0 1px 0 rgba(255,255,255,0.3), 0 2px 3px rgba(0, 0, 0, 0.15) !important;
                }
                </style>
                """, unsafe_allow_html=True)        

            # Place download button right after plot
            # New Visualization button
            col_space1, col_center, col_space2 = st.columns([2, 2, 2])
            with col_center:
                st.markdown("""
                <style>
                /* Target the second column of the last horizontal block */
                div[data-testid="stHorizontalBlock"]:last-of-type > div:nth-child(2) button {
                    background: linear-gradient(to bottom, #cccccc, #999999);
                    color: black;
                    border-radius: 8px;
                    font-weight: bold;
                    border: 1px solid #888;
                    box-shadow: inset 0 1px 0 rgba(255,255,255,0.5), 0 4px 6px rgba(0, 0, 0, 0.15);
                    text-shadow: 0 1px 1px rgba(255,255,255,0.3);
                    transition: all 0.2s ease-in-out;
                }

                div[data-testid="stHorizontalBlock"]:last-of-type > div:nth-child(2) button:hover {
                    background: linear-gradient(to bottom, #dddddd, #aaaaaa);
                }

                div[data-testid="stHorizontalBlock"]:last-of-type > div:nth-child(2) button:active {
                    transform: translateY(2px);
                    box-shadow: inset 0 1px 0 rgba(255,255,255,0.3), 0 2px 3px rgba(0, 0, 0, 0.15);
                }
                </style>
                """, unsafe_allow_html=True)
            
                if st.button("➕ New Visualization", key="new_viz_option2_fig2"):
                    st.session_state.clear()
                    st.rerun()
