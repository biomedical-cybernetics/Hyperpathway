from os import makedirs
from os.path import abspath, dirname, exists as path_exists, join as join_path
from scipy.sparse import csgraph
from scipy.sparse import issparse, csr_matrix
from scipy.sparse import triu
import os
from scipy.sparse.linalg import svds
from convert_pea_to_bipartite_net_command_tool_v2 import process_input_pea_table, build_network, process_adjacency_list, process_list_nodes
import sys
import powerlaw
import itertools 
import io
import scipy.io as sio
import scipy as sp
import numpy as np
import time
from datetime import datetime, timedelta 
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import pandas as pd



def print_progress(phase, done, total, bar_length=50):
    """
    Print a progress bar to the terminal.
    
    Parameters:
    -----------
    phase : str
        Name of the current phase
    done : int
        Number of completed units
    total : int
        Total number of units
    bar_length : int
        Length of the progress bar in characters
    """
    if total == 0:
        return
    
    pct = min(int(done / total * 100), 100)
    filled = int(bar_length * done / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    # Use \r to overwrite the same line
    sys.stdout.write(f'\r{phase}: [{bar}] {pct}% ({done}/{total})')
    sys.stdout.flush()
    
    # Print newline when complete
    if done >= total:
        sys.stdout.write('\n')
        sys.stdout.flush()


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


def _svd_worker_partial(kernel, k, heartbeat=None):
    """
    Perform partial SVD with optional progress tracking.
    
    Parameters:
    -----------
    kernel : ndarray or sparse matrix
        Kernel matrix
    k : int
        Number of singular values to compute
    heartbeat : callable, optional
        Callback function(phase, done, total)
    
    Returns:
    --------
    tuple : (u, s) - left singular vectors and singular values
    """
    # Set seed for reproducibility 
    np.random.seed(42)
    # Convert to sparse if dense (PROPACK works better with sparse)
    if not issparse(kernel):
        # Only convert if reasonably sparse
        sparsity  = np.count_nonzero(kernel) / kernel.size
        if sparsity < 0.5:
            kernel = csr_matrix(kernel)

    if heartbeat:
        heartbeat("svd", 0, 1)

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
    Compute all-pairs shortest path in chunks with progress tracking.
    
    Parameters:
    -----------
    x_ : sparse matrix
        Adjacency matrix
    chunk_size : int
        Number of source nodes to process per chunk
    heartbeat : callable, optional
        Callback function(phase, done, total)
    
    Returns:
    --------
    ndarray : Distance matrix
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
    """
    Compute ISOMAP embedding with progress tracking.
    
    Parameters:
    -----------
    x_ : sparse matrix
        Adjacency matrix
    n_ : int
        Number of dimensions
    centring : str
        'yes' or 'no' for kernel centering
    heartbeat : callable, optional
        Callback function(phase, done, total)
    
    Returns:
    --------
    ndarray : Embedding coordinates
    """
    # Initialization
    x_ = x_.maximum(x_.T)

    # Chunked shortest paths with progress
    kernel = _chunked_shortest_path(x_, chunk_size=100, heartbeat=heartbeat)
    # Kernel centering
    if centring == 'yes':
        kernel = __kernel_centering(kernel)

    # Ensure float64 (required by svds)
    kernel = np.asarray(kernel, dtype=np.float64)

    # -------------------------------------------------
    # Partial SVD (Lanczos / PROPACK-style)
    # -------------------------------------------------
    k = n_  # 3 for 2D embedding

    v_, l_ = _svd_worker_partial(kernel, k, heartbeat=heartbeat)

    v_[:, 1] = v_[:, 1] * -1
    v_[:, 2] = v_[:, 2] * -1

    sqrt_l = np.sqrt(l_[:n_])
    v_ = v_[:, :n_]
    s = np.real(v_ * sqrt_l)

    return s


def __set_radial_coordinates(x_):
    """
    Set radial coordinates based on degree distribution.
    
    Parameters:
    -----------
    x_ : sparse matrix
        Adjacency matrix
    
    Returns:
    --------
    ndarray : Radial coordinates
    """
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
    """
    Set angular coordinates using ISOMAP dimension reduction.
    
    Parameters:
    -----------
    xw : sparse matrix
        Weighted adjacency matrix
    angular_adjustment : str
        'EA' for equidistant adjustment
    heartbeat : callable, optional
        Callback function(phase, done, total)
    
    Returns:
    --------
    ndarray : Angular coordinates
    """
    # dimension reduction
    dr_coords = __isomap_graph_carlo(xw, 3, 'no', heartbeat=heartbeat)

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


def ra1_weighting_sparse(x, heartbeat=None, heartbeat_every=1):
    """
    Apply RA1 weighting to sparse matrix with optional progress tracking.
    
    Parameters:
    -----------
    x : csr_matrix
        Sparse adjacency matrix
    heartbeat : callable, optional
        Callback function(phase, done, total)
    heartbeat_every : int
        How often to send heartbeat (every N operations)
    
    Returns:
    --------
    csr_matrix : Weighted adjacency matrix
    """
    x = x.tocsr().copy()
    x.data[:] = 1  # binarize

    deg = np.asarray(x.sum(axis=1)).ravel()
    cn = (x @ x.T).tocsr()  # sparse common-neighbors

    coo = x.tocoo()
    i, j = coo.row, coo.col

    data = np.empty(len(i))
    total_edges = len(i)

    for idx in range(total_edges):
        data[idx] = (deg[i[idx]] + deg[j[idx]] + deg[i[idx]] * deg[j[idx]]) / (1.0 + cn[i[idx], j[idx]])
        # Send heartbeat periodically
        if heartbeat and (idx % heartbeat_every == 0 or idx == total_edges - 1):
            heartbeat("ra1_weighting", idx + 1, total_edges) 

    return csr_matrix((data, (i, j)), shape=x.shape)


def __coal_embed(x_, pre_weighting, dim_red, angular_adjustment, dims, heartbeat=None):
    """
    Coalescent embedding with progress tracking.
    
    Parameters:
    -----------
    x_ : sparse matrix
        Adjacency matrix
    pre_weighting : str
        Pre-weighting method (e.g., 'RA1')
    dim_red : str
        Dimensionality reduction method
    angular_adjustment : str
        Angular adjustment method (e.g., 'EA')
    dims : int
        Number of dimensions (2 or 3)
    heartbeat : callable, optional
        Callback function(phase, done, total)
    
    Returns:
    --------
    ndarray : Hyperbolic coordinates
    """
    xd = x_.copy()
    xd[x_ == 2] = 0
    x_[x_ == 2] = 1

    # Pre-weighting with progress
    if heartbeat:
        heartbeat("weighting", 0, 1)

    xw = ra1_weighting_sparse((x_ > 0).astype(int), heartbeat=heartbeat, heartbeat_every=1000)

    if heartbeat:
        heartbeat("weighting", 1, 1)

    # dimension reduction and set of hyperbolic coordinates
    if dims == 2:
        coords = np.zeros((x_.shape[0], 2))

        if heartbeat:
            heartbeat("embedding", 0, 2)

        coords[:, 0] = __set_angular_coordinates_ncISO_2D(xw, angular_adjustment, heartbeat=heartbeat)

        if heartbeat:
            heartbeat("embedding", 1, 2)

        coords[:, 1] = __set_radial_coordinates(xd)

        if heartbeat:
            heartbeat("embedding", 2, 2)

    elif dims == 3:
        # TODO: 3D handling
        raise NotImplementedError("coal_embed 3D not implemented")

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
    """
    Generate a blue-to-red colormap similar to MATLAB version.
    
    Parameters:
    -----------
    n : int
        Number of colors in the colormap
    
    Returns:
    --------
    ndarray : (n, 3) RGB color array
    """
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


def plot_hyperpathway_static_gradient_color(x_, coords_native, names, node_shape, 
                                           coloring='popularity', labels=None,
                                           build_edges=True, max_edges=20000, 
                                           output_file='hyperpathway_gradient_color_plot.png', 
                                           dpi=300, figsize=(12, 12), show_labels=False,
                                           node_size_scale=1.0, edge_opacity=0.6):
    """
    Create a static hyperbolic visualization with gradient coloring options.
    
    Parameters:
    -----------
    x_ : sparse matrix
        Adjacency matrix (NxN) of the network
    coords_native : ndarray
        Polar hyperbolic coordinates (Nx2) in the form [theta, r]
    names : list
        Node labels
    node_shape : list
        Node shapes ('o' for circle, 'd' for diamond)
    coloring : str, default='popularity'
        How to color the nodes:
        - 'popularity': nodes colored by degree with blue-to-red colormap
        - 'similarity': nodes colored by angular coordinate with HSV colormap
        - 'labels': nodes colored by labels (requires labels parameter)
    labels : array-like, optional
        Numerical labels for nodes (only needed if coloring='labels')
    build_edges : bool, default=True
        Whether to draw edges
    max_edges : int, default=20000
        Maximum number of edges to render
    output_file : str, default='hyperpathway_gradient_color_plot.png'
        Output filename for the plot
    dpi : int, default=300
        Resolution for saved image
    figsize : tuple, default=(12, 12)
        Figure size in inches
    show_labels : bool, default=False
        Whether to show node labels
    node_size_scale : float, default=1.0
        Scaling factor for node sizes (multiplies the computed sizes)
        
    Returns:
    --------
    str : Path to saved figure
    """
    
    # Input validation
    if coloring not in ['popularity', 'similarity', 'labels']:
        raise ValueError("coloring must be 'popularity', 'similarity', or 'labels'")
    
    if coloring == 'labels' and labels is None:
        raise ValueError("labels parameter is required when coloring='labels'")
    
    n_nodes = x_.shape[0]
    
    # Transform coordinates from polar to Cartesian (native space)
    cart_coords_native = np.zeros_like(coords_native)
    cart_coords_native[:, 0], cart_coords_native[:, 1] = __pol2cart(
        coords_native[:, 0], coords_native[:, 1]
    )

    # ALSO transform to unit disk for hyperbolic edge drawing
    coords_unit = coords_native.copy()
    coords_unit[:, 1] = np.tanh(coords_native[:, 1] / 2)
    cart_coords_unit = np.zeros_like(coords_unit)
    cart_coords_unit[:, 0], cart_coords_unit[:, 1] = __pol2cart(
        coords_unit[:, 0], coords_unit[:, 1]
    )

    
    # Identify diamond and circle nodes
    diamond_mask = np.array([s == 'd' for s in node_shape])
    circle_mask = np.array([s == 'o' for s in node_shape])
    
    # Initialize node colors array
    node_colors = np.zeros((n_nodes, 3))
    
    # Set DIAMOND node colors based on coloring option
    diamond_indices = np.where(diamond_mask)[0]
    
    if coloring == 'popularity':
        # Color by degree (popularity) - only for diamonds
        deg = np.array((x_ > 0).sum(axis=1)).ravel()
        deg_diamonds = deg[diamond_mask]
        
        if len(deg_diamonds) > 0 and np.max(deg_diamonds) != np.min(deg_diamonds):
            # Normalize degrees to range [1, max(deg)]
            deg_normalized = np.round(
                (np.max(deg_diamonds) - 1) * (deg_diamonds - np.min(deg_diamonds)) / 
                (np.max(deg_diamonds) - np.min(deg_diamonds)) + 1
            ).astype(int)
        else:
            deg_normalized = np.ones(len(deg_diamonds), dtype=int)
        
        colors = colormap_blue_to_red(int(np.max(deg_normalized)))
        node_colors[diamond_mask] = colors[deg_normalized - 1, :]
        
    elif coloring == 'similarity':
        # Color by angular coordinate (similarity) - only for diamonds
        angles_diamonds = coords_native[diamond_mask, 0]
        angles_normalized = angles_diamonds / (2 * np.pi)  # Normalize to [0, 1]
        
        # Create HSV colors: (hue, saturation, value)
        hsv_colors = np.zeros((len(angles_normalized), 3))
        hsv_colors[:, 0] = angles_normalized  # Hue from angle
        hsv_colors[:, 1] = 1.0  # Full saturation
        hsv_colors[:, 2] = 1.0  # Full value/brightness
        
        node_colors[diamond_mask] = hsv_to_rgb(hsv_colors)
        
    elif coloring == 'labels':
        # Color by labels - only for diamonds
        labels = np.array(labels)
        labels_diamonds = labels[diamond_mask]
        unique_labels = np.unique(labels_diamonds)
        
        # Remap labels to consecutive integers
        label_mapping = {lab: i for i, lab in enumerate(unique_labels)}
        remapped_labels = np.array([label_mapping[lab] for lab in labels_diamonds])
        
        # Generate HSV colors for labels
        n_colors = len(unique_labels)
        hsv_colors = np.zeros((n_colors, 3))
        hsv_colors[:, 0] = np.linspace(0, 1, n_colors, endpoint=False)
        hsv_colors[:, 1] = 1.0
        hsv_colors[:, 2] = 1.0
        
        colors = hsv_to_rgb(hsv_colors)
        node_colors[diamond_mask] = colors[remapped_labels, :]
    
    # Set CIRCLE node colors as mixture of connected diamond neighbors
    circle_indices = np.where(circle_mask)[0]
    
    for circle_idx in circle_indices:
        # Get neighbors of this circle node
        neighbors = x_[circle_idx, :].nonzero()[1]
        
        # Filter to only diamond neighbors
        diamond_neighbors = [n for n in neighbors if diamond_mask[n]]
        
        if len(diamond_neighbors) > 0:
            # Average the colors of diamond neighbors
            neighbor_colors = node_colors[diamond_neighbors, :]
            node_colors[circle_idx] = np.mean(neighbor_colors, axis=0)
        else:
            # Default gray if no diamond neighbors
            node_colors[circle_idx] = np.array([0.5, 0.5, 0.5])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Draw circle boundary
    radius = np.max(coords_native[:, 1])
    circle_theta = np.linspace(0, 2 * np.pi, 500)
    circle_x = radius * np.cos(circle_theta)
    circle_y = radius * np.sin(circle_theta)
    ax.plot(circle_x, circle_y, 'lightgray', linewidth=1, alpha=0.5, zorder=1)
    
    # Draw edges with hyperbolic geodesics and colored by diamond nodes
    if build_edges:
        e1, e2 = triu(x_, k=1).nonzero()
        
        if len(e1) > max_edges:
            # Subsample edges
            idx = np.random.choice(len(e1), size=max_edges, replace=False)
            e1, e2 = e1[idx], e2[idx]
        
        for e_start, e_end in zip(e1, e2):
            start_coords = cart_coords_unit[e_start]
            end_coords = cart_coords_unit[e_end]   

            # Determine edge color based on diamond node
            # In bipartite network, one endpoint should be diamond, other circle
            if node_shape[e_start] == 'd':
                edge_color = node_colors[e_start]
            elif node_shape[e_end] == 'd':
                edge_color = node_colors[e_end]
            else:
                # If neither is diamond (shouldn't happen in bipartite), use gray
                edge_color = np.array([0.8, 0.8, 0.8])

            # Only draw if coordinates are different
            if np.sum(start_coords - end_coords) != 0:
                try:
                    # Compute hyperbolic geodesic
                    _, _, cart_arc_1, cart_arc_2 = __hyperbolic_line_lipea(
                        start_coords.copy(), end_coords.copy(), 1, [0, 0], 'step', 0.01
                    )

                    # Transform back to native coordinates
                    pol_arc = np.empty((len(cart_arc_1), 2))
                    pol_arc[:] = np.nan
                    pol_arc[:, 0], pol_arc[:, 1] = __cart2pol(cart_arc_1, cart_arc_2)
                    pol_arc[:, 1] = 2 * np.arctanh(pol_arc[:, 1])
                    cart_arc_1, cart_arc_2 = __pol2cart(pol_arc[:, 0], pol_arc[:, 1])

                    ax.plot(cart_arc_1, cart_arc_2, 
                           color=edge_color, linewidth=0.5, alpha=edge_opacity, zorder=2)
                except (ValueError, RuntimeWarning):
                    # If geodesic computation fails, skip this edge
                    pass

    # Calculate degree-based node sizes (matching web app logic)
    degrees = np.array((x_ > 0).sum(axis=1)).ravel()
    
    # Improved degree-based sizing with better visual separation
    min_size = 40
    max_degree = degrees.max() if degrees.max() > 0 else 1
    
    # Normalize degrees to 0-1, then apply power for non-linear scaling
    degree_normalized = degrees / max_degree
    # Use power 0.6 to amplify differences
    degree_scaled = np.power(degree_normalized, 0.6)
    
    # Scale to size range: min_size to min_size + 30
    # Then apply user's scaling factor
    node_size = (min_size + (degree_scaled * 200)) * node_size_scale
    
    # Draw nodes
    node_x = cart_coords_native[:, 0]
    node_y = cart_coords_native[:, 1]
    
    # Map shapes to matplotlib markers
    marker_map = {'o': 'o', 'd': 'D'}
    
    # Plot nodes by shape
    for shape_key in ['o', 'd']:
        mask = np.array([s == shape_key for s in node_shape])
        if not np.any(mask):
            continue
        
        x_coords = node_x[mask]
        y_coords = node_y[mask]
        sizes = node_size[mask]
        colors_subset = node_colors[mask]
        
        ax.scatter(x_coords, y_coords, s=sizes, c=colors_subset,
                  marker=marker_map[shape_key], edgecolors='black',
                  linewidths=0.5, zorder=3, alpha=0.9)
        
        # Add labels if requested
        if show_labels:
            names_subset = [names[i] for i, m in enumerate(mask) if m]
            for x, y, name in zip(x_coords, y_coords, names_subset):
                ax.text(x, y, name, fontsize=8, ha='center', va='bottom', zorder=4)
    
    # Configure plot
    coloring_titles = {
        'popularity': 'Hyperpathway (colored by node degree)',
        'similarity': 'Hyperpathway (colored by angular position)',
        'labels': 'Hyperpathway (colored by labels)'
    }
    
    ax.set_title(coloring_titles[coloring], fontsize=14, pad=20)
    ax.set_aspect('equal')
    ax.set_xlim([-radius * 1.1, radius * 1.1])
    ax.set_ylim([-radius * 1.1, radius * 1.1])
    ax.axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Gradient color plot saved to: {output_file}")

    return output_file


def plot_hyperpathway_static(x_, coords_native, node_colors, names, node_shape, option, 
                          e_colors=None, build_edges=True, max_edges=20000, 
                          corr_1_name='Correction #1', corr_2_name='Correction #2',
                          output_file='hyperpathway_plot.png', dpi=300, figsize=(12, 12),
                          show_labels=False, node_size_scale=1.0, edge_opacity=0.6):
    """
    Create a static hyperbolic visualization and save to file.
    
    Parameters:
    -----------
    x_ : sparse matrix
        Adjacency matrix
    coords_native : ndarray
        Native coordinates (polar)
    node_colors : ndarray or list
        Colors for nodes (RGB tuples or hex strings)
    names : list
        Node labels
    node_shape : list
        Node shapes ('o' for circle, 'd' for diamond)
    option : int
        1 for pathway visualization with legend, other for bipartite network
    e_colors : dict or list, optional
        Edge colors mapping
    build_edges : bool, default=True
        Whether to draw edges
    max_edges : int, default=20000
        Maximum number of edges to render
    corr_1_name : str
        Name for first correction method
    corr_2_name : str
        Name for second correction method
    output_file : str, default='hyperpathway_plot.png'
        Output filename for the plot
    dpi : int, default=300
        Resolution for saved image
    figsize : tuple, default=(12, 12)
        Figure size in inches
    show_labels : bool, default=False
        Whether to show node labels
    node_size_scale : float, default=1.0
        Scaling factor for node sizes (multiplies the computed sizes)
        
    Returns:
    --------
    str : Path to saved figure
    """
    
    # Transform coordinates
    cart_coords_native = np.zeros_like(coords_native)
    cart_coords_native[:, 0], cart_coords_native[:, 1] = __pol2cart(coords_native[:, 0], coords_native[:, 1])
    cart_coords_unit = __compute_plot_coords(coords_native)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Draw circle boundary
    radius = np.max(coords_native[:, 1])
    circle_theta = np.linspace(0, 2 * np.pi, 500)
    circle_x = radius * np.cos(circle_theta)
    circle_y = radius * np.sin(circle_theta)
    ax.plot(circle_x, circle_y, 'lightgray', linewidth=1, alpha=0.5, zorder=1)

    # Draw edges
    if build_edges:
        e1, e2 = triu(x_, k=1).nonzero()
        if len(e1) > max_edges:
            # Subsample edges
            idx = np.random.choice(len(e1), size=max_edges, replace=False)
            e1, e2 = e1[idx], e2[idx]

        default_color = '#CCCCCC'
        
        for edge_idx, (e_start, e_end) in enumerate(zip(e1, e2)):
            start_coords = cart_coords_unit[e_start]
            end_coords = cart_coords_unit[e_end]
            
            if np.sum(start_coords - end_coords) != 0:
                _, _, cart_arc_1, cart_arc_2 = __hyperbolic_line_lipea(
                    start_coords.copy(), end_coords.copy(), 1, [0, 0], 'step', 0.01
                )
                pol_arc = np.empty((len(cart_arc_1), 2))
                pol_arc[:] = np.nan
                pol_arc[:, 0], pol_arc[:, 1] = __cart2pol(cart_arc_1, cart_arc_2)
                pol_arc[:, 1] = 2 * np.arctanh(pol_arc[:, 1])
                cart_arc_1, cart_arc_2 = __pol2cart(pol_arc[:, 0], pol_arc[:, 1])
                
                # Determine edge color
                edge_color = default_color
                if e_colors is not None:
                    if isinstance(e_colors, dict):
                        canonical_key = (min(e_start, e_end), max(e_start, e_end))
                        if canonical_key in e_colors:
                            edge_color = e_colors[canonical_key]
                        elif (e_start, e_end) in e_colors:
                            edge_color = e_colors[(e_start, e_end)]
                        elif (e_end, e_start) in e_colors:
                            edge_color = e_colors[(e_end, e_start)]
                        elif f"{e_start}-{e_end}" in e_colors:
                            edge_color = e_colors[f"{e_start}-{e_end}"]
                        elif f"{e_end}-{e_start}" in e_colors:
                            edge_color = e_colors[f"{e_end}-{e_start}"]
                    elif isinstance(e_colors, (list, np.ndarray)) and edge_idx < len(e_colors):
                        edge_color = e_colors[edge_idx]
                
                ax.plot(cart_arc_1, cart_arc_2, color=edge_color, linewidth=0.5, 
                       alpha=edge_opacity, zorder=2)

    # Calculate degree-based node sizes (matching web app logic)
    degrees = np.array(x_.sum(axis=1)).ravel()
    
    # Improved degree-based sizing with better visual separation
    min_size = 6
    max_degree = degrees.max() if degrees.max() > 0 else 1
    
    # Normalize degrees to 0-1, then apply power for non-linear scaling
    degree_normalized = degrees / max_degree
    # Use power 0.6 to amplify differences (lower = more dramatic)
    degree_scaled = np.power(degree_normalized, 0.6)
    
    # Scale to size range: min_size to min_size + 30
    # Then apply user's scaling factor
    node_size = (min_size + (degree_scaled * 30)) * node_size_scale

    # Draw nodes
    node_x = cart_coords_native[:, 0]
    node_y = cart_coords_native[:, 1]

    # Map shapes to matplotlib markers
    marker_map = {'o': 'o', 'd': 'D'}
    
    # Separate nodes by shape
    for shape_key in ['o', 'd']:
        mask = np.array([s == shape_key for s in node_shape])
        if not np.any(mask):
            continue
            
        x_coords = node_x[mask]
        y_coords = node_y[mask]
        sizes = node_size[mask]
        
        # Process colors for this group
        colors_subset = [node_colors[i] for i, m in enumerate(mask) if m]
        plot_colors = []
        for color_val in colors_subset:
            if isinstance(color_val, str):
                plot_colors.append(color_val)
            else:
                plot_colors.append(color_val)  # matplotlib handles RGB tuples
        
        # Plot nodes
        ax.scatter(x_coords, y_coords, s=sizes, c=plot_colors, 
                  marker=marker_map[shape_key], edgecolors='gray', 
                  linewidths=1, zorder=3, alpha=0.8)
        
        # Add labels (only if show_labels is True)
        if show_labels:
            for i, (x, y, name) in enumerate(zip(x_coords, y_coords, 
                                             [names[j] for j, m in enumerate(mask) if m])):
                ax.text(x, y, name, fontsize=8, ha='center', va='bottom', zorder=4)

    # Configure plot
    if option == 1:
        plot_title = f"Hyperpathway visualization"
        
        # Build legend
        color_legend_map = {
            '#FF0000': f'Significant pathway ({corr_1_name})',
            '#FFA500': f'Significant pathway ({corr_2_name})',
            '#B2B2B2': 'Significant non-corrected pathway',
            '#00AA00': 'Molecules',
            '#0000FF': 'Pathways'
        }
        
        rgb_map = {
            '#FF0000': np.array([1.0, 0.0, 0.0]),
            '#FFA500': np.array([1.0, 0.6, 0.0]),
            '#B2B2B2': np.array([0.7, 0.7, 0.7]),
            '#00AA00': np.array([0.0, 1.0, 0.0]),
            '#0000FF': np.array([0.0, 0.0, 1.0])
        }
        
        legend_elements = []
        for hex_color, label in color_legend_map.items():
            target_rgb = rgb_map[hex_color]
            exists = np.any(np.all(np.isclose(node_colors, target_rgb, atol=0.01), axis=1))
            
            if exists:
                marker_style = 'D' if 'pathway' in label.lower() else 'o'
                legend_elements.append(
                    plt.Line2D([0], [0], marker=marker_style, color='w', 
                             markerfacecolor=hex_color, markersize=10, 
                             label=label, markeredgecolor='gray', markeredgewidth=1)
                )
                if hex_color == '#B2B2B2':
                    plot_title = f"Extended Hyperpathway visualization"
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper center', 
                     bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=True)
    else:
        plot_title = f"Bipartite network visualization in hyperbolic space"
    
    ax.set_title(plot_title, fontsize=14, pad=20)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Plot saved to: {output_file}")
    return output_file


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


def __plot_hyperlipea(x_, coords_native, node_colors, names, node_shape, ncc_, figures_path_):
    # Coordinates transformation
    coords_unit = coords_native.copy()
    coords_unit[:, 1] = np.tanh(coords_native[:, 1] / 2)
    cart_coords_native = np.zeros_like(coords_native)
    cart_coords_unit = np.zeros_like(coords_unit)
    cart_coords_native[:, 0], cart_coords_native[:, 1] = __pol2cart(coords_native[:, 0], coords_native[:, 1])
    cart_coords_unit[:, 0], cart_coords_unit[:, 1] = __pol2cart(coords_unit[:, 0], coords_unit[:, 1])

    fig, ax = plt.subplots()  # figsize=(7, 7)
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.set_dpi(300)

    # Plot circle
    radius = np.max(coords_native[:, 1])
    ax.set_aspect('equal')
    ax.add_patch(plt.Circle((0, 0), radius, edgecolor='0.8', facecolor='none'))

    # Plot links
    e1, e2 = np.where(np.triu(x_, 1))
    for e_start, e_end in zip(e1, e2):
        start_coords = cart_coords_unit[e_start]
        end_coords = cart_coords_unit[e_end]
        if np.sum(start_coords - end_coords) != 0:
            _, _, cart_arc_1, cart_arc_2 = __hyperbolic_line_lipea(start_coords, end_coords, 1, [0, 0], 'step', 0.05)

            # adjustment
            pol_arc = np.empty((len(cart_arc_1), 2))
            pol_arc[:] = np.nan
            pol_arc[:, 0], pol_arc[:, 1] = __cart2pol(cart_arc_1, cart_arc_2)
            pol_arc[:, 1] = 2 * np.arctanh(pol_arc[:, 1])
            cart_arc_1, cart_arc_2 = __pol2cart(pol_arc[:, 0], pol_arc[:, 1])

            ax.plot(cart_arc_1, cart_arc_2, color='0.8', linewidth=0.7)

    # plot nodes
    node_size = np.log10(2 + np.sum(x_, axis=1)) * 15
    for point_index in range(len(cart_coords_native)):
        ax.plot(cart_coords_native[point_index, 0], cart_coords_native[point_index, 1], node_shape[point_index],
                markersize=node_size[point_index],
                markerfacecolor=node_colors[point_index, :],
                markeredgecolor=[0.8, 0.8, 0.8])
        ax.annotate(names[point_index], (cart_coords_native[point_index, 0], cart_coords_native[point_index, 1]))

    # TODO: Refactor legend
    qw = {}  # Initialize an empty dictionary
    # Creating plots with NaN values
    if ncc_ == 1:
        qw[0], = ax.plot([float('nan')], 'd', markerfacecolor=[1, 0, 0], markeredgecolor=[1, 0, 0])
        qw[1], = ax.plot([float('nan')], 'd', markerfacecolor=[1, 0.6, 0], markeredgecolor=[1, 0.6, 0])
        qw[3], = ax.plot([float('nan')], 'o', markerfacecolor=[0, 1, 0], markeredgecolor=[0, 1, 0])
        # Creating legend
        plt.legend([qw[key] for key in qw], ['Bonferroni', 'Benjamini-Hochberg', 'Lipid'], loc='best')
    else:
        qw[0], = ax.plot([float('nan')], 'd', markerfacecolor=[1, 0, 0], markeredgecolor=[1, 0, 0])
        qw[1], = ax.plot([float('nan')], 'd', markerfacecolor=[1, 0.6, 0], markeredgecolor=[1, 0.6, 0])
        qw[2], = ax.plot([float('nan')], 'd', markerfacecolor=[0.7, 0.7, 0.7], markeredgecolor=[0.7, 0.7, 0.7])
        qw[3], = ax.plot([float('nan')], 'o', markerfacecolor=[0, 1, 0], markeredgecolor=[0, 1, 0])
        # Creating legend
        plt.legend([qw[key] for key in qw],
                   ['Bonferroni', 'Benjamini-Hochberg', 'nonsignificant after correction', 'Lipid'], loc='best')

    plot_title = "Coalescent Embedding hyperbolic representation (ncc=" + str(ncc_) + ")"
    plt.title(plot_title)

    if figures_path_ is not None:
        fig_path_svg = join_path(figures_path_, 'figure_' + str(ncc_) + '_python.svg')
        plt.savefig(fig_path_svg, format='svg', dpi=1200)

        fig_path_png = join_path(figures_path_, 'figure_' + str(ncc_) + '_python.png')
        plt.savefig(fig_path_png, format='png', dpi=300)

    # TODO: show plot optionally (command line argument?)
    # plt.show()
    return fig 


def remove_isolated_nodes(x_, w_type, w_color, w_name, w_symbol):
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

    return x_new, w_type_new, w_color_new, w_name_new, w_symbol_new


def hyperpathway(x_, w_type, w_color, w_name, w_symbol, option, e_colors=None, heartbeat=None, corr_1_name='Correction #1', corr_2_name='Correction #2'):
    """
    Main hyperpathway function with progress tracking.
    
    Parameters:
    -----------
    x_ : sparse matrix
        Adjacency matrix
    w_type : ndarray
        Node types
    w_color : ndarray
        Node colors
    w_name : list
        Node names
    w_symbol : list
        Node symbols
    option : int
        Visualization option
    e_colors : dict, optional
        Edge colors
    heartbeat : callable, optional
        Callback function(phase, done, total)
    corr_1_name : str
        Name for first correction method
    corr_2_name : str
        Name for second correction method
    
    Returns:
    --------
    tuple : (coords, excel_buffer)
    """
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

        list_ = np.empty((ncc, 2), dtype=int)
        for ncc_index in range(ncc):
            l1 = np.arange(0, ncc)
            l1 = np.delete(l1, ncc_index)
            m1 = t0[1, l1].T
            m1 = m1[np.where(mask[m1] == np.max(mask[m1]))]
            list_[ncc_index, 0] = t0[0, ncc_index]
            list_[ncc_index, 1] = m1[0]

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

 
def run_hyperpathway_with_progress(x, wtype, wcolor, fixed_names, wsymbol, option,
                                e_colors=None, corr_1_name='Correction #1',
                                corr_2_name='Correction #2', output_file='hyperpathway_plot.png',
                                show_labels=False, show_edges=True, max_edges=20000, edge_opacity=0.6):
    """
    Run hyperpathway computation with terminal progress bar.
    
    Parameters:
    -----------
    x : sparse matrix
        Adjacency matrix
    wtype : ndarray
        Node types
    wcolor : ndarray
        Node colors
    fixed_names : list
        Node names
    wsymbol : list
        Node symbols
    option : int
        Visualization option
    e_colors : dict, optional
        Edge colors
    corr_1_name : str
        Name for first correction method
    corr_2_name : str
        Name for second correction method
    output_file : str
        Output filename for plot
    show_labels : bool
        Whether to show node labels
    show_edges : bool
        Whether to show edges
    max_edges : int
        Maximum number of edges to render
    
    Returns:
    --------
    tuple : (coords, excel_buffer, fig_path)
    """
    print("Starting hyperbolic embedding computation...")
    print(f"Network size: {x.shape[0]} nodes, {x.nnz if hasattr(x, 'nnz') else np.count_nonzero(x)} edges")
    print()
    
    # Define heartbeat callback
    def heartbeat(phase, done, total):
        print_progress(phase, done, total)


    # Compute embedding
    try:
        coords, excel_buffer = hyperpathway(
            x, wtype, wcolor, fixed_names, wsymbol, option,
            e_colors=e_colors,
            heartbeat=heartbeat, 
            corr_1_name=corr_1_name,
            corr_2_name=corr_2_name
        )
        print("\n✓ Embedding computation complete!")
    except Exception as e:
        print(f"\n✗ Error during embedding: {e}")
        raise

    # Generate plot
    print("\nGenerating visualization...")
    try:
        fig_path = plot_hyperpathway_static(
                x, coords, wcolor, fixed_names, wsymbol, option,
                e_colors=e_colors,
                corr_1_name=corr_1_name,
                corr_2_name=corr_2_name,
                output_file=output_file,
                show_labels=show_labels,
                build_edges=show_edges,
                max_edges=max_edges,
                edge_opacity=edge_opacity
            )
        print("✓ Visualization complete!")
    except Exception as e:
        print(f"✗ Error during visualization: {e}")
        raise

    return coords, excel_buffer, fig_path

