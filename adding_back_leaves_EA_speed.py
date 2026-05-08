import numpy as np
import powerlaw
from collections import defaultdict


def compare_angles(A, B):
    # Compute forward difference in [0, 2*pi)
    d = np.mod(B - A, 2 * np.pi)

    # MATLAB: if d == 0
    if d == 0:
        return 0

    # MATLAB: elseif d < pi
    elif d < np.pi:
        return -1

    # Otherwise
    else:
        return 1


def __set_radial_coordinates(x_):
    n_ = x_.shape[0]
    #deg = np.array((x_ > 0).sum(axis=1)).ravel()
    deg = np.asarray(x_.getnnz(axis=1))

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
    seq = deg[idx]


    # -------------------------
    # RANDOMIZE ties (missing part)
    # -------------------------
    #permuted_idx = idx.copy()

    #unique_values = np.unique(seq)

    #for val in unique_values:
    #    same = np.where(seq == val)[0]

    #    if len(same) > 1:
    #        permuted = np.random.permutation(same)
    #        permuted_idx[same] = idx[permuted]


    r = np.zeros(n_)
    r[idx] = np.maximum(0, 2 * beta * np.log1p(np.arange(0, n_)) + 2 * (1 - beta) * np.log(n_))
    # MATLAB: log(1:N)

    #ranks = np.arange(1, n_ + 1)

    #r[permuted_idx] = np.maximum(0, 2 * beta * np.log(ranks) + 2 * (1 - beta) * np.log(n_))

    return r


def adjust_duplicates(coords, dA=1e-7):

    coords = np.asarray(coords).copy()

    # Normalize into [0, 2*pi)
    coords = np.mod(coords, 2 * np.pi)

    # Sort and keep original indices
    idx = np.argsort(coords)
    coords_sorted = coords[idx]

    N = len(coords_sorted)

    # -----------------------------
    # Identify duplicate runs
    # -----------------------------
    duplicate_starts = []
    duplicate_lengths = []

    start_idx = 0  # Python 0-based

    while start_idx < N:
        run_val = coords_sorted[start_idx]
        end_idx = start_idx

        while end_idx < N - 1 and coords_sorted[end_idx + 1] == run_val:
            end_idx += 1

        run_length = end_idx - start_idx + 1

        if run_length > 1:
            duplicate_starts.append(start_idx)
            duplicate_lengths.append(run_length)

        start_idx = end_idx + 1

    # -----------------------------
    # Adjust duplicates
    # -----------------------------
    for s, length in zip(duplicate_starts, duplicate_lengths):
        base_val = coords_sorted[s]
        increments = np.arange(length) * dA
        coords_sorted[s:s + length] = base_val + increments

    # Wrap again
    coords_sorted = np.mod(coords_sorted, 2 * np.pi)

    # -----------------------------
    # Final duplicate check
    # -----------------------------
    unique_angles = np.unique(coords_sorted)
    if len(unique_angles) < N:
        # fallback: external function (must exist)
        coords_sorted, _  = resolve_remaining_duplicates(coords_sorted, dA)

    # -----------------------------
    # Restore original order
    # -----------------------------
    coords_adjusted = np.empty_like(coords_sorted)
    coords_adjusted[idx] = coords_sorted

    return coords_adjusted


def resolve_remaining_duplicates(angles, dA=1e-7):
    changed = False
    angles_fixed = np.asarray(angles).copy()

    # Sort with indices
    sorted_idx = np.argsort(angles_fixed)
    sorted_angles = angles_fixed[sorted_idx]

    N = len(sorted_angles)

    i = 0  # Python 0-based indexing

    while i < N - 1:

        # MATLAB: isequal(sorted_angles(i), sorted_angles(i+1))
        if sorted_angles[i] == sorted_angles[i + 1]:

            # Duplicate found
            candidate = sorted_angles[i + 1]
            tries = 0
            max_tries = int(1e6)

            while tries < max_tries:
                candidate = np.mod(candidate + dA, 2 * np.pi)

                # MATLAB: ~ismember(candidate, sorted_angles(1:i)) && ~ismember(candidate, sorted_angles(i+2:end))
                if (not np.isin(candidate, sorted_angles[:i + 1])) and \
                   (not np.isin(candidate, sorted_angles[i + 2:])):

                    sorted_angles[i + 1] = candidate
                    changed = True
                    break

                tries += 1

            if tries >= max_tries:
                raise RuntimeError("Failed to resolve duplicate after many attempts.")

            # Re-sort (MATLAB behavior)
            resort_idx = np.argsort(sorted_angles)
            sorted_angles = sorted_angles[resort_idx]
            sorted_idx = sorted_idx[resort_idx]

            # restart logic
            i = max(i - 1, 0)

        else:
            i += 1

    # Reconstruct original ordering (MATLAB behavior)
    angles_fixed = np.empty_like(sorted_angles)
    angles_fixed[sorted_idx] = sorted_angles

    return angles_fixed, changed


def adding_back_leaves_EA(x, coords, coords_ra):
    n = x.shape[0]
    coords_new = np.zeros((n, 2))
    degree = np.asarray(x.getnnz(axis=1)).ravel()

    coords_new[:, 1] = __set_radial_coordinates(x)

    # Build global → embedded index map
    non_leave_indices = np.where(degree > 1)[0]
    n_embedded = coords.shape[0]
    if len(non_leave_indices) != n_embedded:
        deficit = n_embedded - len(non_leave_indices)
        extra = np.where(degree == 1)[0][:deficit]
        non_leave_indices = np.sort(np.concatenate([non_leave_indices, extra]))
    assert len(non_leave_indices) == n_embedded, \
        f"Mismatch: {len(non_leave_indices)} vs {n_embedded}"
    global_to_emb = {int(g): e for e, g in enumerate(non_leave_indices)}

    # ea_dist: exact equidistant step of the EA embedding
    ea_dist = 2 * np.pi / n_embedded

    # Nearest neighbour in RAA space (vectorised)
    raa_angles = coords_ra[:, 0]
    diff = np.abs(raa_angles[:, None] - raa_angles[None, :])
    diff = np.minimum(diff, 2 * np.pi - diff)
    np.fill_diagonal(diff, np.inf)
    nearest_neighbor_emb = np.argmin(diff, axis=1)

    # Assign non-leaf angular coords from EA embedding
    non_leave_index = 0
    for i in range(n):
        if degree[i] != 1:
            coords_new[i, 0] = coords[non_leave_index, 0]
            non_leave_index += 1

    # Group leaves by parent
    parent_to_leaves = defaultdict(list)
    for i in range(n):
        if degree[i] == 1:
            parent = int(x[i, :].indices[0])
            parent_to_leaves[parent].append(i)

    # ══ DIAGNOSTIC 1: sanity check sizes ════════════════════════════
    # print(f"[DIAG] n={n}, n_embedded={n_embedded}, coords_ra.shape={coords_ra.shape}")
    # print(f"[DIAG] non_leave_indices range: {non_leave_indices.min()} - {non_leave_indices.max()}")
    # for p in parent_to_leaves:
    #     if degree[p] == 1:
    #         print(f"[DIAG] WARNING: parent {p} has degree 1 — it's a leaf itself!")
# ════════════════════════════════════════════════════════════════

    # Assign leaf angles with explicit ea_dist separation
    for parent, leaves in parent_to_leaves.items():
        parent_emb = global_to_emb[parent]
        nn_emb = nearest_neighbor_emb[parent_emb]

        direction = compare_angles(coords[parent_emb, 0], coords[nn_emb, 0])
        if direction == 0:
            direction = 1

        n_leaves = len(leaves)

            # ══ DIAGNOSTIC 2: print per-parent details ══════════════════
        # print(f"[DIAG] parent={parent} | ea_parent={coords_new[parent,0]:.4f} | "
        #       f"ea_nn={coords[nn_emb,0]:.4f} | direction={direction:+d} | n_leaves={len(leaves)}")
    # ════════════════════════════════════════════════════════════

        # Cap fan arc only when ideal span would wrap around (> full circle)
        # On large graphs ea_dist is tiny so this cap never triggers
        ideal_half_span = (n_leaves - 1) / 2.0 * ea_dist
        max_half_span = np.pi if ideal_half_span > np.pi else ideal_half_span
        half_span = max_half_span
        step = (2 * half_span) / (n_leaves - 1) if n_leaves > 1 else ea_dist
        clamped = ideal_half_span > np.pi

        # Step 1.2: center = angular midpoint between parent and NN in EA space
        parent_ea = coords_new[parent, 0]
        nn_ea     = coords[nn_emb, 0]
        d_to_nn   = np.mod(nn_ea - parent_ea, 2 * np.pi)   # CCW distance parent → nn
        if direction == -1:   # nn is CCW from parent
            center = np.mod(parent_ea + d_to_nn / 2, 2 * np.pi)
        else:                 # nn is CW from parent
            center = np.mod(parent_ea - (2 * np.pi - d_to_nn) / 2, 2 * np.pi)

        if n_leaves == 1:
            coords_new[leaves[0], 0] = np.mod(center, 2 * np.pi)
        else:
            for k, leaf in enumerate(leaves):
                offset = -half_span + k * step
                coords_new[leaf, 0] = np.mod(center + offset, 2 * np.pi)

        # ── Grid-landing check: nudge entire fan if any leaf is on a grid slot ──
        on_grid = any(
            min(coords_new[leaf, 0] % ea_dist,
                ea_dist - coords_new[leaf, 0] % ea_dist) < 1e-9
            for leaf in leaves
        )
        if on_grid:
            nudge = 0.1 * ea_dist  # small irrational-like shift, preserves fan shape
            for leaf in leaves:
                coords_new[leaf, 0] = np.mod(coords_new[leaf, 0] + nudge, 2 * np.pi)

    # ══ DIAGNOSTIC 3: detailed collision report ═════════════════════
    # all_angles = coords_new[:, 0]
    # leaf_indices_list = [i for i in range(n) if degree[i] == 1]
    # non_leaf_indices_list = [i for i in range(n) if degree[i] != 1]
    # leaf_angles = all_angles[leaf_indices_list]
    # non_leaf_angles = all_angles[non_leaf_indices_list]
    # collisions = np.sum(np.isin(np.round(leaf_angles, 9), np.round(non_leaf_angles, 9)))

    # non_leaf_set = set(np.round(all_angles[non_leaf_indices_list], 9))
    # for i in leaf_indices_list:
    #     a = round(all_angles[i], 9)
    #     if a in non_leaf_set:
    #         parent_of_leaf = int(x[i, :].indices[0])
    #         print(f"[COLLISION] leaf={i} parent={parent_of_leaf} "
    #               f"leaf_angle={all_angles[i]:.4f} "
    #               f"ea_parent={coords_new[parent_of_leaf, 0]:.4f}")

    # print(f"[DIAG] Leaf-nonleaf angle collisions before adjust_duplicates: {collisions}")
    # print(f"[DIAG] ea_dist={ea_dist:.8f} rad = {np.degrees(ea_dist):.6f} deg")
    # ════════════════════════════════════════════════════════════════

    coords_new[:, 0] = adjust_duplicates(coords_new[:, 0])

    return coords_new