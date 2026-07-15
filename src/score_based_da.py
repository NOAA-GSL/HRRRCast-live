"""
Observation operators for score-based data assimilation (SDA).

This module implements sparse and local observation maps H together with their
adjoints H_adjoint for use in analytical likelihood guidance.

Notation:
    - x: full model state on the grid, typically shape (batch, lat, lon, channels)
    - y = H(x): observations in observation space
    - H_adjoint(r): adjoint action that maps an observation-space residual r back
      to the full state space

Adjoint interpretation:
    For a general nonlinear observation map H(x), the relevant object in data
    assimilation is the transpose of the Jacobian, J_H(x)^T.

    All observation operators currently implemented in this file are linear in x
    once their indices and weights are constructed. Therefore:

        J_H(x) = H
        H_adjoint = H^T = (dH/dx)^T

    This is why the guidance computation can use H_adjoint directly without
    automatic differentiation.

Operator families in this module:
    - Index operator: direct sparse sampling at exact grid points
    - Interpolation operator: weighted off-grid sampling using fixed stencils
    - Neighborhood operator: local averaging using fixed window kernels

Implementation pattern:
    Each operator factory precomputes observation-space indices and weights,
    then returns closures for:
        - H: gather -> weight -> reduce
        - H_adjoint: expand residual -> weight -> scatter

    The returned adjoint tensors are sparse in full state space. Entries that are
    never addressed by the operator remain zero. If multiple observation terms map
    to the same grid entry, their adjoint contributions are summed.
"""

import numpy as np
import tensorflow as tf
import logging

from typing import Optional, Callable, Tuple, List

from diffusion_params import ALPHA_BAR

# ==== Score-Based Data Assimilation (SDA) Observation Operators ====

def create_observation_operator_index(
    grid_shape: Tuple[int, int],
    num_variables: int,
    variable_indices: Optional[List[int]] = None,
    station_locations: Optional[List[Tuple[int, int]]] = None,
    obs_mask: Optional[np.ndarray] = None,
) -> Tuple[Callable, Callable]:
    """
    Create index-based linear observation operator H and adjoint H^T for sparse observations.
    
    Supports:
    - Selecting subset of variables (e.g., last k channels)
    - Selecting subset of spatial locations (e.g., station locations)
    - Combined: specific variables at specific locations
    - Custom observation mask
    
    Args:
        grid_shape (Tuple[int, int]): Spatial grid shape (lat, lon).
        num_variables (int): Total number of variables/channels.
        variable_indices (List[int], optional): Indices of observed variables.
                                               If None, observes all variables.
                                               Example: [2, 3] for last 2 variables.
        station_locations (List[Tuple[int, int]], optional): List of (lat_idx, lon_idx)
                                                            for observation locations.
                                                            If None, observes all locations.
        obs_mask (np.ndarray, optional): Custom observation mask of shape
                                        (lat, lon, num_variables).
                                        Takes precedence over other options.
    
    Returns:
        Tuple[H, H_adjoint]:
            - H(x): Forward operator, extracts observed components
            - H_adjoint(residual): Adjoint operator, scatters back to full space

        Notes:
            - This operator is linear in x. Its Jacobian is H itself, so
                H_adjoint = H^T = (dH/dx)^T.
            - H_adjoint returns a sparse full-state tensor with zeros at all
                unobserved variable-location entries.
            - If duplicate indices were ever provided, adjoint contributions at the
                same grid entry would sum.
    
    Examples:
        # Example 1: Observe last 2 variables everywhere
        H, H_adj = create_observation_operator_index(
            grid_shape=(64, 64),
            num_variables=5,
            variable_indices=[3, 4]
        )
        
        # Example 2: Observe all variables at specific stations
        stations = [(10, 20), (30, 40), (50, 55)]
        H, H_adj = create_observation_operator_index(
            grid_shape=(64, 64),
            num_variables=5,
            station_locations=stations
        )
        
        # Example 3: Observe last 2 variables at 5 stations
        H, H_adj = create_observation_operator_index(
            grid_shape=(64, 64),
            num_variables=5,
            variable_indices=[3, 4],
            station_locations=[(10, 20), (30, 40), (50, 55), (15, 45), (55, 10)]
        )
        
        # Example 4: Custom mask
        custom_mask = np.zeros((64, 64, 5))
        custom_mask[10:20, 30:40, 3:5] = 1.0  # observe region + last 2 vars
        H, H_adj = create_observation_operator_index(
            grid_shape=(64, 64),
            num_variables=5,
            obs_mask=custom_mask
        )
    """
    
    lat_size, lon_size = grid_shape
    
    # Create observation mask
    if obs_mask is not None:
        # Use provided mask
        mask_np = np.asarray(obs_mask, dtype=np.float32)

    else:
        # Build mask from variable_indices and station_locations
        mask_np = np.zeros((lat_size, lon_size, num_variables), dtype=np.float32)
        
        if station_locations is not None:
            # Sparse spatial locations
            if variable_indices is not None:
                # Specific variables at specific locations
                for lat_idx, lon_idx in station_locations:
                    for var_idx in variable_indices:
                        mask_np[lat_idx, lon_idx, var_idx] = 1.0
            else:
                # All variables at specific locations
                for lat_idx, lon_idx in station_locations:
                    mask_np[lat_idx, lon_idx, :] = 1.0
        else:
            # Full grid observations
            if variable_indices is not None:
                # Specific variables everywhere
                for var_idx in variable_indices:
                    mask_np[:, :, var_idx] = 1.0
            else:
                # All variables everywhere (identity)
                mask_np[:, :, :] = 1.0
        
    selected_vars_np = np.where(np.any(mask_np > 0, axis=(0, 1)))[0].astype(np.int32)
    if selected_vars_np.size == 0:
        raise ValueError("Observation mask selects no variables.")

    # Keep exact station count per selected variable.
    selected_locs_linear_per_var: list[tf.Tensor] = []
    for var_idx in selected_vars_np:
        locs_var = np.argwhere(mask_np[:, :, var_idx] > 0).astype(np.int32)
        if locs_var.size == 0:
            continue
        locs_linear = locs_var[:, 0] * lon_size + locs_var[:, 1]
        selected_locs_linear_per_var.append(tf.constant(locs_linear, dtype=tf.int32))

    # Remove variables with zero selected stations.
    selected_vars_np = np.array(
        [v for v in selected_vars_np if np.any(mask_np[:, :, v] > 0)], dtype=np.int32
    )
    for i, var_idx in enumerate(selected_vars_np):
        logging.debug(f"Variable {var_idx}: {selected_locs_linear_per_var[i].shape[0]} observed locations.")

    if selected_vars_np.size == 0:
        raise ValueError("Observation mask selects no variable-location pairs.")
    
    # Precompute scatter indices for H_adjoint at creation time.
    # Build concatenated scatter indices once, reuse at every inference call.
    all_batch_multipliers = []
    all_loc_ids_static = []
    all_var_ids_static = []

    for i, var_idx in enumerate(selected_vars_np):
        locs_linear = selected_locs_linear_per_var[i]
        # Extract static shape; locs_linear is a tf.Tensor created with tf.constant
        locs_numpy = locs_linear.numpy()
        n_obs_i = len(locs_numpy)
        
        # For this variable: location IDs (tiled across batch at inference)
        all_loc_ids_static.append(locs_numpy)
        
        # Variable ID (constant for all observations of this variable)
        all_var_ids_static.append(np.full(n_obs_i, var_idx, dtype=np.int32))
        
        # Batch multiplier: how many obs per batch for this variable
        all_batch_multipliers.append(n_obs_i)

    # Concatenate static indices
    scatter_loc_ids_static = np.concatenate(all_loc_ids_static, axis=0).astype(np.int32)
    scatter_var_ids_static = np.concatenate(all_var_ids_static, axis=0).astype(np.int32)
    scatter_batch_multiplier = np.array(all_batch_multipliers, dtype=np.int32)  # per-variable counts
    total_observations = int(scatter_loc_ids_static.shape[0])
    obs_split_sizes = tuple(int(count) for count in scatter_batch_multiplier)
    
    # Convert to TensorFlow constants
    scatter_loc_ids_const = tf.constant(scatter_loc_ids_static, dtype=tf.int32)
    scatter_var_ids_const = tf.constant(scatter_var_ids_static, dtype=tf.int32)
    scatter_batch_multiplier_const = tf.constant(scatter_batch_multiplier, dtype=tf.int32)
    
    def H(x: tf.Tensor) -> tf.Tensor:
        """
        Forward observation operator: extract observed components.
        
        Args:
            x: Full state, shape (..., lat, lon, channels)
        
        Returns:
            List of observed tensors, one per selected variable.
            Each tensor has shape (batch, n_stations_for_variable).
        """
        x = tf.cast(x, tf.float32)
        batch_size = tf.shape(x)[0]
        x_flat = tf.reshape(x, [batch_size, lat_size * lon_size, num_variables])
        x_flat = tf.transpose(x_flat, perm=[0, 2, 1])

        batch_ids = tf.repeat(tf.range(batch_size, dtype=tf.int32), total_observations)
        var_ids_all = tf.tile(scatter_var_ids_const, [batch_size])
        loc_ids_all = tf.tile(scatter_loc_ids_const, [batch_size])
        gather_indices = tf.stack([batch_ids, var_ids_all, loc_ids_all], axis=1)

        gathered = tf.gather_nd(x_flat, gather_indices)
        gathered = tf.reshape(gathered, [batch_size, total_observations])

        return tf.split(gathered, obs_split_sizes, axis=1)
    
    def H_adjoint(residual: tf.Tensor) -> tf.Tensor:
        """
        Adjoint (transpose) of observation operator: scatter residuals back.
        
        Args:
            residual: Observation-space residual as list/tuple with one tensor per selected variable.
                     Each tensor has shape (batch, n_stations_for_variable)
                     or (n_stations_for_variable,).
        
        Returns:
            Full-space gradient, shape (..., lat, lon, channels)
        """
        if not isinstance(residual, (list, tuple)):
            raise TypeError("H_adjoint expects residual as list/tuple per selected variable.")

        if len(residual) != len(selected_vars_np):
            raise ValueError(
                f"H_adjoint residual length {len(residual)} does not match selected vars {len(selected_vars_np)}."
            )

        first = tf.cast(residual[0], tf.float32)
        if first.shape.rank == 1:
            batch_size = 1
        else:
            batch_size = tf.shape(first)[0]

        # Flatten all residuals into single concatenated vector
        all_updates = []
        for i in range(len(selected_vars_np)):
            res_i = tf.cast(residual[i], tf.float32)
            if res_i.shape.rank == 1:
                res_i = tf.expand_dims(res_i, axis=0)
            all_updates.append(tf.reshape(res_i, [-1]))
        updates = tf.concat(all_updates, axis=0)
        
        # Build batch IDs using precomputed per-variable counts
        total_obs = tf.reduce_sum(scatter_batch_multiplier_const)
        batch_ids = tf.repeat(tf.range(batch_size, dtype=tf.int32), total_obs)
        
        # Tile precomputed location/variable IDs across batch
        loc_ids_all = tf.tile(scatter_loc_ids_const, [batch_size])
        var_ids_all = tf.tile(scatter_var_ids_const, [batch_size])

        # Single scatter_nd call.
        scatter_indices = tf.stack([batch_ids, loc_ids_all, var_ids_all], axis=1)
        full_flat = tf.scatter_nd(
            scatter_indices,
            updates,
            [batch_size, lat_size * lon_size, num_variables],
        )

        result = tf.reshape(full_flat, [batch_size, lat_size, lon_size, num_variables])
        
        return result
    
    # Attach metadata to H and H_adjoint for use in guidance computation
    H.selected_vars = selected_vars_np
    H_adjoint.selected_vars = selected_vars_np
    
    return H, H_adjoint


def _lagrange_weights_1d(
    f: np.ndarray,
    size: int,
    order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D Lagrange interpolation node indices and weights.

    Args:
        f:     Fractional positions in grid-index coordinates, shape (N,).
        size:  Grid size along this dimension (for boundary clamping).
        order: Polynomial order: 1 (linear), 2 (quadratic), 3 (cubic).

    Returns:
        nodes:   Integer node indices, shape (N, order+1), clamped to [0, size-1].
        weights: Lagrange basis weights, shape (N, order+1). Partition of unity.
    """
    i0 = np.floor(f).astype(np.int32)
    t = (f - i0).astype(np.float32)  # fractional offset in [0, 1)

    if order == 1:
        # Stencil: [i0, i0+1]
        nodes = np.stack([
            np.clip(i0,     0, size - 1),
            np.clip(i0 + 1, 0, size - 1),
        ], axis=1)
        w = np.stack([1.0 - t, t], axis=1)

    elif order == 2:
        # Stencil: [i0-1, i0, i0+1]; Lagrange nodes at {-1, 0, +1} relative to i0.
        # L_{-1}(t) = t(t-1)/2,  L_0(t) = 1-t^2,  L_{+1}(t) = t(t+1)/2
        nodes = np.stack([
            np.clip(i0 - 1, 0, size - 1),
            np.clip(i0,     0, size - 1),
            np.clip(i0 + 1, 0, size - 1),
        ], axis=1)
        w = np.stack([
            t * (t - 1.0) / 2.0,
            1.0 - t * t,
            t * (t + 1.0) / 2.0,
        ], axis=1)

    else:  # order == 3
        # Stencil: [i0-1, i0, i0+1, i0+2]; Lagrange nodes at {-1, 0, 1, 2} relative to i0.
        # Cubic Lagrange basis evaluated at t in [0, 1):
        #   L_{-1}(t) = -t(t-1)(t-2)/6
        #   L_0(t)    =  (t+1)(t-1)(t-2)/2
        #   L_{+1}(t) = -(t+1)t(t-2)/2
        #   L_{+2}(t) =  (t+1)t(t-1)/6
        nodes = np.stack([
            np.clip(i0 - 1, 0, size - 1),
            np.clip(i0,     0, size - 1),
            np.clip(i0 + 1, 0, size - 1),
            np.clip(i0 + 2, 0, size - 1),
        ], axis=1)
        w = np.stack([
            -t * (t - 1.0) * (t - 2.0) / 6.0,
             (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0,
            -(t + 1.0) * t * (t - 2.0) / 2.0,
             (t + 1.0) * t * (t - 1.0) / 6.0,
        ], axis=1)

    return nodes.astype(np.int32), w.astype(np.float32)


def _compute_interp_stencil(
    lat_f: np.ndarray,
    lon_f: np.ndarray,
    lat_size: int,
    lon_size: int,
    order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D tensor-product interpolation stencil via outer product of 1D Lagrange stencils.

    Args:
        lat_f:    Fractional lat positions, shape (N,).
        lon_f:    Fractional lon positions, shape (N,).
        lat_size: Grid size in lat dimension.
        lon_size: Grid size in lon dimension.
        order:    Polynomial order (1, 2, or 3).

    Returns:
        neigh_locs: Linear grid indices of stencil neighbors, shape (N, K)
                    where K = (order+1)^2.
        neigh_w:    Interpolation weights, shape (N, K). Each row sums to 1.
    """
    lat_nodes, lat_w = _lagrange_weights_1d(lat_f, lat_size, order)  # (N, S)
    lon_nodes, lon_w = _lagrange_weights_1d(lon_f, lon_size, order)  # (N, S)

    N = lat_f.shape[0]
    K = (order + 1) ** 2

    # Outer product over lat and lon stencil nodes
    lat_n  = lat_nodes[:, :, np.newaxis]  # (N, S, 1)
    lon_n  = lon_nodes[:, np.newaxis, :]  # (N, 1, S)
    lat_ww = lat_w[:, :, np.newaxis]      # (N, S, 1)
    lon_ww = lon_w[:, np.newaxis, :]      # (N, 1, S)

    neigh_locs = (lat_n * lon_size + lon_n).reshape(N, K).astype(np.int32)
    neigh_w    = (lat_ww * lon_ww).reshape(N, K).astype(np.float32)

    return neigh_locs, neigh_w


def create_observation_operator_interp(
    grid_shape: Tuple[int, int],
    num_variables: int,
    variable_indices: Optional[List[int]] = None,
    station_locations: Optional[List[Tuple[float, float]]] = None,
    order: int = 1,
) -> Tuple[Callable, Callable]:
    """
    Create interpolation-based linear observation operator H and adjoint H^T.

    Supports:
    - Selecting subset of variables (e.g., last k channels)
    - Off-grid station locations specified as fractional grid indices
    - Bilinear (order=1), biquadratic (order=2), or bicubic (order=3) interpolation
    - Exact adjoint H^T using the same stencil weights as H

    Station locations are interpreted in fractional grid-index coordinates
    (lat_idx, lon_idx), not geographic degrees. Interpolation uses a
    (order+1) × (order+1) tensor-product Lagrange stencil. The adjoint H^T
    distributes each station residual back to those same neighbors using
    the same Lagrange weights.

    Args:
        grid_shape (Tuple[int, int]): Spatial grid shape (lat, lon).
        num_variables (int): Total number of variables/channels.
        variable_indices (List[int], optional): Indices of observed variables.
                                               If None, observes all variables.
                                               Example: [2, 3] for last 2 variables.
        station_locations (List[Tuple[float, float]], optional): List of
                                               fractional (lat_idx, lon_idx)
                                               observation locations.
                                               Example: [(10.25, 20.75), (30.5, 40.1)].
                                               Must be provided and non-empty.
        order (int): Interpolation polynomial order. Must be 1, 2, or 3.
                     1 = bilinear     (2×2 stencil, K=4  neighbors per station)
                     2 = biquadratic  (3×3 stencil, K=9  neighbors per station)
                     3 = bicubic      (4×4 stencil, K=16 neighbors per station)
                     Higher orders are smoother but may produce small negative
                     weights near grid boundaries due to clamping. Default: 1.

    Returns:
        Tuple[H, H_adjoint]:
            - H(x): Forward operator, interpolates observations
                    at station locations
            - H_adjoint(residual): Adjoint operator, scatters weighted
                    residuals back to full space

    Notes:
        - This operator is linear in x once interpolation stencils are fixed.
            Therefore H_adjoint = H^T = (dH/dx)^T.
        - Out-of-bounds station indices are clipped to valid grid range.
        - If a station lies exactly on a grid point, interpolation reduces
          to direct point sampling at that location.
        - For order >= 2, stencil nodes that fall outside the grid are clamped,
          which reduces effective polynomial order at boundaries.
        - Biquadratic and bicubic weights can be negative (Runge phenomenon
          suppression near boundaries); this is mathematically correct.

    Examples:
        # Example 1: Bilinear (default) — observe last 2 variables at off-grid stations
        H, H_adj = create_observation_operator_interp(
            grid_shape=(64, 64),
            num_variables=5,
            variable_indices=[3, 4],
            station_locations=[(10.25, 20.75), (30.5, 40.1), (50.0, 55.0)]
        )

        # Example 2: Biquadratic — smoother gradients
        H, H_adj = create_observation_operator_interp(
            grid_shape=(64, 64),
            num_variables=5,
            station_locations=[(12.1, 8.9), (31.3, 44.7)],
            order=2
        )

        # Example 3: Bicubic — highest-order interpolation
        H, H_adj = create_observation_operator_interp(
            grid_shape=(64, 64),
            num_variables=5,
            variable_indices=[3, 4],
            station_locations=[(10.25, 20.75), (30.5, 40.1)],
            order=3
        )
    """
    lat_size, lon_size = grid_shape

    if station_locations is None or len(station_locations) == 0:
        raise ValueError("station_locations must be provided for interpolation operator.")

    if order not in (1, 2, 3):
        raise ValueError(f"order must be 1, 2, or 3; got {order}.")

    if variable_indices is None:
        selected_vars_np = np.arange(num_variables, dtype=np.int32)
    else:
        selected_vars_np = np.asarray(variable_indices, dtype=np.int32)

    if selected_vars_np.size == 0:
        raise ValueError("No variables selected for interpolation operator.")

    stations_np = np.asarray(station_locations, dtype=np.float32)
    if stations_np.ndim != 2 or stations_np.shape[1] != 2:
        raise ValueError("station_locations must be a list of (lat_idx, lon_idx).")

    # Clamp stations to valid index range before building stencil neighbors.
    lat_f = np.clip(stations_np[:, 0], 0.0, float(lat_size - 1))
    lon_f = np.clip(stations_np[:, 1], 0.0, float(lon_size - 1))

    neigh_locs, neigh_w = _compute_interp_stencil(lat_f, lon_f, lat_size, lon_size, order)

    num_sel_vars = int(selected_vars_np.shape[0])
    num_stations = int(stations_np.shape[0])
    neighbors_per_station = (order + 1) ** 2
    total_obs_per_var = num_stations
    total_points_per_var = num_stations * neighbors_per_station
    total_points_all_vars = num_sel_vars * total_points_per_var

    var_ids_points = np.repeat(selected_vars_np, total_points_per_var).astype(np.int32)
    loc_ids_points = np.tile(neigh_locs.reshape(-1), num_sel_vars).astype(np.int32)
    w_points = np.tile(neigh_w.reshape(-1), num_sel_vars).astype(np.float32)

    var_ids_points_const = tf.constant(var_ids_points, dtype=tf.int32)
    loc_ids_points_const = tf.constant(loc_ids_points, dtype=tf.int32)
    w_points_const = tf.constant(w_points, dtype=tf.float32)

    obs_split_sizes = tuple([total_obs_per_var] * num_sel_vars)

    def H(x: tf.Tensor) -> tf.Tensor:
        """
        Forward observation operator: interpolate observed values at station locations.

        For each station, the value is a weighted sum of (order+1)^2 surrounding grid
        points using precomputed tensor-product Lagrange weights.

        Args:
            x: Full model state, shape (batch, lat, lon, channels).

        Returns:
            List of interpolated observation tensors, one per selected variable.
            Each tensor has shape (batch, n_stations).
        """
        x = tf.cast(x, tf.float32)
        batch_size = tf.shape(x)[0]

        # [B, V, L] for indexed gather on (batch, var, location)
        x_flat = tf.reshape(x, [batch_size, lat_size * lon_size, num_variables])
        x_flat = tf.transpose(x_flat, perm=[0, 2, 1])

        batch_ids = tf.repeat(tf.range(batch_size, dtype=tf.int32), total_points_all_vars)
        var_ids_all = tf.tile(var_ids_points_const, [batch_size])
        loc_ids_all = tf.tile(loc_ids_points_const, [batch_size])

        gather_indices = tf.stack([batch_ids, var_ids_all, loc_ids_all], axis=1)
        gathered = tf.gather_nd(x_flat, gather_indices)

        w_all = tf.tile(w_points_const, [batch_size])
        weighted = gathered * w_all

        weighted = tf.reshape(
            weighted,
            [batch_size, num_sel_vars, num_stations, neighbors_per_station],
        )
        obs = tf.reduce_sum(weighted, axis=3)  # [B, Vsel, Nstations]

        return tf.split(tf.reshape(obs, [batch_size, num_sel_vars * num_stations]), obs_split_sizes, axis=1)

    def H_adjoint(residual: tf.Tensor) -> tf.Tensor:
        """
        Adjoint (transpose) of interpolation operator: scatter weighted residuals back to grid.

        Distributes each station residual to its (order+1)^2 neighboring grid points using
        the same Lagrange weights used in H, ensuring H^T is the exact mathematical
        transpose of H (i.e., <H x, r> = <x, H^T r> for all x, r).

        Args:
            residual: Observation-space residual as list/tuple with one tensor per selected
                     variable. Each tensor has shape (batch, n_stations) or (n_stations,).

        Returns:
            Full-space gradient, shape (batch, lat, lon, channels).
        """
        if not isinstance(residual, (list, tuple)):
            raise TypeError("H_adjoint expects residual as list/tuple per selected variable.")

        if len(residual) != num_sel_vars:
            raise ValueError(
                f"H_adjoint residual length {len(residual)} does not match selected vars {num_sel_vars}."
            )

        first = tf.cast(residual[0], tf.float32)
        if first.shape.rank == 1:
            batch_size = 1
        else:
            batch_size = tf.shape(first)[0]

        residual_list = []
        for i in range(num_sel_vars):
            res_i = tf.cast(residual[i], tf.float32)
            if res_i.shape.rank == 1:
                res_i = tf.expand_dims(res_i, axis=0)
            residual_list.append(res_i)

        res_cat = tf.stack(residual_list, axis=1)  # [B, Vsel, Nstations]
        res_expanded = tf.expand_dims(res_cat, axis=3)  # [B, Vsel, Nstations, 1]
        w_reshaped = tf.reshape(w_points_const, [num_sel_vars, num_stations, neighbors_per_station])
        weighted = res_expanded * w_reshaped[tf.newaxis, :, :, :]  # [B, Vsel, Nstations, 4]

        updates = tf.reshape(weighted, [-1])

        batch_ids = tf.repeat(tf.range(batch_size, dtype=tf.int32), total_points_all_vars)
        loc_ids_all = tf.tile(loc_ids_points_const, [batch_size])
        var_ids_all = tf.tile(var_ids_points_const, [batch_size])

        scatter_indices = tf.stack([batch_ids, loc_ids_all, var_ids_all], axis=1)
        full_flat = tf.scatter_nd(
            scatter_indices,
            updates,
            [batch_size, lat_size * lon_size, num_variables],
        )

        return tf.reshape(full_flat, [batch_size, lat_size, lon_size, num_variables])

    H.selected_vars = selected_vars_np
    H_adjoint.selected_vars = selected_vars_np

    return H, H_adjoint


def create_observation_operator_neighborhood(
    grid_shape: Tuple[int, int],
    num_variables: int,
    variable_indices: Optional[List[int]] = None,
    station_locations: Optional[List[Tuple[float, float]]] = None,
    window_size: int = 3,
    kernel: str = "uniform",
    sigma: float = 1.0,
) -> Tuple[Callable, Callable]:
    """
    Create neighborhood-averaging linear observation operator H and adjoint H^T.

    This operator observes local neighborhood aggregates around each station instead
    of point values. It supports both box averaging (uniform kernel) and Gaussian-
    weighted averaging over an odd-sized square window.

    Args:
        grid_shape (Tuple[int, int]): Spatial grid shape (lat, lon).
        num_variables (int): Total number of variables/channels.
        variable_indices (List[int], optional): Indices of observed variables.
                                               If None, observes all variables.
        station_locations (List[Tuple[float, float]], optional): Station centers in
                                               grid-index coordinates. Fractional values
                                               are rounded to nearest grid index.
                                               Must be provided and non-empty.
        window_size (int): Odd neighborhood width/height. 3 means 3x3, 5 means 5x5.
                          Must be >= 1 and odd. Default: 3.
        kernel (str): Neighborhood kernel type:
                     - "uniform": equal weights over the full window
                     - "gaussian": Gaussian weights using sigma
                     Default: "uniform".
        sigma (float): Gaussian stddev in grid-index units when kernel="gaussian".
                      Must be > 0. Default: 1.0.

    Returns:
        Tuple[H, H_adjoint]:
            - H(x): Forward operator, returns neighborhood-aggregated observations
            - H_adjoint(residual): Adjoint operator, scatters weighted residuals
              back to the full grid.

    Notes:
        - This operator is linear in x once station centers and kernel weights are
            fixed. Therefore H_adjoint = H^T = (dH/dx)^T.
        - Out-of-bounds neighborhood nodes are clipped to the nearest valid grid cell
          (replicate boundary condition).
        - The adjoint is exact for this boundary treatment because it uses the same
          clipped neighbor locations and identical kernel weights.
    """
    lat_size, lon_size = grid_shape

    if station_locations is None or len(station_locations) == 0:
        raise ValueError("station_locations must be provided for neighborhood operator.")

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(f"window_size must be odd and >= 1; got {window_size}.")

    kernel = str(kernel).lower()
    if kernel not in ("uniform", "gaussian"):
        raise ValueError(f"kernel must be 'uniform' or 'gaussian'; got {kernel}.")

    if kernel == "gaussian" and sigma <= 0:
        raise ValueError(f"sigma must be > 0 for gaussian kernel; got {sigma}.")

    if variable_indices is None:
        selected_vars_np = np.arange(num_variables, dtype=np.int32)
    else:
        selected_vars_np = np.asarray(variable_indices, dtype=np.int32)

    if selected_vars_np.size == 0:
        raise ValueError("No variables selected for neighborhood operator.")

    stations_np = np.asarray(station_locations, dtype=np.float32)
    if stations_np.ndim != 2 or stations_np.shape[1] != 2:
        raise ValueError("station_locations must be a list of (lat_idx, lon_idx).")

    center_lat = np.rint(stations_np[:, 0]).astype(np.int32)
    center_lon = np.rint(stations_np[:, 1]).astype(np.int32)
    center_lat = np.clip(center_lat, 0, lat_size - 1)
    center_lon = np.clip(center_lon, 0, lon_size - 1)

    radius = window_size // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.int32)
    dlat_grid, dlon_grid = np.meshgrid(offsets, offsets, indexing="ij")
    dlat = dlat_grid.reshape(-1)
    dlon = dlon_grid.reshape(-1)
    neighbors_per_station = int(dlat.shape[0])

    if kernel == "uniform":
        base_w = np.ones(neighbors_per_station, dtype=np.float32)
    else:
        dist2 = dlat.astype(np.float32) ** 2 + dlon.astype(np.float32) ** 2
        base_w = np.exp(-0.5 * dist2 / (float(sigma) ** 2)).astype(np.float32)
    base_w = base_w / np.sum(base_w)

    neigh_lat = np.clip(center_lat[:, np.newaxis] + dlat[np.newaxis, :], 0, lat_size - 1)
    neigh_lon = np.clip(center_lon[:, np.newaxis] + dlon[np.newaxis, :], 0, lon_size - 1)
    neigh_locs = (neigh_lat * lon_size + neigh_lon).astype(np.int32)
    neigh_w = np.tile(base_w[np.newaxis, :], [stations_np.shape[0], 1]).astype(np.float32)

    num_sel_vars = int(selected_vars_np.shape[0])
    num_stations = int(stations_np.shape[0])
    total_obs_per_var = num_stations
    total_points_per_var = num_stations * neighbors_per_station
    total_points_all_vars = num_sel_vars * total_points_per_var

    var_ids_points = np.repeat(selected_vars_np, total_points_per_var).astype(np.int32)
    loc_ids_points = np.tile(neigh_locs.reshape(-1), num_sel_vars).astype(np.int32)
    w_points = np.tile(neigh_w.reshape(-1), num_sel_vars).astype(np.float32)

    var_ids_points_const = tf.constant(var_ids_points, dtype=tf.int32)
    loc_ids_points_const = tf.constant(loc_ids_points, dtype=tf.int32)
    w_points_const = tf.constant(w_points, dtype=tf.float32)

    obs_split_sizes = tuple([total_obs_per_var] * num_sel_vars)

    def H(x: tf.Tensor) -> tf.Tensor:
        """
        Forward observation operator: neighborhood-averaged observations.

        For each station and selected variable, computes a weighted average over
        an odd-sized local window centered at the rounded station index.
        """
        x = tf.cast(x, tf.float32)
        batch_size = tf.shape(x)[0]

        x_flat = tf.reshape(x, [batch_size, lat_size * lon_size, num_variables])
        x_flat = tf.transpose(x_flat, perm=[0, 2, 1])

        batch_ids = tf.repeat(tf.range(batch_size, dtype=tf.int32), total_points_all_vars)
        var_ids_all = tf.tile(var_ids_points_const, [batch_size])
        loc_ids_all = tf.tile(loc_ids_points_const, [batch_size])

        gather_indices = tf.stack([batch_ids, var_ids_all, loc_ids_all], axis=1)
        gathered = tf.gather_nd(x_flat, gather_indices)

        w_all = tf.tile(w_points_const, [batch_size])
        weighted = gathered * w_all

        weighted = tf.reshape(
            weighted,
            [batch_size, num_sel_vars, num_stations, neighbors_per_station],
        )
        obs = tf.reduce_sum(weighted, axis=3)

        obs_flat = tf.reshape(obs, [batch_size, num_sel_vars * num_stations])
        return tf.split(obs_flat, obs_split_sizes, axis=1)

    def H_adjoint(residual: tf.Tensor) -> tf.Tensor:
        """
        Adjoint of neighborhood operator: scatter weighted residuals to neighborhoods.

        Uses the same neighborhood kernel and clipped locations as H, ensuring exact
        transpose consistency under the replicate-boundary convention.
        """
        if not isinstance(residual, (list, tuple)):
            raise TypeError("H_adjoint expects residual as list/tuple per selected variable.")

        if len(residual) != num_sel_vars:
            raise ValueError(
                f"H_adjoint residual length {len(residual)} does not match selected vars {num_sel_vars}."
            )

        first = tf.cast(residual[0], tf.float32)
        if first.shape.rank == 1:
            batch_size = 1
        else:
            batch_size = tf.shape(first)[0]

        residual_list = []
        for i in range(num_sel_vars):
            res_i = tf.cast(residual[i], tf.float32)
            if res_i.shape.rank == 1:
                res_i = tf.expand_dims(res_i, axis=0)
            residual_list.append(res_i)

        res_cat = tf.stack(residual_list, axis=1)
        res_expanded = tf.expand_dims(res_cat, axis=3)
        w_reshaped = tf.reshape(w_points_const, [num_sel_vars, num_stations, neighbors_per_station])
        weighted = res_expanded * w_reshaped[tf.newaxis, :, :, :]

        updates = tf.reshape(weighted, [-1])

        batch_ids = tf.repeat(tf.range(batch_size, dtype=tf.int32), total_points_all_vars)
        loc_ids_all = tf.tile(loc_ids_points_const, [batch_size])
        var_ids_all = tf.tile(var_ids_points_const, [batch_size])

        scatter_indices = tf.stack([batch_ids, loc_ids_all, var_ids_all], axis=1)
        full_flat = tf.scatter_nd(
            scatter_indices,
            updates,
            [batch_size, lat_size * lon_size, num_variables],
        )

        return tf.reshape(full_flat, [batch_size, lat_size, lon_size, num_variables])

    H.selected_vars = selected_vars_np
    H_adjoint.selected_vars = selected_vars_np

    return H, H_adjoint

# ==== Compute SDA guidance using observation operators ====

def compute_sda_guidance(
    x_0: tf.Tensor,
    y_obs: tf.Tensor | list[tf.Tensor] | tuple[tf.Tensor, ...],
    H: Callable,
    H_adjoint: Callable,
    t: tf.Tensor,
    weights: tf.Tensor | list | np.ndarray | None = None,  # Per-variable weights
    R_obs: float = 0.01, # [0.02, 0.2]
    gamma: float = 0.01, # [0.005, 0.05]
    guidance_scale: float = 0.3, # [0.05, 0.3]
    
) -> tf.Tensor:
    """
    Compute score-based data assimilation (SDA) guidance using observation operators.
    
    Computes the observation likelihood gradient with noise-aware variance scaling:
        σ²_t  = 1 − ᾱ_t
        V     = R_obs + γ·σ²_t / (γ·ᾱ_t + σ²_t)   (bounded posterior variance)
        ∇_{x_0} log p(y | x_0) = −H^T W·(H x_0 − y) / V

    where W are per-variable weights (optional). V is the posterior variance of x̂_0 given x_t 
    assuming x_0 ~ N(0, γI). It is bounded in [R_obs, R_obs + γ] for all t, ensuring guidance 
    is active throughout the full reverse trajectory (unlike the (1−ᾱ)/ᾱ·γ formula, which 
    diverges to ∞ at high-noise steps and kills guidance).
    
    This is the "guidance" signal that steers the diffusion reverse process
    toward being consistent with observations. No GradientTape is needed for the
    current operators in this module because they are linear in x_0 and expose
    their exact adjoints directly.
    
    Args:
        x_0 (tf.Tensor): Predicted x_0 from diffusion model.
                        Shape: (batch, lat, lon, channels) or (batch, ...)
        y_obs (tf.Tensor | list[tf.Tensor]): Observations in observation space.
                  For per-variable operator, pass list with one tensor
                  per selected variable.
        H (Callable): Forward observation operator H(x).
                     Extracts observed components from full state.
        H_adjoint (Callable): Adjoint (transpose) of H.
                     For the linear operators in this module,
                     H_adjoint = H^T = (dH/dx)^T and maps
                     observation-space residuals back to full state.
        t (tf.Tensor or int): Timestep index in the diffusion process.
                             Used to compute ALPHA_BAR_t = tf.gather(ALPHA_BAR, t).
        R_obs (float): Observation error variance.
                      Controls measurement uncertainty.
                      Default: 1.0
        gamma (float): Prior/model error variance scaling factor.
                      Controls relative weight of diffusion model uncertainty.
                      Default: 1.0
        guidance_scale (float): Multiplicative scaling for final guidance.
                               < 1.0: weaker constraint
                               = 1.0: theory-optimal
                               > 1.0: stronger constraint
                               Default: 1.0
        weights (tf.Tensor | list | np.ndarray, optional): Per-variable weights.
                      Shape: (num_channels,) or compatible with residual shape.
                      Used for consistent weighting with training loss.
                      If None, uniform weighting (all weights = 1.0).
                      Default: None
    
    Returns:
        tf.Tensor: Likelihood gradient in full state space.
                  Shape: same as x_0.
                  
                  The returned value is guidance in x_0 space.
                  Usage in posterior sampling:
                      score = compute_sda_guidance(x_0, y_obs, H, H_adjoint, t, 
                                                  R_obs=1.0, gamma=1.0)
    
    Examples:
        # Example 1: Sparse observations with H and H_adjoint
        H, H_adj = create_observation_operator(
            grid_shape=(64, 64),
            num_variables=5,
            variable_indices=[3, 4],
            station_locations=[(10, 20), (30, 40), (50, 55)]
        )
        
        score = compute_sda_guidance(
            x_0=model_prediction,
            y_obs=observations,
            H=H,
            H_adjoint=H_adj,
            t=timestep,
            R_obs=0.1,
            gamma=1.0,
            guidance_scale=1.0
        )
        
        # Example 2: With per-variable weights (consistent with training)
        weights = tf.constant([1.0, 1.0, 2.0, 2.0, 1.5])  # 5 variables
        score = compute_sda_guidance(
            x_0=model_prediction,
            y_obs=observations,
            H=H,
            H_adjoint=H_adj,
            t=timestep,
            R_obs=0.1,
            gamma=1.0,
            guidance_scale=1.0,
            weights=weights
        )
    """
    
    # Get cumulative alpha at timestep t
    ALPHA_BAR_t = tf.gather(ALPHA_BAR, t)

    # Posterior variance of x̂_0 given x_t with Gaussian prior x_0 ~ N(0, γI):
    #   Σ_t = γ(1−ᾱ_t) / (γᾱ_t + (1−ᾱ_t))
    # This is bounded in [0, γ] for all t. Using (1−ᾱ_t)/ᾱ_t instead (the
    # γ→∞ limit) diverges when ᾱ_t≈0 (early reverse steps), making V enormous
    # and guidance ≈ 0 for most of the trajectory when gamma != 0.
    sigma_sq = 1.0 - ALPHA_BAR_t
    posterior_var = gamma * sigma_sq / (gamma * ALPHA_BAR_t + sigma_sq + 1e-12)

    # Total variance: V = R_obs + Σ_t  (bounded in [R_obs, R_obs + gamma])
    V = tf.cast(R_obs + posterior_var, tf.float32)
    
    # Forward observation operator
    y_hat = H(x_0)

    # Residual in observation space
    residual = []
    selected_vars = H.selected_vars
    for i, (y_hat_i, y_obs_i) in enumerate(zip(y_hat, y_obs)):
        res_i = y_hat_i - y_obs_i
        if weights is not None:
            weights_f32 = tf.cast(weights, tf.float32)
            w_i = weights_f32[selected_vars[i]]
            res_i = res_i * w_i
        residual.append(res_i)
    weighted_residual = [res_i / V for res_i in residual]

    # Adjoint: map residual back to full state space
    # Divide by V for noise-aware scaling: H^T (H x_0 - y) / V
    score = -H_adjoint(weighted_residual)
 
    # Apply guidance scaling
    score = guidance_scale * score * tf.sqrt(sigma_sq)
    
    return score

# ==== Create station locations for observation operators ====

def create_station_locations(
    lat_size: int,
    lon_size: int,
    subsampling_factor: int,
    layout: str,
    operator_type: str,
    random_seed: int,
) -> list[tuple[float, float]]:
    """Create station locations with either uniform or random layout."""
    if subsampling_factor < 1:
        raise ValueError(f"subsampling_factor must be >= 1; got {subsampling_factor}.")

    if layout == "uniform":
        # interp and neighborhood both benefit from cell-centre positions (offset 0.5);
        # index uses exact grid-point indices (offset 0.0).
        offset = 0.0 if operator_type == "index" else 0.5
        lat_stations = np.arange(offset, lat_size, subsampling_factor, dtype=np.float32)
        lon_stations = np.arange(offset, lon_size, subsampling_factor, dtype=np.float32)
        return [(lat, lon) for lat in lat_stations for lon in lon_stations]

    # Random layout: match nominal count from a regular subsampled grid.
    n_stations = len(range(0, lat_size, subsampling_factor)) * len(range(0, lon_size, subsampling_factor))
    rng = np.random.default_rng(seed=random_seed)

    if operator_type == "index":
        lat_rand = rng.integers(0, lat_size, n_stations, dtype=np.int32).astype(np.float32)
        lon_rand = rng.integers(0, lon_size, n_stations, dtype=np.int32).astype(np.float32)
    else:
        lat_rand = rng.uniform(0.0, lat_size - 1, n_stations).astype(np.float32)
        lon_rand = rng.uniform(0.0, lon_size - 1, n_stations).astype(np.float32)

    return list(zip(lat_rand.tolist(), lon_rand.tolist()))

# ==== Build observation operator from runtime configuration ====

def build_observation_operator(
    args,
    lat_size: int,
    lon_size: int,
    num_variables: int,
) -> tuple[Callable, Callable]:
    """Build H and H_adjoint from runtime configuration."""
    station_locs = create_station_locations(
        lat_size=lat_size,
        lon_size=lon_size,
        subsampling_factor=args.subsampling_factor,
        layout=args.station_layout,
        operator_type=args.obs_operator,
        random_seed=args.obs_random_seed,
    )

    if len(station_locs) == 0:
        raise ValueError("No station locations were generated.")

    if args.obs_operator == "index":
        obs_mask = np.zeros((lat_size, lon_size, num_variables), dtype=np.float32)
        for lat, lon in station_locs:
            lat_idx = int(np.clip(np.rint(lat), 0, lat_size - 1))
            lon_idx = int(np.clip(np.rint(lon), 0, lon_size - 1))
            obs_mask[lat_idx, lon_idx, :] = 1.0

        logging.info(
            "Using index observation operator with %d %s stations (subsampling_factor=%d).",
            len(station_locs),
            args.station_layout,
            args.subsampling_factor,
        )
        return create_observation_operator_index(
            grid_shape=(lat_size, lon_size),
            num_variables=num_variables,
            obs_mask=obs_mask,
        )

    if args.obs_operator == "interp":
        logging.info(
            "Using interp observation operator with %d %s stations (order=%d, subsampling_factor=%d).",
            len(station_locs),
            args.station_layout,
            args.interp_order,
            args.subsampling_factor,
        )
        return create_observation_operator_interp(
            grid_shape=(lat_size, lon_size),
            num_variables=num_variables,
            station_locations=station_locs,
            order=args.interp_order,
        )

    # neighborhood
    window_size = args.neighborhood_window_size
    if window_size is None:
        window_size = args.subsampling_factor
        if window_size % 2 == 0:
            window_size -= 1
        window_size = max(window_size, 1)

    sigma = args.neighborhood_sigma
    if sigma is None:
        sigma = 0.425 * args.subsampling_factor

    logging.info(
        "Using neighborhood observation operator with %d %s stations (window_size=%d, kernel=%s, sigma=%.3f).",
        len(station_locs),
        args.station_layout,
        window_size,
        args.neighborhood_kernel,
        sigma,
    )
    return create_observation_operator_neighborhood(
        grid_shape=(lat_size, lon_size),
        num_variables=num_variables,
        station_locations=station_locs,
        window_size=window_size,
        kernel=args.neighborhood_kernel,
        sigma=sigma,
    )