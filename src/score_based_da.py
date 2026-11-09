"""
Observation operators for score-based data assimilation (SDA).

This module implements observation maps H together with their adjoint/Jacobian-
transpose actions H_adjoint for use in analytical likelihood guidance.

Notation:
    - x: full model state on the grid, typically shape (batch, lat, lon, channels)
    - y: observations in observation space
    - H: observation operator that maps full state x to observation space
    - r: residual in observation space, r = H(x) - y
    - H_adjoint(r): linear adjoint action that maps an observation-space residual
        r back to the full state space
    - H_adjoint(r, x): nonlinear Jacobian-transpose action J_H(x)^T r

Adjoint interpretation:
    For a general nonlinear observation map H(x), the relevant Jacobian object
    is J_H(x), and the data-assimilation update is based on the covariance-
    weighted residual:
        delta_x = J_H(x)^T R^{-1} (H(x) - y).

    The equivalent Kalman-gain term is therefore
        K_eff(x) = J_H(x)^T R^{-1},
    so the update can be written as
        delta_x = K_eff(x) (H(x) - y).

    In the diagonal-Gaussian approximation used here, R is represented by the
    observation variance V_t, so the update reduces to
        delta_x = J_H(x)^T (H(x) - y) / V_t.

    For the sparse/local stencil operators implemented in this file, the stencil
    geometry is fixed once the indices and weights are constructed, so the
    operator is effectively linear with respect to x within that fixed geometry:
        H(x) = H * x,
        J_H(x) = H,
        delta_x = H^T R^{-1} (H(x) - y).

    In the scalar-variance case used by this module, this becomes
        delta_x = H^T * (H(x) - y) / V_t,
    with
        V_t = R_obs + gamma * (1 - alpha_bar_t) / alpha_bar_t.

    The nonlinear adapter in this module follows the same pattern, but the
    transpose action is evaluated at the current state x via the operator's
    Jacobian or an explicit adjoint implementation.

    Nonlinear external operators and neural emulators can instead be adapted with
    create_nonlinear_observation_operator. Their adjoint requires the state x at
    which the Jacobian is evaluated.

Operator families in this module:
    - Index operator: direct sparse sampling at exact grid points
    - Interpolation operator: weighted off-grid sampling using fixed stencils
    - Neighborhood operator: local averaging using fixed window kernels
    - Footprint operator: uniform averaging over a circular grid-cell footprint
    - Finite-difference operator: centered gradients or a five-point Laplacian
    - Linear-combination operator: fixed combinations of colocated variable blocks
    - Composition adapter: chain a state-to-local operator with an observation transform
    - Nonlinear adapter: explicit external adjoint or TensorFlow automatic VJP

Implementation pattern:
    Each operator factory precomputes observation-space indices and weights,
    then returns closures for:
        - H: gather -> weight -> reduce
        - H_adjoint: expand residual -> weight -> scatter

    Atmospheric observation blocks may optionally provide full-column
    ``vertical_weight_profiles``. These replace the default two-level linear
    interpolation and are tensor-composed with the selected horizontal stencil.

    The returned adjoint tensors are sparse in full state space. Entries that are
    never addressed by the operator remain zero. If multiple observation terms map
    to the same grid entry, their adjoint contributions are summed.
"""
import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import tensorflow as tf

from diffusion_params import ALPHA_BAR

_StationLocation = tuple[float, float, float]
_StationLocations = list[_StationLocation]
_StationGroups = dict[str, _StationLocations]
_StationSource = _StationLocations | _StationGroups
_HorizontalStencil = tuple[np.ndarray, np.ndarray]
_HorizontalStencilBuilder = Callable[
    [_StationLocations, tuple[int, int]], _HorizontalStencil
]
_VerticalWeightProfiles = dict[int, np.ndarray]

# ==== Helpers for constructing obs operators ====

def _normalize_station_locations(station_locations) -> _StationLocations:
    """Normalize station locations to (lat, lon, level) triplets."""
    if station_locations is None:
        raise ValueError("station_locations must not be None.")

    normalized = []
    for loc in station_locations:
        if isinstance(loc, np.ndarray):
            loc = loc.tolist()

        if not isinstance(loc, (tuple, list)) or len(loc) != 3:
            raise ValueError(
                "Each station location must be exactly (lat, lon, level)."
            )

        lat = float(loc[0])
        lon = float(loc[1])
        level = float(loc[2])
        if not np.isfinite(lat) or not np.isfinite(lon):
            raise ValueError(f"Station location {loc} contains a non-finite latitude/longitude.")
        if not np.isfinite(level):
            raise ValueError(f"Station location {loc} contains a non-finite vertical level.")
        normalized.append((lat, lon, level))

    # All downstream stencil builders assume a consistent (lat, lon, level)
    # triplet format; this conversion makes grouped and shared station inputs
    # interchangeable.
    return normalized


def _validate_and_normalize_residuals(
    residual,
    expected_station_counts,
    operator_name: str,
) -> tuple[list[tf.Tensor], tf.Tensor | int]:
    """Validate adjoint residual structure and return rank-2 float tensors."""
    if not isinstance(residual, (list, tuple)):
        raise TypeError(
            f"{operator_name} expects a list/tuple with one tensor "
            "per selected variable."
        )
    if len(residual) != len(expected_station_counts):
        raise ValueError(
            f"{operator_name} residual length {len(residual)} does not "
            f"match selected vars {len(expected_station_counts)}."
        )

    ranks = [tf.convert_to_tensor(value).shape.rank for value in residual]
    if any(rank not in (1, 2) for rank in ranks):
        raise ValueError(
            f"{operator_name} residual tensors must have rank 1 or 2; "
            f"got ranks {ranks}."
        )
    if any(rank != ranks[0] for rank in ranks[1:]):
        raise ValueError(
            f"{operator_name} residual tensors must all be either "
            f"unbatched or batched; got ranks {ranks}."
        )

    normalized = []
    for i, (value, expected_count) in enumerate(zip(residual, expected_station_counts)):
        tensor = tf.cast(value, tf.float32)
        if ranks[0] == 1:
            tensor = tf.expand_dims(tensor, axis=0)

        checks = [
            tf.debugging.assert_equal(
                tf.shape(tensor)[1],
                expected_count,
                message=(
                    f"{operator_name} residual {i} has the wrong "
                    "number of stations."
                ),
            )
        ]
        if normalized:
            checks.append(
                tf.debugging.assert_equal(
                    tf.shape(tensor)[0],
                    tf.shape(normalized[0])[0],
                    message=(
                        f"{operator_name} residual {i} has an "
                        "inconsistent batch size."
                    ),
                )
            )
        with tf.control_dependencies([check for check in checks if check is not None]):
            normalized.append(tf.identity(tensor))

    # Residuals are expected to be block-aligned: one tensor per selected
    # variable, with a shared batch dimension and one observation per station.
    return normalized, tf.shape(normalized[0])[0]


def _validate_guidance_observations(
    y_obs,
    metadata,
    expected_batch_size: tf.Tensor,
) -> list[tf.Tensor]:
    """Validate guidance observations and return block-aligned rank-2 tensors."""
    normalized, observation_batch_size = _validate_and_normalize_residuals(
        y_obs,
        metadata.observation_counts,
        "guidance y_obs",
    )
    check = tf.debugging.assert_equal(
        observation_batch_size,
        expected_batch_size,
        message="Guidance observations must match the model-state batch size.",
    )
    with tf.control_dependencies([check] if check is not None else []):
        return [tf.identity(block) for block in normalized]


def _validate_state_shape(
    x: tf.Tensor,
    grid_shape: tuple[int, int],
    num_variables: int,
    operator_name: str,
) -> tf.Tensor:
    """Cast and validate a full state tensor before stencil gathering."""
    x = tf.cast(x, tf.float32)
    if x.shape.rank != 4:
        raise ValueError(
            f"{operator_name} H expects rank-4 state "
            "(batch, lat, lon, channels)."
        )

    expected = (*grid_shape, num_variables)
    for axis, (actual, wanted) in enumerate(zip(x.shape[1:], expected), start=1):
        if actual is not None and actual != wanted:
            raise ValueError(
                f"{operator_name} H state axis {axis} has size {actual}; "
                f"expected {wanted}."
            )

    check = tf.debugging.assert_equal(
        tf.shape(x)[1:],
        tf.constant(expected, dtype=tf.int32),
        message=f"{operator_name} H state shape does not match its configured grid.",
    )
    with tf.control_dependencies([check] if check is not None else []):
        return tf.identity(x)


def _effective_observation_weights(H: Callable, weights) -> list[tf.Tensor]:
    """Project per-channel weights into each observation's channel mixture."""
    weights_f32 = tf.cast(weights, tf.float32)
    if weights_f32.shape.rank != 1:
        raise ValueError(
            "Guidance weights must be a rank-1 vector with one value per channel."
        )
    metadata = getattr(H, "metadata", None)
    if metadata is not None:
        expected_count = metadata.num_variables
        actual_count = weights_f32.shape[0]
        if actual_count is not None and actual_count != expected_count:
            raise ValueError(
                f"Guidance weights contain {actual_count} channels; "
                f"expected {expected_count}."
            )
        check = tf.debugging.assert_equal(
            tf.shape(weights_f32)[0],
            expected_count,
            message="Guidance weights have the wrong number of channels.",
        )
        with tf.control_dependencies([check] if check is not None else []):
            weights_f32 = tf.identity(weights_f32)

    channel_indices = getattr(H, "observation_channel_indices", None)
    channel_weights = getattr(H, "observation_channel_weights", None)
    if channel_indices is None or channel_weights is None:
        # For a plain state-channel operator, the guidance weight is just the
        # one associated with the selected variable itself.
        return [weights_f32[var_idx] for var_idx in H.selected_vars]

    # For mixed-channel operators, a single guidance weight vector must be
    # projected through the observation-specific channel mixture before it can be
    # compared to the observation residuals.
    return [
        tf.reduce_sum(
            tf.gather(weights_f32, tf.constant(indices, dtype=tf.int32))
            * tf.constant(mixing_weights, dtype=tf.float32),
            axis=1,
        )
        for indices, mixing_weights in zip(channel_indices, channel_weights)
    ]


def _validate_vertical_layout(
    num_variables: int,
    num_levels: Optional[int],
    num_3d_fields: Optional[int],
    num_2d_fields: Optional[int],
) -> bool:
    """Validate optional field-layout metadata and report whether it is enabled."""
    layout_args = (num_levels, num_3d_fields, num_2d_fields)
    vertical_interp = all(value is not None for value in layout_args)
    if any(value is not None for value in layout_args) and not vertical_interp:
        raise ValueError(
            "num_levels, num_3d_fields, and num_2d_fields must be provided together."
        )
    if not vertical_interp:
        # Only 3D+2D field layouts that specify all three metadata values trigger
        # the vertical-interpolation code path.
        return False

    if num_levels <= 0:
        raise ValueError(f"num_levels must be positive; got {num_levels}.")
    if num_3d_fields < 0 or num_2d_fields < 0:
        raise ValueError("num_3d_fields and num_2d_fields must be non-negative.")

    expected_variables = num_3d_fields * num_levels + num_2d_fields
    if num_variables != expected_variables:
        raise ValueError(
            f"num_variables={num_variables} does not match the field layout "
            f"({num_3d_fields} * {num_levels} + {num_2d_fields} = "
            f"{expected_variables})."
        )
    return True


def _validate_operator_inputs(
    grid_shape: tuple[int, int],
    num_variables: int,
    variable_indices,
    station_groups,
    station_locations_per_var,
) -> Optional[np.ndarray]:
    """Validate inputs shared by all observation-operator factories."""
    if len(grid_shape) != 2 or any(int(size) != size or size <= 0 for size in grid_shape):
        raise ValueError(f"grid_shape must contain two positive integers; got {grid_shape}.")
    if int(num_variables) != num_variables or num_variables <= 0:
        raise ValueError(f"num_variables must be a positive integer; got {num_variables}.")

    if (station_groups is None) != (station_locations_per_var is None):
        raise ValueError(
            "station_groups and station_locations_per_var must be provided together."
        )

    # Shared-station operators and grouped-station operators share the same
    # validation path: the only difference is whether variables are mapped to a
    # common station set or to named station groups.
    if station_groups is not None:
        if not isinstance(station_groups, dict) or not station_groups:
            raise ValueError("station_groups must be a non-empty dictionary.")
        if not isinstance(station_locations_per_var, dict) or not station_locations_per_var:
            raise ValueError("station_locations_per_var must be a non-empty dictionary.")
        values = (
            list(station_locations_per_var.keys())
            if variable_indices is None
            else variable_indices
        )
    elif variable_indices is None:
        return None
    else:
        values = variable_indices

    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("At least one variable index must be supplied as a 1D sequence.")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise ValueError(f"Variable indices must be integers; got {values}.")

    selected = raw.astype(np.int32)
    invalid = (selected < 0) | (selected >= num_variables)
    if np.any(invalid):
        raise ValueError(
            f"Variable indices must lie in [0, {num_variables}); "
            f"got {selected[invalid].tolist()}."
        )
    if station_groups is not None:
        unmapped = [
            int(variable)
            for variable in selected
            if int(variable) not in station_locations_per_var
        ]
        if unmapped:
            raise ValueError(
                "Selected variable indices must have station-group mappings; "
                f"missing mappings for {unmapped}."
            )
    return selected


# ==== Data classes for compiled observation stencils ====
# The remaining objects in this file convert the validated operator geometry into
# a compact, block-structured stencil representation that can be reused by both
# the forward map H and its adjoint H_adjoint.

@dataclass(frozen=True)
class _StationAssignments:
    """Normalized station configuration used by compiled observation stencils.

    This object converts raw station input into a single internal model so all
    observation operators can reason about the same layout:

    - which state variables are selected,
    - which station groups exist,
    - which group each variable uses,
    - and the original source data used to build the configuration.

    The main purpose is to hide whether the caller supplied a single shared list
    of stations or a per-variable grouped configuration. Both cases are reduced
    to the same dictionary-based representation before any stencil or adjoint
    logic is built.
    """

    selected_variables: np.ndarray
    stations_by_group: _StationGroups
    group_by_variable: dict[int, str]
    source: _StationSource


def _resolve_station_assignments(
    num_variables: int,
    validated_variables: Optional[np.ndarray],
    station_locations: Optional[_StationLocations],
    station_groups: Optional[_StationGroups],
    station_locations_per_var: Optional[dict[int, str]],
) -> _StationAssignments:
    """Normalize station inputs into a single grouped assignment model.

    The observation operators in this module support two input styles:

    1. one shared list of station locations used for every selected variable, or
    2. multiple station groups, where each variable may reference a different set
       of locations.

    This function validates the user configuration, checks that all referenced
    groups exist, converts all locations to a consistent `(lat, lon, level)` form,
    and returns a `_StationAssignments` object. The returned structure is the
    canonical format consumed by later stencil-building code, making the layout
    independent of the original input form.
    """
    if station_groups is not None:
        if validated_variables is None:
            raise ValueError("Grouped stations must select at least one variable.")
        selected_variables = np.sort(validated_variables)
        group_by_variable = {
            int(variable): station_locations_per_var[int(variable)]
            for variable in selected_variables
        }
        used_groups = set(group_by_variable.values())
        unknown_groups = used_groups.difference(station_groups)
        if unknown_groups:
            raise ValueError(
                f"Unknown station groups {sorted(unknown_groups)}; "
                f"available groups are {list(station_groups)}."
            )

        stations_by_group = {}
        for group_name, group_stations in station_groups.items():
            if group_name not in used_groups:
                continue
            if not group_stations:
                raise ValueError(f"Station group '{group_name}' is empty.")
            stations_by_group[group_name] = _normalize_station_locations(group_stations)

        return _StationAssignments(
            selected_variables=selected_variables,
            stations_by_group=stations_by_group,
            group_by_variable=group_by_variable,
            source=station_groups,
        )

    if station_locations is None or len(station_locations) == 0:
        raise ValueError(
            "Either station_locations or grouped station configuration must be provided."
        )
    selected_variables = (
        np.arange(num_variables, dtype=np.int32)
        if validated_variables is None
        else validated_variables
    )
    shared_group = "__shared__"
    return _StationAssignments(
        selected_variables=selected_variables,
        stations_by_group={
            shared_group: _normalize_station_locations(station_locations)
        },
        group_by_variable={
            int(variable): shared_group for variable in selected_variables
        },
        source=station_locations,
    )


@dataclass(frozen=True)
class _ObservationBlock:
    """Compiled numerical stencil data for one selected state variable.

    This object stores the tensorized contribution pattern for a single model
    variable in the observation operator. Each entry represents one local stencil
    contribution at a given station and point within that station's footprint.

    The arrays are intentionally flattened and aligned so the shared executor can
    apply the same gather/scatter logic to all blocks. The `mixed_channel_*`
    arrays preserve any channel mixing implied by the operator (for example,
    vertical interpolation or profile-based weighting), which is later used to
    project guidance weights into observation space.
    """

    variable_index: int
    state_channel_indices: np.ndarray
    location_indices: np.ndarray
    weights: np.ndarray
    observation_count: int
    points_per_observation: int
    mixed_channel_indices: np.ndarray
    mixed_channel_weights: np.ndarray
    station_levels: np.ndarray

    def __post_init__(self) -> None:
        if self.observation_count <= 0 or self.points_per_observation <= 0:
            raise ValueError(
                "Compiled blocks require positive observation and point counts."
            )
        contribution_count = self.observation_count * self.points_per_observation
        contribution_shapes = {
            self.state_channel_indices.shape,
            self.location_indices.shape,
            self.weights.shape,
        }
        if contribution_shapes != {(contribution_count,)}:
            raise ValueError(
                "Compiled state channels, locations, and weights must each have "
                f"shape ({contribution_count},); got {contribution_shapes}."
            )
        if (
            self.mixed_channel_indices.ndim != 2
            or self.mixed_channel_indices.shape != self.mixed_channel_weights.shape
        ):
            raise ValueError(
                "Mixed channel indices and weights must be aligned rank-2 arrays."
            )
        if self.mixed_channel_indices.shape[0] != self.observation_count:
            raise ValueError(
                "Channel mixtures must contain one row per observation."
            )
        if self.station_levels.shape != (self.observation_count,):
            raise ValueError(
                "station_levels must contain one level per observation."
            )


@dataclass(frozen=True)
class _CompiledObservationStencil:
    """The complete compiled linear stencil for one observation operator.

    This container aggregates all per-variable observation blocks into a single
    reusable stencil object. The forward operator `H` and its adjoint
    `H_adjoint` consume the same compiled representation, so the operator is
    defined once in a block-split way and then executed through a shared
    gather/scatter implementation.

    The object also keeps the original station source so downstream code can
    still trace whether the operator was built from a shared station list or a
    variable-specific station grouping.
    """

    blocks: tuple[_ObservationBlock, ...]
    station_locations: _StationSource

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("A compiled observation stencil must contain a block.")

    @property
    def selected_variables(self) -> np.ndarray:
        return np.asarray(
            [block.variable_index for block in self.blocks], dtype=np.int32
        )

    @property
    def observation_counts(self) -> tuple[int, ...]:
        return tuple(block.observation_count for block in self.blocks)


@dataclass(frozen=True)
class _ObservationMetadata:
    """Stable block-aligned metadata for a forward/adjoint observation pair.

    This metadata object records the shape and alignment information needed to
    validate residual tensors and guidance weights without re-deriving the block
    structure from scratch. It keeps the selected variable set, block-wise
    observation counts, station-level information, and the channel-mixture
    structure used by the operator.

    The purpose is to ensure that the list of residuals passed to `H_adjoint`
    and the list of guidance weights passed to the likelihood term remain
    consistent with the compiled stencil layout.
    """

    num_variables: int
    selected_variables: np.ndarray
    observation_counts: tuple[int, ...]
    station_levels_by_block: tuple[np.ndarray, ...]
    channel_indices_by_block: tuple[np.ndarray, ...]
    channel_weights_by_block: tuple[np.ndarray, ...]
    station_source: _StationSource

    def __post_init__(self) -> None:
        if self.num_variables <= 0:
            raise ValueError("Observation metadata requires a positive channel count.")
        block_count = len(self.observation_counts)
        aligned_values = (
            self.selected_variables,
            self.station_levels_by_block,
            self.channel_indices_by_block,
            self.channel_weights_by_block,
        )
        if any(len(values) != block_count for values in aligned_values):
            raise ValueError("Observation metadata must align one entry per block.")

        for index, observation_count in enumerate(self.observation_counts):
            if self.station_levels_by_block[index].shape != (observation_count,):
                raise ValueError(
                    f"Station levels for block {index} do not match its observations."
                )
            if (
                self.channel_indices_by_block[index].shape
                != self.channel_weights_by_block[index].shape
            ):
                raise ValueError(
                    f"Channel indices and weights for block {index} are not aligned."
                )
            if self.channel_indices_by_block[index].shape[0] != observation_count:
                raise ValueError(
                    f"Channel mixtures for block {index} do not match its observations."
                )


#==== Compiled forward/adjoint executor ====

def _create_compiled_linear_operator(
    grid_shape: tuple[int, int],
    num_variables: int,
    stencil: _CompiledObservationStencil,
    operator_name: str,
) -> tuple[Callable, Callable]:
    """Build the shared forward and adjoint operators for a compiled stencil.

    The compiled stencil contains all required station, state-channel, and weight
    metadata. This function turns that abstract representation into the actual
    TensorFlow operator pair used in the SDA likelihood:

    - `H(x)` gathers the relevant state values, multiplies by stencil weights,
      and returns one observation vector per selected variable block.
    - `H_adjoint(residual)` expands the residual back into full-grid state space
      by scattering each weighted contribution to the matching location and
      channel index.

    The implementation is intentionally block-structured and vectorized so the
    same logic works for all operator families after their geometry is compiled
    into a common stencil format.
    """
    lat_size, lon_size = grid_shape
    blocks = stencil.blocks
    observation_counts = stencil.observation_counts
    contribution_counts = tuple(
        block.observation_count * block.points_per_observation for block in blocks
    )

    state_channel_indices = np.concatenate(
        [block.state_channel_indices for block in blocks]
    ).astype(np.int32)
    location_indices = np.concatenate(
        [block.location_indices for block in blocks]
    ).astype(np.int32)
    weights = np.concatenate([block.weights for block in blocks]).astype(np.float32)
    total_contributions = int(weights.size)

    state_channel_indices_tf = tf.constant(state_channel_indices, dtype=tf.int32)
    location_indices_tf = tf.constant(location_indices, dtype=tf.int32)
    weights_tf = tf.constant(weights, dtype=tf.float32)
    block_weights_tf = tuple(
        tf.constant(
            block.weights.reshape(
                block.observation_count, block.points_per_observation
            ),
            dtype=tf.float32,
        )
        for block in blocks
    )

    def H(x: tf.Tensor) -> list[tf.Tensor]:
        """Gather and reduce compiled stencil contributions."""
        x = _validate_state_shape(x, grid_shape, num_variables, operator_name)
        batch_size = tf.shape(x)[0]
        # Flatten the state into [batch, grid_cell, channel] so the stencil can
        # be applied by a single gather-and-weight step per contribution.
        x_flat = tf.transpose(
            tf.reshape(x, [batch_size, lat_size * lon_size, num_variables]),
            perm=[0, 2, 1],
        )

        gather_indices = tf.stack(
            [
                tf.repeat(tf.range(batch_size, dtype=tf.int32), total_contributions),
                tf.tile(state_channel_indices_tf, [batch_size]),
                tf.tile(location_indices_tf, [batch_size]),
            ],
            axis=1,
        )
        gathered = tf.gather_nd(x_flat, gather_indices)
        weighted = gathered * tf.tile(weights_tf, [batch_size])
        weighted = tf.reshape(weighted, [batch_size, total_contributions])

        observations = []
        for block, values in zip(blocks, tf.split(weighted, contribution_counts, axis=1)):
            values = tf.reshape(
                values,
                [batch_size, block.observation_count, block.points_per_observation],
            )
            observations.append(tf.reduce_sum(values, axis=2))
        return observations

    def H_adjoint(residual) -> tf.Tensor:
        """Expand and scatter residuals using the compiled forward weights."""
        residual, batch_size = _validate_and_normalize_residuals(
            residual, observation_counts, f"{operator_name} H_adjoint"
        )

        updates_by_batch = []
        for block_residual, block_weights in zip(residual, block_weights_tf):
            # Each residual contribution is weighted by the same stencil weights
            # used in the forward pass, then flattened so it can be scattered back
            # to the state tensor at the matching (location, channel) index.
            weighted = block_residual[:, :, tf.newaxis] * block_weights[tf.newaxis, :, :]
            updates_by_batch.append(tf.reshape(weighted, [batch_size, -1]))
        updates = tf.reshape(tf.concat(updates_by_batch, axis=1), [-1])

        scatter_indices = tf.stack(
            [
                tf.repeat(tf.range(batch_size, dtype=tf.int32), total_contributions),
                tf.tile(location_indices_tf, [batch_size]),
                tf.tile(state_channel_indices_tf, [batch_size]),
            ],
            axis=1,
        )
        full_state = tf.scatter_nd(
            scatter_indices,
            updates,
            [batch_size, lat_size * lon_size, num_variables],
        )
        ones_state = tf.scatter_nd(
            scatter_indices,
            tf.ones_like(updates),
            [batch_size, lat_size * lon_size, num_variables],
        )
        full_state = tf.where(ones_state == 0, full_state, full_state / ones_state)
        return tf.reshape(full_state, [batch_size, lat_size, lon_size, num_variables])

    metadata = _ObservationMetadata(
        num_variables=num_variables,
        selected_variables=stencil.selected_variables,
        observation_counts=observation_counts,
        station_levels_by_block=tuple(block.station_levels for block in blocks),
        channel_indices_by_block=tuple(
            block.mixed_channel_indices for block in blocks
        ),
        channel_weights_by_block=tuple(
            block.mixed_channel_weights for block in blocks
        ),
        station_source=stencil.station_locations,
    )

    # Preserve station_levels as a compatibility view. New code should use the
    # consistently block-aligned metadata.station_levels_by_block representation.
    if isinstance(stencil.station_locations, dict):
        legacy_station_levels = np.concatenate(metadata.station_levels_by_block)
    else:
        legacy_station_levels = metadata.station_levels_by_block[0]

    for operator in (H, H_adjoint):
        operator.metadata = metadata
        operator.selected_vars = metadata.selected_variables
        operator.observation_channel_indices = list(
            metadata.channel_indices_by_block
        )
        operator.observation_channel_weights = list(
            metadata.channel_weights_by_block
        )
        operator.observation_station_levels = metadata.station_levels_by_block
        operator.station_levels = legacy_station_levels
        operator.station_locations = metadata.station_source
        operator.compiled_stencil = stencil
        operator.grid_shape = grid_shape
        operator.num_variables = num_variables

    return H, H_adjoint


def _validate_vertical_weight_profiles(
    profiles,
    assignments: _StationAssignments,
    num_levels: Optional[int],
    num_3d_fields: Optional[int],
) -> Optional[_VerticalWeightProfiles]:
    """Validate and broadcast optional full-column vertical profiles.

    Some atmospheric observations use a full vertical weighting profile instead
    of the default linear interpolation between adjacent model levels. This
    function ensures those custom profiles are valid, align with the selected
    variables and station groups, and are broadcast into the same shape expected
    by the compiled stencil.

    The output is a per-variable dictionary keyed by variable index, where each
    value is a station-by-level matrix of finite, nonzero vertical weights.
    """
    if profiles is None:
        return None
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("vertical_weight_profiles must be a non-empty dictionary.")
    if num_levels is None or num_3d_fields is None:
        raise ValueError(
            "vertical_weight_profiles require complete vertical field-layout metadata."
        )

    selected = {int(index) for index in assignments.selected_variables}
    normalized = {}
    for variable_index, values in profiles.items():
        if (
            isinstance(variable_index, (bool, np.bool_))
            or not isinstance(variable_index, (int, np.integer))
        ):
            raise ValueError("Vertical weight profile keys must be integer variable indices.")
        variable_index = int(variable_index)
        if variable_index not in selected:
            raise ValueError(
                f"Vertical weight profile variable {variable_index} is not selected."
            )
        if variable_index < 0 or variable_index >= num_3d_fields * num_levels:
            raise ValueError(
                "Vertical weight profiles can only be assigned to atmospheric "
                f"3D variables; got variable {variable_index}."
            )

        group_name = assignments.group_by_variable[variable_index]
        observation_count = len(assignments.stations_by_group[group_name])
        profile = np.asarray(values, dtype=np.float32)
        if profile.ndim == 1:
            if profile.shape != (num_levels,):
                raise ValueError(
                    f"Vertical weight profile for variable {variable_index} must "
                    f"contain {num_levels} levels; got shape {profile.shape}."
                )
            profile = np.broadcast_to(
                profile[np.newaxis, :], (observation_count, num_levels)
            ).copy()
        elif profile.shape != (observation_count, num_levels):
            raise ValueError(
                f"Vertical weight profiles for variable {variable_index} must have "
                f"shape ({observation_count}, {num_levels}); got {profile.shape}."
            )
        if not np.all(np.isfinite(profile)):
            raise ValueError(
                f"Vertical weight profiles for variable {variable_index} must be finite."
            )
        if np.any(np.all(profile == 0.0, axis=1)):
            raise ValueError(
                f"Vertical weight profiles for variable {variable_index} must have "
                "at least one nonzero weight per observation."
            )
        normalized[variable_index] = profile

    return normalized


def _compose_horizontal_vertical_stencil(
    var_idx: int,
    station_locations: _StationLocations,
    horizontal_locs: np.ndarray,
    horizontal_weights: np.ndarray,
    num_levels: Optional[int] = None,
    num_3d_fields: Optional[int] = None,
    vertical_weight_profile: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Combine horizontal geometry with vertical weights into one stencil.

    This is the function that turns a local horizontal stencil into the actual
    full observation stencil for a variable. It combines each horizontal point
    with its corresponding vertical contribution, producing the flattened
    `state_channel_indices`, `location_indices`, and `weights` used by the
    compiled operator.

    For 2D variables, the result is a simple horizontal stencil. For 3D
    atmospheric variables, the function builds a tensor product of the
    horizontal stencil and the vertical interpolation/profile weights, so each
    observation uses all relevant horizontal and vertical neighbors.
    """
    num_stations = len(station_locations)
    if horizontal_locs.shape != horizontal_weights.shape:
        raise ValueError("Horizontal stencil locations and weights must have equal shapes.")
    if horizontal_locs.ndim != 2 or horizontal_locs.shape[0] != num_stations:
        raise ValueError(
            "Horizontal stencil must have shape (number of stations, points per station)."
        )

    is_3d_var = (
        num_levels is not None
        and num_3d_fields is not None
        and var_idx < num_3d_fields * num_levels
    )
    if not is_3d_var:
        points_per_station = horizontal_weights.shape[1]
        channel_indices = np.full((num_stations, 1), var_idx, dtype=np.int32)
        channel_weights = np.ones((num_stations, 1), dtype=np.float32)
        return (
            np.full(num_stations * points_per_station, var_idx, dtype=np.int32),
            horizontal_locs.reshape(-1).astype(np.int32),
            horizontal_weights.reshape(-1).astype(np.float32),
            points_per_station,
            channel_indices,
            channel_weights,
        )

    level_values = np.asarray(
        [level for _, _, level in station_locations], dtype=np.float32
    )
    invalid = (level_values < 0.0) | (level_values > float(num_levels - 1))
    if np.any(invalid):
        invalid_values = np.unique(level_values[invalid]).tolist()
        raise ValueError(
            "Atmospheric station levels must lie within the model-level range "
            f"[0, {num_levels - 1}]; got {invalid_values}. The pseudo-level is "
            "valid only for 2D variables."
        )

    field_idx = int(var_idx // num_levels)
    if vertical_weight_profile is None:
        # Default vertical interpolation is deliberately linear and independent
        # of the horizontal stencil family or polynomial order.
        vertical_nodes, vertical_weights = _lagrange_weights_1d(
            level_values, num_levels, order=1
        )
        channel_indices = (field_idx * num_levels + vertical_nodes).astype(np.int32)
    else:
        if vertical_weight_profile.shape != (num_stations, num_levels):
            raise ValueError(
                "Compiled vertical weight profiles must align with stations and levels."
            )
        vertical_weights = vertical_weight_profile
        channel_indices = np.broadcast_to(
            field_idx * num_levels + np.arange(num_levels, dtype=np.int32),
            (num_stations, num_levels),
        ).copy()

    horizontal_points = horizontal_weights.shape[1]
    vertical_points = vertical_weights.shape[1]
    # A 3D observation stencil is the tensor product of the horizontal stencil
    # and the vertical weighting profile. The flattening below turns that
    # product into the same linear form consumed by the generic gather/scatter
    # executor.
    shape = (num_stations, horizontal_points, vertical_points)
    var_ids = np.broadcast_to(
        channel_indices[:, np.newaxis, :], shape
    ).reshape(-1)
    loc_ids = np.broadcast_to(
        horizontal_locs[:, :, np.newaxis], shape
    ).reshape(-1)
    combined_weights = (
        horizontal_weights[:, :, np.newaxis]
        * vertical_weights[:, np.newaxis, :]
    ).reshape(-1)

    return (
        var_ids.astype(np.int32),
        loc_ids.astype(np.int32),
        combined_weights.astype(np.float32),
        horizontal_points * vertical_points,
        channel_indices,
        vertical_weights.astype(np.float32),
    )


def _compile_horizontal_stencils(
    assignments: _StationAssignments,
    grid_shape: tuple[int, int],
    stencil_builder: _HorizontalStencilBuilder,
) -> dict[str, _HorizontalStencil]:
    """Precompute one reusable horizontal stencil per station group.

    Many variables may share the same station group and therefore the same
    geometric neighborhood structure. This function builds that geometry once per
    group and caches it, avoiding redundant work while keeping the compiled
    stencil consistent across variables.
    """
    return {
        group_name: stencil_builder(group_stations, grid_shape)
        for group_name, group_stations in assignments.stations_by_group.items()
    }


def _compile_observation_stencil(
    assignments: _StationAssignments,
    horizontal_stencils: dict[str, _HorizontalStencil],
    num_levels: Optional[int] = None,
    num_3d_fields: Optional[int] = None,
    vertical_weight_profiles: Optional[_VerticalWeightProfiles] = None,
) -> _CompiledObservationStencil:
    """Assemble the final compiled observation stencil for all selected variables.

    This function loops over every selected variable, identifies the station
    group it belongs to, combines its geometric stencil with the correct
    vertical weights, and converts the result into a `_ObservationBlock`.

    The output is a full `_CompiledObservationStencil`, which is the canonical
    representation consumed by the shared `H`/`H_adjoint` executor.
    """
    blocks = []
    for variable_index in assignments.selected_variables:
        group_name = assignments.group_by_variable[int(variable_index)]
        group_stations = assignments.stations_by_group[group_name]
        horizontal_locations, horizontal_weights = horizontal_stencils[group_name]
        (
            state_channels,
            locations,
            weights,
            points_per_observation,
            mixed_channels,
            mixed_weights,
        ) = _compose_horizontal_vertical_stencil(
            int(variable_index),
            group_stations,
            horizontal_locations,
            horizontal_weights,
            num_levels=num_levels,
            num_3d_fields=num_3d_fields,
            vertical_weight_profile=(
                vertical_weight_profiles.get(int(variable_index))
                if vertical_weight_profiles is not None
                else None
            ),
        )
        blocks.append(
            _ObservationBlock(
                variable_index=int(variable_index),
                state_channel_indices=state_channels,
                location_indices=locations,
                weights=weights,
                observation_count=len(group_stations),
                points_per_observation=points_per_observation,
                mixed_channel_indices=mixed_channels,
                mixed_channel_weights=mixed_weights,
                station_levels=np.asarray(
                    [level for _, _, level in group_stations], dtype=np.float32
                ),
            )
        )

    return _CompiledObservationStencil(
        blocks=tuple(blocks),
        station_locations=assignments.source,
    )


def _create_stencil_observation_operator(
    grid_shape: tuple[int, int],
    num_variables: int,
    variable_indices: Optional[list[int]],
    station_locations: Optional[_StationLocations],
    station_groups: Optional[_StationGroups],
    station_locations_per_var: Optional[dict[int, str]],
    horizontal_stencil_builder: _HorizontalStencilBuilder,
    operator_name: str,
    num_levels: Optional[int],
    num_3d_fields: Optional[int],
    num_2d_fields: Optional[int],
    vertical_weight_profiles=None,
) -> tuple[Callable, Callable]:
    """Run the full stencil-construction pipeline for a fixed linear operator.

    This is the top-level factory used by concrete observation operators such as
    index or interpolation operators. It validates the user configuration,
    normalizes station group assignments, validates any optional vertical
    weighting profiles, builds the geometric stencils, and converts the final
    data into the shared forward/adjoint operator pair returned by
    `_create_compiled_linear_operator`.
    """
    validated_variables = _validate_operator_inputs(
        grid_shape,
        num_variables,
        variable_indices,
        station_groups,
        station_locations_per_var,
    )
    vertical_interpolation = _validate_vertical_layout(
        num_variables, num_levels, num_3d_fields, num_2d_fields
    )
    assignments = _resolve_station_assignments(
        num_variables,
        validated_variables,
        station_locations,
        station_groups,
        station_locations_per_var,
    )
    validated_vertical_profiles = _validate_vertical_weight_profiles(
        vertical_weight_profiles,
        assignments,
        num_levels if vertical_interpolation else None,
        num_3d_fields if vertical_interpolation else None,
    )
    horizontal_stencils = _compile_horizontal_stencils(
        assignments, grid_shape, horizontal_stencil_builder
    )
    stencil = _compile_observation_stencil(
        assignments,
        horizontal_stencils,
        num_levels=num_levels if vertical_interpolation else None,
        num_3d_fields=num_3d_fields if vertical_interpolation else None,
        vertical_weight_profiles=validated_vertical_profiles,
    )
    return _create_compiled_linear_operator(
        grid_shape, num_variables, stencil, operator_name=operator_name
    )

# ==== Interpolation obs operators ====

def _lagrange_weights_1d(
    f: np.ndarray,
    size: int,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
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
    # The station location is expressed in fractional grid coordinates, so we
    # first identify the enclosing lower-index node and the local offset within
    # that cell. The weights are then built from the Lagrange basis evaluated at
    # that offset. This yields a local interpolation stencil that is exact for the
    # chosen polynomial order.
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
) -> _HorizontalStencil:
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

    # A 2D interpolation stencil is the tensor product of the 1D lat and lon
    # stencils. Each station gets a local neighborhood of size K whose weights
    # factorize into the product of the two 1D Lagrange basis weights.
    lat_n  = lat_nodes[:, :, np.newaxis]  # (N, S, 1)
    lon_n  = lon_nodes[:, np.newaxis, :]  # (N, 1, S)
    lat_ww = lat_w[:, :, np.newaxis]      # (N, S, 1)
    lon_ww = lon_w[:, np.newaxis, :]      # (N, 1, S)

    neigh_locs = (lat_n * lon_size + lon_n).reshape(N, K).astype(np.int32)
    neigh_w    = (lat_ww * lon_ww).reshape(N, K).astype(np.float32)

    return neigh_locs, neigh_w


def create_observation_operator_interp(
    grid_shape: tuple[int, int],
    num_variables: int,
    variable_indices: Optional[list[int]] = None,
    station_locations: Optional[_StationLocations] = None,
    station_groups: Optional[_StationGroups] = None,
    station_locations_per_var: Optional[dict[int, str]] = None,
    order: int = 1,
    num_levels: Optional[int] = None,
    num_3d_fields: Optional[int] = None,
    num_2d_fields: Optional[int] = None,
    vertical_weight_profiles=None,
) -> tuple[Callable, Callable]:
    """Create a Lagrange-interpolating observation operator and its adjoint.

    Every station must be a ``(lat, lon, level)`` triplet in fractional grid
    coordinates. ``station_locations`` supplies one shared station set for the
    selected variables. Alternatively, ``station_groups`` and
    ``station_locations_per_var`` assign a named station set to each variable.

    Horizontal interpolation uses a tensor-product Lagrange stencil of order
    1, 2, or 3 (respectively 2x2, 3x3, or 4x4 points). Horizontal coordinates
    are clipped to the grid, and stencil nodes beyond an edge are clamped to
    boundary cells; repeated boundary contributions retain their weights.

    When all field-layout arguments are supplied, 3D channels are interpolated
    linearly between adjacent model levels. Levels outside the 3D range are
    rejected. Appended 2D channels use the same horizontal stencil without
    vertical mixing; their station level is metadata only.

    ``vertical_weight_profiles`` may map a selected atmospheric variable index
    to either one ``(num_levels,)`` profile shared by its observations or an
    ``(num_observations, num_levels)`` array of observation-specific profiles.
    Profiles are applied as supplied and are not automatically normalized.

    The horizontal and vertical weights are compiled once and delegated to the
    shared linear-stencil executor used by both the forward and adjoint paths.

    Returns:
        ``(H, H_adjoint)``. ``H(x)`` returns one observation tensor per selected
        variable, and ``H_adjoint(residual)`` applies the exact transpose of the
        same compiled stencil to a matching residual sequence.
    """
    if order not in (1, 2, 3):
        raise ValueError(f"order must be 1, 2, or 3; got {order}.")

    def build_horizontal_stencil(group_stations, builder_grid_shape):
        lat_size, lon_size = builder_grid_shape
        coordinates = np.asarray(
            [(lat, lon) for lat, lon, _ in group_stations], dtype=np.float32
        )
        lat_f = np.clip(coordinates[:, 0], 0.0, float(lat_size - 1))
        lon_f = np.clip(coordinates[:, 1], 0.0, float(lon_size - 1))
        return _compute_interp_stencil(lat_f, lon_f, lat_size, lon_size, order)

    return _create_stencil_observation_operator(
        grid_shape=grid_shape,
        num_variables=num_variables,
        variable_indices=variable_indices,
        station_locations=station_locations,
        station_groups=station_groups,
        station_locations_per_var=station_locations_per_var,
        horizontal_stencil_builder=build_horizontal_stencil,
        operator_name="interp",
        num_levels=num_levels,
        num_3d_fields=num_3d_fields,
        num_2d_fields=num_2d_fields,
        vertical_weight_profiles=vertical_weight_profiles,
    )

# ==== Neighborhood obs operators ====

def _compute_neighborhood_stencil(
    station_locations: _StationLocations,
    grid_shape: tuple[int, int],
    dlat: np.ndarray,
    dlon: np.ndarray,
    base_weights: np.ndarray,
) -> _HorizontalStencil:
    """Build a clipped horizontal neighborhood stencil for normalized stations."""
    lat_size, lon_size = grid_shape
    coordinates = np.asarray(
        [(lat, lon) for lat, lon, _ in station_locations], dtype=np.float32
    )
    # Neighborhood operators work on integer grid centers: round the station's
    # fractional position to the nearest grid cell, then offset it by each local
    # neighborhood displacement. Out-of-range offsets are clipped to the boundary,
    # which creates the replicate-boundary behavior used throughout the file.
    center_lat = np.clip(np.rint(coordinates[:, 0]), 0, lat_size - 1).astype(np.int32)
    center_lon = np.clip(np.rint(coordinates[:, 1]), 0, lon_size - 1).astype(np.int32)

    neighbor_lat = np.clip(
        center_lat[:, np.newaxis] + dlat[np.newaxis, :], 0, lat_size - 1
    )
    neighbor_lon = np.clip(
        center_lon[:, np.newaxis] + dlon[np.newaxis, :], 0, lon_size - 1
    )
    locations = (neighbor_lat * lon_size + neighbor_lon).astype(np.int32)
    weights = np.broadcast_to(base_weights, locations.shape).astype(np.float32).copy()
    return locations, weights


def create_observation_operator_neighborhood(
    grid_shape: tuple[int, int],
    num_variables: int,
    variable_indices: Optional[list[int]] = None,
    station_locations: Optional[_StationLocations] = None,
    station_groups: Optional[_StationGroups] = None,
    station_locations_per_var: Optional[dict[int, str]] = None,
    window_size: int = 3,
    kernel: str = "uniform",
    sigma: float = 1.0,
    num_levels: Optional[int] = None,
    num_3d_fields: Optional[int] = None,
    num_2d_fields: Optional[int] = None,
    vertical_weight_profiles=None,
) -> tuple[Callable, Callable]:
    """Create a neighborhood-averaging observation operator and its adjoint.

    Every station must be a ``(lat, lon, level)`` triplet in fractional grid
    coordinates. ``station_locations`` supplies one shared station set for the
    selected variables. Alternatively, ``station_groups`` and
    ``station_locations_per_var`` assign a named station set to each variable.

    Each horizontal center is rounded to the nearest grid index and clipped to
    the grid. An odd ``window_size`` defines a square neighborhood with either
    normalized uniform weights or normalized Gaussian weights using ``sigma``.
    Window nodes beyond an edge are clipped to boundary cells, so their repeated
    contributions implement replicate-boundary behavior.

    When all field-layout arguments are supplied, 3D channels are interpolated
    linearly between adjacent model levels. Levels outside the 3D range are
    rejected. Appended 2D channels are averaged horizontally without vertical
    mixing; their station level is metadata only.

    The horizontal and vertical weights are compiled once and delegated to the
    shared linear-stencil executor used by both the forward and adjoint paths.

    Returns:
        ``(H, H_adjoint)``. ``H(x)`` returns one observation tensor per selected
        variable, and ``H_adjoint(residual)`` applies the exact transpose of the
        same compiled stencil to a matching residual sequence.
    """
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(f"window_size must be odd and >= 1; got {window_size}.")

    kernel = str(kernel).lower()
    if kernel not in ("uniform", "gaussian"):
        raise ValueError(f"kernel must be 'uniform' or 'gaussian'; got {kernel}.")

    if kernel == "gaussian" and sigma <= 0:
        raise ValueError(f"sigma must be > 0 for gaussian kernel; got {sigma}.")

    radius = window_size // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.int32)
    dlat_grid, dlon_grid = np.meshgrid(offsets, offsets, indexing="ij")
    dlat = dlat_grid.reshape(-1)
    dlon = dlon_grid.reshape(-1)
    neighbors_per_station = int(dlat.shape[0])

    if kernel == "uniform":
        # A uniform neighborhood is just a constant kernel normalized to sum to 1.
        base_w = np.ones(neighbors_per_station, dtype=np.float32)
    else:
        # A Gaussian neighborhood gives more weight to points close to the center.
        dist2 = dlat.astype(np.float32) ** 2 + dlon.astype(np.float32) ** 2
        base_w = np.exp(-0.5 * dist2 / (float(sigma) ** 2)).astype(np.float32)
    base_w = base_w / np.sum(base_w)

    def build_horizontal_stencil(group_stations, builder_grid_shape):
        return _compute_neighborhood_stencil(
            group_stations, builder_grid_shape, dlat, dlon, base_w
        )

    return _create_stencil_observation_operator(
        grid_shape=grid_shape,
        num_variables=num_variables,
        variable_indices=variable_indices,
        station_locations=station_locations,
        station_groups=station_groups,
        station_locations_per_var=station_locations_per_var,
        horizontal_stencil_builder=build_horizontal_stencil,
        operator_name="neighborhood",
        num_levels=num_levels,
        num_3d_fields=num_3d_fields,
        num_2d_fields=num_2d_fields,
        vertical_weight_profiles=vertical_weight_profiles,
    )


def create_observation_operator_footprint(
    grid_shape: tuple[int, int],
    num_variables: int,
    variable_indices: Optional[list[int]] = None,
    station_locations: Optional[_StationLocations] = None,
    station_groups: Optional[_StationGroups] = None,
    station_locations_per_var: Optional[dict[int, str]] = None,
    radius: float = 1.0,
    num_levels: Optional[int] = None,
    num_3d_fields: Optional[int] = None,
    num_2d_fields: Optional[int] = None,
    vertical_weight_profiles=None,
) -> tuple[Callable, Callable]:
    """Create a uniform circular-footprint operator and its exact adjoint.

    ``radius`` is measured in grid cells around the nearest grid point. Integer
    offsets whose Euclidean distance does not exceed the radius receive equal
    normalized weights. At grid boundaries, out-of-domain points are clipped
    to boundary cells, matching the neighborhood operator's replicate-boundary
    behavior. Vertical interpolation and grouped station assignment use the
    shared fixed-stencil pipeline.
    """
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"radius must be finite and > 0; got {radius}.")

    # A circular footprint is just the set of integer offsets whose Euclidean
    # distance to the center is within the requested radius. The resulting kernel
    # is uniform across all included cells, which makes the footprint operator a
    # local area average over a disk.
    offset_limit = int(np.ceil(radius))
    offsets = np.arange(-offset_limit, offset_limit + 1, dtype=np.int32)
    dlat_grid, dlon_grid = np.meshgrid(offsets, offsets, indexing="ij")
    inside = dlat_grid.astype(np.float64) ** 2 + dlon_grid.astype(np.float64) ** 2
    inside = inside <= float(radius) ** 2
    dlat = dlat_grid[inside].reshape(-1)
    dlon = dlon_grid[inside].reshape(-1)
    base_weights = np.full(dlat.size, 1.0 / dlat.size, dtype=np.float32)

    def build_horizontal_stencil(group_stations, builder_grid_shape):
        return _compute_neighborhood_stencil(
            group_stations, builder_grid_shape, dlat, dlon, base_weights
        )

    return _create_stencil_observation_operator(
        grid_shape=grid_shape,
        num_variables=num_variables,
        variable_indices=variable_indices,
        station_locations=station_locations,
        station_groups=station_groups,
        station_locations_per_var=station_locations_per_var,
        horizontal_stencil_builder=build_horizontal_stencil,
        operator_name="footprint",
        num_levels=num_levels,
        num_3d_fields=num_3d_fields,
        num_2d_fields=num_2d_fields,
        vertical_weight_profiles=vertical_weight_profiles,
    )


def create_observation_operator_finite_difference(
    grid_shape: tuple[int, int],
    num_variables: int,
    variable_indices: Optional[list[int]] = None,
    station_locations: Optional[_StationLocations] = None,
    station_groups: Optional[_StationGroups] = None,
    station_locations_per_var: Optional[dict[int, str]] = None,
    difference: str = "lat",
    grid_spacing=1.0,
    num_levels: Optional[int] = None,
    num_3d_fields: Optional[int] = None,
    num_2d_fields: Optional[int] = None,
    vertical_weight_profiles=None,
) -> tuple[Callable, Callable]:
    """Create a centered first derivative or five-point Laplacian operator.

    ``difference`` is ``"lat"``, ``"lon"``, or ``"laplacian"``.
    ``grid_spacing`` is either one positive scalar or ``(lat, lon)``. Stencils
    are centered on the nearest grid point. At boundaries, offsets are clipped
    to boundary cells; repeated contributions preserve zero response to a
    constant field and the compiled scatter remains the exact adjoint.
    """
    difference = str(difference).lower()
    if difference not in ("lat", "lon", "laplacian"):
        raise ValueError(
            "difference must be 'lat', 'lon', or 'laplacian'; "
            f"got {difference}."
        )

    spacing = np.asarray(grid_spacing, dtype=np.float64)
    if spacing.ndim == 0:
        lat_spacing = lon_spacing = float(spacing)
    elif spacing.shape == (2,):
        lat_spacing, lon_spacing = map(float, spacing)
    else:
        raise ValueError("grid_spacing must be a scalar or a (lat, lon) pair.")
    if (
        not np.isfinite(lat_spacing)
        or not np.isfinite(lon_spacing)
        or lat_spacing <= 0
        or lon_spacing <= 0
    ):
        raise ValueError("grid_spacing values must be finite and positive.")

    if difference == "lat":
        # Central difference in latitude: f(lat + 1) - f(lat - 1) over 2*dy.
        dlat = np.array([-1, 1], dtype=np.int32)
        dlon = np.array([0, 0], dtype=np.int32)
        base_weights = np.array(
            [-0.5 / lat_spacing, 0.5 / lat_spacing], dtype=np.float32
        )
    elif difference == "lon":
        # Central difference in longitude: f(lon + 1) - f(lon - 1) over 2*dx.
        dlat = np.array([0, 0], dtype=np.int32)
        dlon = np.array([-1, 1], dtype=np.int32)
        base_weights = np.array(
            [-0.5 / lon_spacing, 0.5 / lon_spacing], dtype=np.float32
        )
    else:
        # Five-point Laplacian: second derivative in both directions with a
        # standard second-order stencil centered at the target cell.
        inv_lat2 = 1.0 / lat_spacing**2
        inv_lon2 = 1.0 / lon_spacing**2
        dlat = np.array([0, -1, 1, 0, 0], dtype=np.int32)
        dlon = np.array([0, 0, 0, -1, 1], dtype=np.int32)
        base_weights = np.array(
            [
                -2.0 * inv_lat2 - 2.0 * inv_lon2,
                inv_lat2,
                inv_lat2,
                inv_lon2,
                inv_lon2,
            ],
            dtype=np.float32,
        )

    def build_horizontal_stencil(group_stations, builder_grid_shape):
        return _compute_neighborhood_stencil(
            group_stations, builder_grid_shape, dlat, dlon, base_weights
        )

    return _create_stencil_observation_operator(
        grid_shape=grid_shape,
        num_variables=num_variables,
        variable_indices=variable_indices,
        station_locations=station_locations,
        station_groups=station_groups,
        station_locations_per_var=station_locations_per_var,
        horizontal_stencil_builder=build_horizontal_stencil,
        operator_name=f"finite difference ({difference})",
        num_levels=num_levels,
        num_3d_fields=num_3d_fields,
        num_2d_fields=num_2d_fields,
        vertical_weight_profiles=vertical_weight_profiles,
    )


def create_observation_operator_index(
    grid_shape: tuple[int, int],
    num_variables: int,
    variable_indices: Optional[list[int]] = None,
    station_locations: Optional[_StationLocations] = None,
    station_groups: Optional[_StationGroups] = None,
    station_locations_per_var: Optional[dict[int, str]] = None,
    num_levels: Optional[int] = None,
    num_3d_fields: Optional[int] = None,
    num_2d_fields: Optional[int] = None,
    vertical_weight_profiles=None,
) -> tuple[Callable, Callable]:
    """Create exact-point observations using a one-point neighborhood stencil.

    This wrapper delegates to ``create_observation_operator_neighborhood`` with
    ``window_size=1`` and a uniform kernel. All station selection, vertical
    interpolation, adjoint, validation, and metadata behavior is shared with
    that operator.

    Station coordinates are ``(lat, lon, level)`` triplets. The optional layout
    arguments must be supplied together to enable vertical interpolation.

    Returns:
        The shared ``(H, H_adjoint)`` operator pair.
    """

    # Exact index sampling is the one-point special case of a neighborhood
    # stencil. The neighborhood implementation therefore supplies the complete
    # forward, adjoint, vertical interpolation, validation, and metadata path.
    # This is just a convenience wrapper so callers can request exact points
    # without constructing a custom stencil themselves.
    return create_observation_operator_neighborhood(
        grid_shape=grid_shape,
        num_variables=num_variables,
        variable_indices=variable_indices,
        station_locations=station_locations,
        station_groups=station_groups,
        station_locations_per_var=station_locations_per_var,
        window_size=1,
        kernel="uniform",
        num_levels=num_levels,
        num_3d_fields=num_3d_fields,
        num_2d_fields=num_2d_fields,
        vertical_weight_profiles=vertical_weight_profiles,
    )

# ==== Linear combination of obs operators ====

def create_observation_operator_linear_combination(
    base_operator: Callable,
    coefficients,
) -> tuple[Callable, Callable]:
    """Create a new observation operator as a weighted linear combination.

    This helper takes a compiled observation operator whose internal stencil has
    multiple per-variable blocks and produces a single new operator
    representing:

        H_combo(x) = sum_k c_k * H_k(x)

    where each ``c_k`` is the scalar coefficient associated with one base block.
    The returned operator keeps the same SDA contract: forward evaluation and an
    exact adjoint via the shared compiled linear executor.

    Important requirements:

    - The base operator must be built from a single shared station list, not a
      grouped per-variable station configuration.
    - All blocks must be colocated at the same station locations and must have
      the same observation count and station-level alignment.
    - Different blocks may have different horizontal or vertical stencils; their
      contributions are simply scaled and concatenated into one merged stencil.

    This is useful when a single observation term is defined as a weighted sum of
    several colocated observation sources, such as a combined scalar product or
    a blended multi-variable measurement. Because the merged result is compiled
    back into a valid `_ObservationBlock`, the adjoint remains exact and matches
    the combined forward map.
    """
    stencil = getattr(base_operator, "compiled_stencil", None)
    grid_shape = getattr(base_operator, "grid_shape", None)
    num_variables = getattr(base_operator, "num_variables", None)
    if stencil is None or grid_shape is None or num_variables is None:
        raise ValueError("base_operator must be a compiled observation operator.")
    if isinstance(stencil.station_locations, dict):
        raise ValueError(
            "Linear combinations require a shared station list so all variable "
            "blocks are colocated."
        )

    coefficients = np.asarray(coefficients, dtype=np.float32)
    block_count = len(stencil.blocks)
    if coefficients.shape != (block_count,):
        raise ValueError(
            f"coefficients must contain one value for each of the {block_count} "
            f"base blocks; got shape {coefficients.shape}."
        )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Linear-combination coefficients must be finite.")
    if np.all(coefficients == 0.0):
        raise ValueError("At least one linear-combination coefficient must be nonzero.")

    blocks = stencil.blocks
    observation_count = blocks[0].observation_count
    station_levels = blocks[0].station_levels
    for block in blocks[1:]:
        if block.observation_count != observation_count:
            raise ValueError(
                "Linear-combination blocks must have equal observation counts."
            )
        if not np.array_equal(block.station_levels, station_levels):
            raise ValueError(
                "Linear-combination blocks must have aligned station levels."
            )

    state_channels = np.concatenate(
        [
            block.state_channel_indices.reshape(
                observation_count, block.points_per_observation
            )
            for block in blocks
        ],
        axis=1,
    )
    locations = np.concatenate(
        [
            block.location_indices.reshape(
                observation_count, block.points_per_observation
            )
            for block in blocks
        ],
        axis=1,
    )
    # Each block contributes to the same station locations, so we can stack its
    # contributions side-by-side and scale each one by its coefficient before
    # collapsing them into a single stencil block.
    combined_weights = np.concatenate(
        [
            coefficient
            * block.weights.reshape(observation_count, block.points_per_observation)
            for coefficient, block in zip(coefficients, blocks)
        ],
        axis=1,
    )
    mixed_channels = np.concatenate(
        [block.mixed_channel_indices for block in blocks], axis=1
    )
    mixed_weights = np.concatenate(
        [
            coefficient * block.mixed_channel_weights
            for coefficient, block in zip(coefficients, blocks)
        ],
        axis=1,
    )
    combined_block = _ObservationBlock(
        variable_index=-1,
        state_channel_indices=state_channels.reshape(-1).astype(np.int32),
        location_indices=locations.reshape(-1).astype(np.int32),
        weights=combined_weights.reshape(-1).astype(np.float32),
        observation_count=observation_count,
        points_per_observation=combined_weights.shape[1],
        mixed_channel_indices=mixed_channels.astype(np.int32),
        mixed_channel_weights=mixed_weights.astype(np.float32),
        station_levels=station_levels.copy(),
    )
    combined_stencil = _CompiledObservationStencil(
        blocks=(combined_block,),
        station_locations=stencil.station_locations,
    )
    return _create_compiled_linear_operator(
        grid_shape,
        num_variables,
        combined_stencil,
        operator_name="linear combination",
    )

# ==== Nonlinear obs operator ====

def create_nonlinear_observation_operator(
    forward_operator: Callable[[tf.Tensor], tf.Tensor],
    state_shape: tuple[int, int, int],
    observation_count: int,
    adjoint_operator: Optional[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]] = None,
    operator_name: str = "nonlinear observation operator",
) -> tuple[Callable, Callable]:
    """Adapt a nonlinear forward model to the SDA ``(H, H_adjoint)`` contract.

    ``forward_operator(x)`` must return one rank-2 tensor with shape
    ``(batch, observation_count)``. If supplied, ``adjoint_operator(x, r)`` must
    return the Jacobian-transpose product ``J_H(x)^T r`` with the same shape as
    ``x``. This is the appropriate hook for CRTM's tangent-linear/adjoint or
    K-matrix interface.

    When ``adjoint_operator`` is omitted, TensorFlow computes the vector-Jacobian
    product with ``GradientTape``. That path is intended for a differentiable
    TensorFlow neural-network emulator; it cannot differentiate through
    ``tf.numpy_function`` or an external CRTM library.

    The adapter exposes one observation block. Per-state-channel ``weights`` in
    ``compute_likelihood_gradient`` are deliberately unsupported because radiance
    channels do not have a one-to-one mapping to state channels.
    """
    if (
        len(state_shape) != 3
        or any(isinstance(size, (bool, np.bool_)) for size in state_shape)
        or any(int(size) != size or size <= 0 for size in state_shape)
    ):
        raise ValueError(
            "state_shape must be three positive integers (lat, lon, channels)."
        )
    if (
        isinstance(observation_count, (bool, np.bool_))
        or int(observation_count) != observation_count
        or observation_count <= 0
    ):
        raise ValueError("observation_count must be a positive integer.")

    lat_size, lon_size, num_variables = (int(size) for size in state_shape)
    observation_count = int(observation_count)

    def _forward_tensor(x: tf.Tensor) -> tf.Tensor:
        values = tf.cast(forward_operator(x), tf.float32)
        if values.shape.rank != 2:
            raise ValueError(
                f"{operator_name} must return rank-2 observations "
                "(batch, observations)."
            )
        static_count = values.shape[1]
        if static_count is not None and static_count != observation_count:
            raise ValueError(
                f"{operator_name} returned {static_count} observations; "
                f"expected {observation_count}."
            )
        checks = [
            tf.debugging.assert_equal(
                tf.shape(values)[0],
                tf.shape(x)[0],
                message=f"{operator_name} changed the batch size.",
            ),
            tf.debugging.assert_equal(
                tf.shape(values)[1],
                observation_count,
                message=f"{operator_name} returned the wrong observation count.",
            ),
        ]
        with tf.control_dependencies([check for check in checks if check is not None]):
            return tf.identity(values)

    def H(x: tf.Tensor) -> list[tf.Tensor]:
        x = _validate_state_shape(
            x, (lat_size, lon_size), num_variables, operator_name
        )
        return [_forward_tensor(x)]

    def H_adjoint(residual, x: tf.Tensor) -> tf.Tensor:
        """Evaluate the state-dependent vector-Jacobian product J_H(x)^T r."""
        x = _validate_state_shape(
            x, (lat_size, lon_size), num_variables, operator_name
        )
        residual, batch_size = _validate_and_normalize_residuals(
            residual, (observation_count,), f"{operator_name} H_adjoint"
        )
        batch_check = tf.debugging.assert_equal(
            batch_size,
            tf.shape(x)[0],
            message=f"{operator_name} residual batch size does not match the state.",
        )
        with tf.control_dependencies([batch_check] if batch_check is not None else []):
            residual_tensor = tf.identity(residual[0])

        if adjoint_operator is None:
            with tf.GradientTape() as tape:
                tape.watch(x)
                observations = _forward_tensor(x)
                objective = tf.reduce_sum(observations * residual_tensor)
            result = tape.gradient(objective, x)
            if result is None:
                raise ValueError(
                    f"{operator_name} is not differentiable by TensorFlow; provide "
                    "an explicit adjoint_operator."
                )
        else:
            result = tf.cast(adjoint_operator(x, residual_tensor), tf.float32)

        result = tf.convert_to_tensor(result)
        shape_check = tf.debugging.assert_equal(
            tf.shape(result),
            tf.shape(x),
            message=f"{operator_name} adjoint must return the full state shape.",
        )
        with tf.control_dependencies([shape_check] if shape_check is not None else []):
            return tf.identity(result)

    metadata = _ObservationMetadata(
        num_variables=num_variables,
        selected_variables=np.array([-1], dtype=np.int32),
        observation_counts=(observation_count,),
        station_levels_by_block=(np.zeros(observation_count, dtype=np.float32),),
        channel_indices_by_block=(
            np.zeros((observation_count, 1), dtype=np.int32),
        ),
        channel_weights_by_block=(
            np.ones((observation_count, 1), dtype=np.float32),
        ),
        station_source=[],
    )
    for operator in (H, H_adjoint):
        operator.metadata = metadata
        operator.selected_vars = metadata.selected_variables
        operator.grid_shape = (lat_size, lon_size)
        operator.num_variables = num_variables
        operator.requires_state = True
        operator.supports_channel_weights = False

    return H, H_adjoint


# ==== Composable obs operators through chaining ====

def create_composed_observation_operator(
    base_operator: Callable,
    base_adjoint: Callable,
    transform_operator: Callable[[list[tf.Tensor]], tf.Tensor],
    observation_count: int,
    transform_adjoint: Optional[
        Callable[[list[tf.Tensor], tf.Tensor], list[tf.Tensor]]
    ] = None,
    operator_name: str = "composed observation operator",
) -> tuple[Callable, Callable]:
    """Compose a state-to-local operator with an observation-space transform.

    The returned forward map is

        ``H(x) = transform_operator(base_operator(x))``.

    ``base_operator`` must follow this module's block-list contract. The
    transform receives that list and returns one rank-2 tensor with shape
    ``(batch, observation_count)``. It may be linear or nonlinear.

    This general composition can reproduce the linear-combination operator when
    ``transform_operator`` computes a fixed weighted sum of colocated base
    blocks. The dedicated linear-combination factory remains useful because it
    merges the weights into one compiled stencil and therefore avoids evaluating
    a separate transform and VJP at runtime.

    It can also represent a nonlinear observation operator whenever that map
    factors through the base operator:

        ``H_nonlinear(x) = F(base_operator(x))``.

    This condition is important. Composition cannot reproduce a full-state
    nonlinear operator if the base operator discards information needed by
    ``F``. The standalone nonlinear adapter remains appropriate when the
    nonlinear model consumes the full grid directly or already supplies a
    model-space adjoint.

    A primary use case is separating geometric sampling from observation
    physics, for example

        ``model state -> interpolated atmospheric profiles -> CRTM radiances``.

    In that case the base blocks must retain and consistently align every
    profile, surface, cloud, aerosol, and geometry input required by the
    radiative-transfer transform.

    The adjoint applies the chain rule in reverse. TensorFlow computes the
    transform VJP by default, after which ``base_adjoint`` maps the resulting
    block residuals back to model space. For an external transform such as CRTM,
    ``transform_adjoint(local_state, residual)`` may instead provide the VJP and
    must return one residual tensor per base observation block. Thus the reverse
    action is

        ``J_base(x)^T J_transform(base_operator(x))^T residual``.

    Unlike the standalone nonlinear adapter's explicit adjoint, which returns a
    full model-state tensor, ``transform_adjoint`` returns residuals in the
    intermediate block space; this wrapper then invokes ``base_adjoint``.

    Composition is recursive: because the returned pair follows the same SDA
    contract, it can be supplied as the base pair of another composition to
    construct chains containing more than two stages.
    """
    metadata = getattr(base_operator, "metadata", None)
    grid_shape = getattr(base_operator, "grid_shape", None)
    num_variables = getattr(base_operator, "num_variables", None)
    if metadata is None or grid_shape is None or num_variables is None:
        raise ValueError(
            "base_operator must expose observation metadata, grid_shape, and "
            "num_variables."
        )
    if not callable(base_adjoint) or not callable(transform_operator):
        raise TypeError(
            "base_adjoint and transform_operator must both be callable."
        )
    if transform_adjoint is not None and not callable(transform_adjoint):
        raise TypeError("transform_adjoint must be callable when provided.")
    if (
        isinstance(observation_count, (bool, np.bool_))
        or int(observation_count) != observation_count
        or observation_count <= 0
    ):
        raise ValueError("observation_count must be a positive integer.")

    lat_size, lon_size = (int(size) for size in grid_shape)
    num_variables = int(num_variables)
    observation_count = int(observation_count)
    base_observation_counts = metadata.observation_counts

    def _local_state(x: tf.Tensor) -> list[tf.Tensor]:
        values = base_operator(x)
        values, batch_size = _validate_and_normalize_residuals(
            values, base_observation_counts, f"{operator_name} base operator"
        )
        batch_check = tf.debugging.assert_equal(
            batch_size,
            tf.shape(x)[0],
            message=f"{operator_name} base operator changed the batch size.",
        )
        with tf.control_dependencies(
            [batch_check] if batch_check is not None else []
        ):
            return [tf.identity(value) for value in values]

    def _transform(local_state: list[tf.Tensor]) -> tf.Tensor:
        values = tf.cast(transform_operator(local_state), tf.float32)
        if values.shape.rank != 2:
            raise ValueError(
                f"{operator_name} transform must return rank-2 observations "
                "(batch, observations)."
            )
        static_count = values.shape[1]
        if static_count is not None and static_count != observation_count:
            raise ValueError(
                f"{operator_name} transform returned {static_count} observations; "
                f"expected {observation_count}."
            )
        checks = [
            tf.debugging.assert_equal(
                tf.shape(values)[0],
                tf.shape(local_state[0])[0],
                message=f"{operator_name} transform changed the batch size.",
            ),
            tf.debugging.assert_equal(
                tf.shape(values)[1],
                observation_count,
                message=f"{operator_name} transform returned the wrong observation count.",
            ),
        ]
        with tf.control_dependencies([check for check in checks if check is not None]):
            return tf.identity(values)

    def H(x: tf.Tensor) -> list[tf.Tensor]:
        x = _validate_state_shape(
            x, (lat_size, lon_size), num_variables, operator_name
        )
        return [_transform(_local_state(x))]

    def H_adjoint(residual, x: tf.Tensor) -> tf.Tensor:
        """Apply the chained VJP J_base(x)^T J_transform(local)^T residual."""
        x = _validate_state_shape(
            x, (lat_size, lon_size), num_variables, operator_name
        )
        residual, batch_size = _validate_and_normalize_residuals(
            residual, (observation_count,), f"{operator_name} H_adjoint"
        )
        batch_check = tf.debugging.assert_equal(
            batch_size,
            tf.shape(x)[0],
            message=f"{operator_name} residual batch size does not match the state.",
        )
        with tf.control_dependencies([batch_check] if batch_check is not None else []):
            residual_tensor = tf.identity(residual[0])

        local_state = _local_state(x)
        if transform_adjoint is None:
            with tf.GradientTape() as tape:
                tape.watch(local_state)
                observations = _transform(local_state)
                objective = tf.reduce_sum(observations * residual_tensor)
            local_residual = tape.gradient(
                objective,
                local_state,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
        else:
            local_residual = transform_adjoint(local_state, residual_tensor)

        local_residual, local_batch_size = _validate_and_normalize_residuals(
            local_residual,
            base_observation_counts,
            f"{operator_name} transform adjoint",
        )
        local_batch_check = tf.debugging.assert_equal(
            local_batch_size,
            tf.shape(x)[0],
            message=f"{operator_name} transform adjoint changed the batch size.",
        )
        with tf.control_dependencies(
            [local_batch_check] if local_batch_check is not None else []
        ):
            local_residual = [tf.identity(value) for value in local_residual]

        if getattr(base_adjoint, "requires_state", False):
            return base_adjoint(local_residual, x)
        return base_adjoint(local_residual)

    composed_metadata = _ObservationMetadata(
        num_variables=num_variables,
        selected_variables=np.array([-1], dtype=np.int32),
        observation_counts=(observation_count,),
        station_levels_by_block=(
            np.zeros(observation_count, dtype=np.float32),
        ),
        channel_indices_by_block=(
            np.zeros((observation_count, 1), dtype=np.int32),
        ),
        channel_weights_by_block=(
            np.ones((observation_count, 1), dtype=np.float32),
        ),
        station_source=[],
    )
    for operator in (H, H_adjoint):
        operator.metadata = composed_metadata
        operator.selected_vars = composed_metadata.selected_variables
        operator.grid_shape = (lat_size, lon_size)
        operator.num_variables = num_variables
        operator.requires_state = True
        operator.supports_channel_weights = False
        operator.base_operator = base_operator
        operator.transform_operator = transform_operator

    return H, H_adjoint

# ==== Compute likelihood guidance in the clean state ====

def compute_likelihood_gradient(
    x_0: tf.Tensor,
    y_obs: list[tf.Tensor] | tuple[tf.Tensor, ...],
    H: Callable,
    H_adjoint: Callable,
    t: tf.Tensor,
    weights: tf.Tensor | list | np.ndarray | None = None,  # Per-variable weights
    R_obs: float = 0.1,
    gamma: float = 0.01,
    guidance_scale: float = 1.0,
) -> tf.Tensor:
    """Compute the clean-state factor of the SDA likelihood score.

    The paper approximates ``p(y | x_t)`` by a Gaussian centered on
    ``H(x_0_hat)`` with VP-coordinate variance

        V_t = R_obs + gamma * (1 - alpha_bar_t) / alpha_bar_t.

    This function returns ``∇_{x_0} log p(y | x_t)``. It does not apply the
    denoiser Jacobian transpose; use ``compute_sda_posterior_guidance`` for the
    complete paper-style noisy-state likelihood score.
    """

    if not isinstance(y_obs, (list, tuple)):
        raise TypeError("y_obs must be a list/tuple with one tensor per observation block.")
    metadata = getattr(H, "metadata", None)
    expected_blocks = (
        len(metadata.observation_counts)
        if metadata is not None
        else len(getattr(H, "selected_vars", ()))
    )
    if len(y_obs) != expected_blocks:
        raise ValueError(
            f"y_obs contains {len(y_obs)} blocks; expected {expected_blocks}."
        )
    if not np.isfinite(R_obs) or R_obs < 0:
        raise ValueError(f"R_obs must be finite and non-negative; got {R_obs}.")
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError(f"gamma must be finite and non-negative; got {gamma}.")
    if not np.isfinite(guidance_scale):
        raise ValueError(f"guidance_scale must be finite; got {guidance_scale}.")
    if weights is not None and not getattr(H, "supports_channel_weights", True):
        raise ValueError(
            "Per-state-channel guidance weights are not supported by this "
            "observation operator. Apply observation-space weights inside H or "
            "its adjoint instead."
        )

    # Get cumulative alpha at timestep t
    ALPHA_BAR_t = tf.gather(ALPHA_BAR, t)

    # Paper likelihood variance: V = R + (sigma_t / mu_t)^2 Gamma.
    sigma_sq = 1.0 - ALPHA_BAR_t
    diffusion_obs_var = gamma * sigma_sq / (ALPHA_BAR_t + 1e-12)
    V = tf.cast(R_obs + diffusion_obs_var, tf.float32)
    
    # Forward observation operator: this is the current model prediction at the
    # state x_0, which is compared to the observed values in observation space.
    y_hat = H(x_0)
    if metadata is not None:
        y_obs = _validate_guidance_observations(
            y_obs, metadata, tf.shape(y_hat[0])[0]
        )
    else:
        y_obs = [tf.cast(block, tf.float32) for block in y_obs]

    # Residual in observation space with optional per-variable weighting.
    # This is the Gaussian mismatch term driving the guidance update.
    residual = []
    observation_weights = (
        _effective_observation_weights(H, weights) if weights is not None else None
    )
    for i, (y_hat_i, y_obs_i) in enumerate(zip(y_hat, y_obs)):
        res_i = y_hat_i - y_obs_i
        if observation_weights is not None:
            res_i = res_i * observation_weights[i]

        residual.append(res_i * guidance_scale)

    weighted_residual = [res_i / V for res_i in residual]

    # Adjoint: map residual back to full state space.
    # Divide by V for noise-aware scaling: H^T (H x_0 - y) / V.
    if getattr(H_adjoint, "requires_state", False):
        score = -H_adjoint(weighted_residual, x_0)
    else:
        score = -H_adjoint(weighted_residual)

    return score

# ==== Compute background guidance in the clean state ====

def compute_background_gradient(
    x_0: tf.Tensor,
    x_b: tf.Tensor,
    t: tf.Tensor,
    R_bck: tf.Tensor | float = 0.1,
    background_gamma: tf.Tensor | float = 0.01,
    background_guidance_scale: float = 1.0,
) -> tf.Tensor:
    """Compute the clean-state gradient of a Gaussian background factor.

    The background is centered at ``x_b`` with diagonal covariance
    ``R_bck``. As with the observation likelihood, uncertainty in the clean
    prediction is represented in VP coordinates by the predictive variance

        V_b,t = R_bck + background_gamma * (1 - alpha_bar_t) / alpha_bar_t.

    ``R_bck`` and ``background_gamma`` may be scalars or positive tensors
    broadcastable to the state shape. This returns the closed-form plug-in
    gradient ``background_guidance_scale * (x_b - x_0) / V_b,t``.
    """
    if not np.isfinite(background_guidance_scale):
        raise ValueError(f"background_guidance_scale must be finite; got {background_guidance_scale}.")

    x_b = tf.cast(x_b, tf.float32)
    x_b = tf.stop_gradient(tf.identity(x_b))
    R_bck = tf.cast(R_bck, tf.float32)
    background_gamma = tf.cast(background_gamma, tf.float32)
    checks = [
        tf.debugging.assert_all_finite(R_bck, "R_bck must be finite."),
        tf.debugging.assert_positive(R_bck, "R_bck must be positive."),
        tf.debugging.assert_all_finite(
            background_gamma, "background_gamma must be finite."
        ),
        tf.debugging.assert_non_negative(
            background_gamma, "background_gamma must be non-negative."
        ),
    ]
    with tf.control_dependencies([check for check in checks if check is not None]):
        alpha_bar_t = tf.cast(tf.gather(ALPHA_BAR, t), tf.float32)
        diffusion_background_var = (
            background_gamma
            * (1.0 - alpha_bar_t)
            / (alpha_bar_t + 1e-12)
        )
        V_b = tf.stop_gradient(tf.identity(R_bck + diffusion_background_var))
    return (
        tf.cast(background_guidance_scale, tf.float32)
        * (x_b - x_0) / V_b
    )


# ==== Compute SDA guidance through the denoiser ====

def compute_sda_posterior_guidance(
    x_t: tf.Tensor,
    denoise_fn: Callable[[tf.Tensor], tf.Tensor],
    y_obs: list[tf.Tensor] | tuple[tf.Tensor, ...] | None,
    H: Callable | None,
    H_adjoint: Callable | None,
    t: tf.Tensor,
    weights: tf.Tensor | list | np.ndarray | None = None,
    R_obs: float = 0.1,
    gamma: float = 0.01,
    guidance_scale: float = 1.0,
    x_b: tf.Tensor | None = None,
    R_bck: tf.Tensor | float = 0.1,
    background_gamma: tf.Tensor | float = 0.01,
    background_guidance_scale: float = 1.0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Propagate clean-state likelihood and background guidance through the denoiser.

    ``compute_likelihood_gradient`` supplies the clean-state guidance using the
    paper likelihood variance. When ``x_b`` is supplied, it defines an
    additional Gaussian background factor centered at ``x_b`` with predictive
    variance ``R_bck + background_gamma * sigma_t^2 / mu_t^2``. Its
    clean-state log-density gradient,
    scaled by ``background_guidance_scale``, is added to the likelihood
    gradient. This function then computes
    ``J_D(x_t)^T g_0`` and converts that noisy-state score to the equivalent VP
    clean-prediction correction
    ``delta_x_0 = (sigma_t^2 / mu_t) * likelihood_score``.

    Returns ``(guided_x_0, posterior_guidance_score)``. Without ``x_b``, the
    second value is the likelihood score used previously.
    """
    x_t = tf.cast(x_t, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_t)
        x_0 = tf.cast(denoise_fn(x_t), tf.float32)

    clean_posterior_gradient = tf.zeros_like(x_0, dtype=tf.float32)
    if y_obs is not None:
        clean_likelihood_gradient = compute_likelihood_gradient(
            x_0=x_0,
            y_obs=y_obs,
            H=H,
            H_adjoint=H_adjoint,
            t=t,
            weights=weights,
            R_obs=R_obs,
            gamma=gamma,
            guidance_scale=guidance_scale,
        )
        clean_posterior_gradient = clean_posterior_gradient + clean_likelihood_gradient
    if x_b is not None:
        clean_background_gradient = compute_background_gradient(
            x_0=x_0,
            x_b=x_b,
            t=t,
            R_bck=R_bck,
            background_gamma=background_gamma,
            background_guidance_scale=background_guidance_scale,
        )
        clean_posterior_gradient = clean_posterior_gradient + clean_background_gradient

    posterior_guidance_score = tape.gradient(
        x_0,
        x_t,
        output_gradients=tf.stop_gradient(clean_posterior_gradient),
    )
    if posterior_guidance_score is None:
        raise ValueError(
            "The denoiser output is not differentiable with respect to x_t; "
            "paper-style SDA guidance requires the denoiser vector-Jacobian product."
        )

    alpha_bar_t = tf.cast(tf.gather(ALPHA_BAR, t), tf.float32)
    clean_correction_scale = (1.0 - alpha_bar_t) / tf.sqrt(
        alpha_bar_t + 1e-12
    )
    guided_x_0 = x_0 + clean_correction_scale * posterior_guidance_score
    return guided_x_0, posterior_guidance_score

# ==== Create station locations for pseudo-observations ====

def create_pseudo_station_locations(
    lat_size: int,
    lon_size: int,
    subsampling_factor: int,
    layout: str,
    operator_type: str,
    random_seed: int,
    level_size: int,
    num_variables: int,
    num_3d_fields: int,
    num_2d_fields: int,
) -> tuple[_StationGroups, dict[int, str]]:
    """Create pseudo-station groups and assign them to model variables.

    Station locations are generated as ``(lat, lon, level)`` triplets. Their
    vertical coordinates are fractional model-level indices. Locations are then
    grouped by their lower-bracketing level, and each model variable is assigned
    to the group corresponding to its channel level. Appended 2D fields use the
    final pseudo-level group.
    """
    if lat_size <= 0 or lon_size <= 0:
        raise ValueError(
            f"lat_size and lon_size must be positive; got {(lat_size, lon_size)}."
        )
    if subsampling_factor < 1:
        raise ValueError(f"subsampling_factor must be >= 1; got {subsampling_factor}.")
    if level_size <= 0:
        raise ValueError(f"level_size must be positive; got {level_size}.")
    if operator_type not in (
        "index",
        "interp",
        "neighborhood",
        "footprint",
        "finite_difference",
    ):
        raise ValueError(f"Unsupported observation operator type: {operator_type}.")
    if layout not in ("uniform", "random"):
        raise ValueError(f"layout must be 'uniform' or 'random'; got {layout}.")

    if layout == "uniform":
        # Interpolating and footprint operators benefit from cell-centre positions;
        # index uses exact grid-point indices (offset 0.0).
        offset = 0.0 if operator_type in ("index", "finite_difference") else 0.5
        lat_stations = np.arange(offset, lat_size, subsampling_factor, dtype=np.float32)
        lon_stations = np.arange(offset, lon_size, subsampling_factor, dtype=np.float32)
        base = [(lat, lon) for lat in lat_stations for lon in lon_stations]
    else:
        # Random layout: match nominal count from a regular subsampled grid.
        n_stations = len(range(0, lat_size, subsampling_factor)) * len(range(0, lon_size, subsampling_factor))
        rng = np.random.default_rng(seed=random_seed)

        if operator_type == "index":
            lat_rand = rng.integers(0, lat_size, n_stations, dtype=np.int32).astype(np.float32)
            lon_rand = rng.integers(0, lon_size, n_stations, dtype=np.int32).astype(np.float32)
        else:
            lat_rand = rng.uniform(0.0, lat_size - 1, n_stations).astype(np.float32)
            lon_rand = rng.uniform(0.0, lon_size - 1, n_stations).astype(np.float32)

        base = list(zip(lat_rand.tolist(), lon_rand.tolist()))

    # Preserve the original horizontal observation density at every level.
    # Atmospheric stations lie halfway toward the next level so vertical
    # interpolation is active. The final coordinate is either the reserved 2D
    # pseudo-level or the exact top atmospheric level, depending on the layout.
    num_atmospheric_levels = level_size - 1
    level_values = [
        min(float(level_idx), float(num_atmospheric_levels - 1))
        for level_idx in range(num_atmospheric_levels)
    ]
    level_values.append(float(level_size - 1))
    station_locations = [
        (lat, lon, level)
        for level in level_values
        for lat, lon in base
    ]
    if not station_locations:
        raise ValueError("No station locations were generated.")

    station_groups = {
        f"level_{level_idx}": [] for level_idx in range(level_size)
    }
    for lat, lon, level in _normalize_station_locations(station_locations):
        # Fractional atmospheric observations belong to their lower bracketing
        # level; the interpolation stencil also includes the upper level. The
        # final integer level is reserved for 2D / integrated fields.
        lower_level = int(np.clip(np.floor(level), 0, level_size - 1))
        station_groups[f"level_{lower_level}"].append(
            (lat, lon, float(level))
        )

    num_levels = level_size - 1 if num_2d_fields > 0 else level_size
    station_locations_per_var = {}
    for var_idx in range(num_variables):
        if var_idx < num_3d_fields * num_levels:
            level_idx = var_idx % num_levels
            group_name = f"level_{level_idx}"
        else:
            group_name = f"level_{num_levels}"
        station_locations_per_var[var_idx] = group_name

    return station_groups, station_locations_per_var

# ==== Load station locations and data from file for real observations according to Evan's files ====

def load_real_station_locations(
    station_locations_path: str,
) -> tuple[_StationGroups, dict[int, str]]:
    """Load real-observation station locations indexed by model variable.

    ``station_locations_path`` must be a NumPy ``.npz`` archive whose entry
    names are model channel indices. Each entry stores a ``(station, 3)`` array
    of ``(lat, lon, level)`` coordinates in model-grid coordinates.

    Returns the same ``(station_groups, station_locations_per_var)`` pair as
    :func:`create_pseudo_station_locations`, ready for
    :func:`build_observation_operator`. The group assignments are derived from
    the NPZ entry names; no separate mapping file is required.
    """
    archive = np.load(station_locations_path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError(
            f"station_locations_path must contain a NumPy .npz archive; "
            f"got {station_locations_path}."
        )

    try:
        if not archive.files:
            raise ValueError("The station-locations NPZ archive is empty.")

        station_groups = {}
        station_locations_per_var = {}
        for raw_variable in archive.files:
            try:
                variable_index = int(raw_variable)
            except ValueError as exc:
                raise ValueError(
                    f"NPZ entry '{raw_variable}' is not an integer variable index."
                ) from exc
            if variable_index < 0 or str(variable_index) != raw_variable:
                raise ValueError(
                    f"NPZ entry '{raw_variable}' must be a non-negative integer "
                    "variable index."
                )

            locations = np.asarray(archive[raw_variable])
            if locations.ndim != 2 or locations.shape[1] != 3:
                raise ValueError(
                    f"Station locations for variable {variable_index} must have "
                    f"shape (stations, 3); got {locations.shape}."
                )
            normalized = _normalize_station_locations(locations)
            if not normalized:
                logging.warning(
                    f"Station locations for variable {variable_index} are empty."
                )
                continue
            station_groups[raw_variable] = normalized
            station_locations_per_var[variable_index] = raw_variable
    finally:
        archive.close()

    return station_groups, station_locations_per_var


def load_real_observation_data(
    obs_data_path: str,
) -> dict[int, np.ndarray]:
    """Load real station observations indexed directly by model variable.

    ``obs_data_path`` must be a NumPy ``.npz`` archive whose entry names are
    model channel indices. Each entry stores either a ``(stations,)`` array or a
    ``(batch, stations)`` array of observations.

    One-dimensional arrays are promoted to a single-batch array. Returned data
    are finite ``float32`` arrays, ready to order by the selected variables and
    pass as ``y_obs``.
    """
    archive = np.load(obs_data_path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError(
            f"obs_data_path must contain a NumPy .npz archive; "
            f"got {obs_data_path}."
        )

    try:
        if not archive.files:
            raise ValueError("The station-data NPZ archive is empty.")

        obs_data_per_var = {}
        for raw_variable in archive.files:
            try:
                variable_index = int(raw_variable)
            except ValueError as exc:
                raise ValueError(
                    f"NPZ entry '{raw_variable}' is not an integer variable index."
                ) from exc
            if variable_index < 0 or str(variable_index) != raw_variable:
                raise ValueError(
                    f"NPZ entry '{raw_variable}' must be a non-negative integer "
                    "variable index."
                )

            try:
                values = np.asarray(archive[raw_variable], dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Station data for variable {variable_index} must be numeric."
                ) from exc
            if values.ndim == 1:
                values = values[np.newaxis, :]
            elif values.ndim != 2:
                raise ValueError(
                    f"Station data for variable {variable_index} must have shape "
                    f"(stations,) or (batch, stations); got {values.shape}."
                )
            if values.shape[0] == 0 or values.shape[1] == 0:
                logging.warning(
                    f"Station data for variable {variable_index} is empty."
                )
                continue
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Station data for variable {variable_index} contains non-finite values."
                )
            obs_data_per_var[variable_index] = values
    finally:
        archive.close()

    return obs_data_per_var


# ==== Build observation operators from specification of stations per variable ====

def build_observation_operator(
    args,
    station_groups: _StationGroups,
    station_locations_per_var: dict[int, str],
    lat_size: int,
    lon_size: int,
    level_size: int,
    num_variables: int,
    num_3d_fields: int,
    num_2d_fields: int,
) -> tuple[Callable | None, Callable | None]:
    """Build H and H_adjoint for prepared station assignments.

    num_3d_fields / num_2d_fields encode the actual model layout: 3D fields are
    stored as blocks of NUM_LEVELS channels, while 2D fields are appended as a
    single pseudo-level block after the 3D stack. This lets each channel map to
    the correct vertical subset rather than flattening everything into a single
    variable index space. Both pseudo and real observations can use this shared
    construction path after preparing their station groups and variable mapping.
    """
    num_levels = level_size - 1 if num_2d_fields > 0 else level_size
    station_count = sum(len(group) for group in station_groups.values())
    if station_count == 0:
        logging.warning("No station locations were provided.")
        return None, None

    common_operator_kwargs = {
        "grid_shape": (lat_size, lon_size),
        "num_variables": num_variables,
        "station_groups": station_groups,
        "station_locations_per_var": station_locations_per_var,
        "num_levels": num_levels,
        "num_3d_fields": num_3d_fields,
        "num_2d_fields": num_2d_fields,
    }

    if args.obs_operator == "index":
        logging.info(
            "Using index observation operator with %d %s stations (subsampling_factor=%d).",
            station_count,
            args.station_layout,
            args.subsampling_factor,
        )
        return create_observation_operator_index(
            **common_operator_kwargs,
        )

    if args.obs_operator == "interp":
        logging.info(
            "Using interp observation operator with %d %s stations (order=%d, subsampling_factor=%d).",
            station_count,
            args.station_layout,
            args.interp_order,
            args.subsampling_factor,
        )
        return create_observation_operator_interp(
            order=args.interp_order,
            **common_operator_kwargs,
        )

    if args.obs_operator == "footprint":
        logging.info(
            "Using circular footprint observation operator with %d %s stations (radius=%.3f grid cells).",
            station_count,
            args.station_layout,
            args.footprint_radius,
        )
        return create_observation_operator_footprint(
            radius=args.footprint_radius,
            **common_operator_kwargs,
        )

    if args.obs_operator == "finite_difference":
        spacing = (
            args.finite_difference_lat_spacing,
            args.finite_difference_lon_spacing,
        )
        logging.info(
            "Using %s finite-difference observation operator with %d %s stations (spacing=%s).",
            args.finite_difference_kind,
            station_count,
            args.station_layout,
            spacing,
        )
        return create_observation_operator_finite_difference(
            difference=args.finite_difference_kind,
            grid_spacing=spacing,
            **common_operator_kwargs,
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
        sigma = 0.425 * window_size

    logging.info(
        "Using neighborhood observation operator with %d %s stations (window_size=%d, kernel=%s, sigma=%.3f).",
        station_count,
        args.station_layout,
        window_size,
        args.neighborhood_kernel,
        sigma,
    )
    return create_observation_operator_neighborhood(
        window_size=window_size,
        kernel=args.neighborhood_kernel,
        sigma=sigma,
        **common_operator_kwargs,
    )
