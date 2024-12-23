from typing import List, Optional, Tuple

import numpy as np
from open3d._ml3d.torch.models.kpconv import batch_grid_subsampling, batch_neighbors
import torch

import detectionmetrics.utils.lidar as ul


def transform_input(points: np.ndarray, cfg: dict, sampler: ul.Sampler) -> Tuple[
    Tuple[
        torch.Tensor,
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ],
    List[np.ndarray],
]:
    """Transform point cloud data into input data for the model

    :param points: Point cloud data
    :type points: np.ndarray
    :param cfg: Dictionary containing model configuration file
    :type cfg: dict
    :param sampler: Object for sampling point cloud
    :type sampler: ul.Sampler
    :return: Model input data and selected indices
    :rtype: Tuple[ Tuple[ torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], ], List[np.ndarray], ]
    """
    p_list = []
    p0_list = []
    r_mask_list = []

    # Divide point cloud into chunks of at most max_in_points
    curr_num_points = 0
    min_in_points = cfg.get("min_in_points", 10000)

    while curr_num_points < min_in_points:
        # Sample points if required
        new_points, selected_indices, center_point = sampler.sample(
            points,
            num_points=min_in_points,
            radius=cfg.get("in_radius", 4.0),
        )

        new_points = new_points - center_point

        # Recenter point cloud if required
        if "recenter" in cfg:
            new_points = ul.recenter(new_points, cfg["recenter"]["dims"])

        in_pts = new_points
        n = in_pts.shape[0]

        # Randomly drop some points if max number of points have been exceeded
        residual_num_points = cfg.get("max_in_points", 20000) - curr_num_points
        if n > residual_num_points:
            input_inds = np.random.choice(n, size=residual_num_points, replace=False)
            in_pts = in_pts[input_inds, :]
            selected_indices = selected_indices[input_inds]
            n = input_inds.shape[0]

        curr_num_points += n

        # Accumulate each chunk data
        p_list += [in_pts]
        p0_list += [center_point]
        r_mask_list += [selected_indices]

    # Stack all data
    stacked_points = np.concatenate(p_list, axis=0)
    stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
    stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
    stacked_features = np.hstack((stacked_features, stacked_points[:, 2:3]))

    # From this point until the end of the function, we will get data ready to work as
    # inputs for the model
    conv_radius = cfg.get("conv_radius", 2.5)
    r_normal = cfg.get("first_subsampling_dl", 0.06) * conv_radius

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_stack_lengths = []

    # Generate input data for each block in the model
    layer_blocks = []
    arch = cfg["architecture"]
    for block in arch:
        # Get all blocks of the layer
        if not (
            "pool" in block
            or "strided" in block
            or "global" in block
            or "upsample" in block
        ):
            layer_blocks += [block]
            continue

        # Convolution neighbors indices
        if layer_blocks:
            conv_i = batch_neighbors(
                stacked_points, stacked_points, stack_lengths, stack_lengths, r_normal
            )
        else:
            # This layer only perform pooling, no neighbors required
            conv_i = np.zeros((0, 1), dtype=np.int32)

        # Pooling neighbors indices
        # If end of layer is a pooling operation
        if "pool" in block or "strided" in block:
            # New subsampling length
            dl = 2 * r_normal / conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling(
                stacked_points, stack_lengths, sampleDl=dl
            )

            # Subsample indices
            pool_i = batch_neighbors(
                pool_p, stacked_points, pool_b, stack_lengths, r_normal
            )

            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_neighbors(
                stacked_points, pool_p, stack_lengths, pool_b, 2 * r_normal
            )
        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = np.zeros((0, 1), dtype=np.int32)
            pool_p = np.zeros((0, 3), dtype=np.float32)
            pool_b = np.zeros((0,), dtype=np.int32)
            up_i = np.zeros((0, 1), dtype=np.int32)

        # Updating input lists
        input_points += [stacked_points]
        input_neighbors += [conv_i.astype(np.int64)]
        input_pools += [pool_i.astype(np.int64)]
        input_upsamples += [up_i.astype(np.int64)]
        input_stack_lengths += [stack_lengths]

        # New points for next layer
        stacked_points = pool_p
        stack_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer_blocks = []

        # Stop when meeting a global pooling or upsampling
        if "global" in block or "upsample" in block:
            break

    # Convert to torch tensors
    stacked_features = torch.from_numpy(stacked_features)
    input_points = [torch.from_numpy(a) for a in input_points]
    input_pools = [torch.from_numpy(a) for a in input_pools]
    input_neighbors = [torch.from_numpy(a) for a in input_neighbors]
    input_upsamples = [torch.from_numpy(a) for a in input_upsamples]
    input_stack_lengths = [torch.from_numpy(a) for a in input_stack_lengths]

    return (
        stacked_features,
        input_points,
        input_pools,
        input_neighbors,
        input_upsamples,
        input_stack_lengths,
    ), r_mask_list


def update_probs(
    new_probs: torch.Tensor,
    indices: torch.Tensor,
    test_probs: torch.Tensor,
    lengths: int,
    weight: float = 0.95,
) -> torch.Tensor:
    """Update test probabilities with new model output using weighted average for a
    smooth transition between predictions

    :param new_probs: New probabilities to be added to the test probabilities
    :type new_probs: torch.Tensor
    :param indices: Corresponding indices of the new probabilities
    :type indices: torch.Tensor
    :param test_probs: Test probabilities to be updated
    :type test_probs: torch.Tensor
    :param lengths: Length of each subsampled set of points
    :type lengths: int
    :param weight: Weight used in the weighted average, defaults to 0.95
    :type weight: float, optional
    :return: Updated test probabilities
    :rtype: torch.Tensor
    """
    # Format new probabilities
    new_probs = torch.nn.functional.softmax(new_probs, dim=-1)

    if isinstance(test_probs, np.ndarray):
        test_probs = torch.tensor(test_probs, device=new_probs.device)

    # Update probabilities using a weighted average for each subsampled set of points
    i0 = 0
    lengths = lengths[0].cpu().numpy()
    for b_i, length in enumerate(lengths):
        probs = new_probs[i0 : i0 + length]
        proj_mask = indices[b_i]
        test_probs[proj_mask] = weight * test_probs[proj_mask] + (1 - weight) * probs
        i0 += length

    return test_probs
