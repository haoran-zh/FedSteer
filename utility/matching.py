import torch
import numpy as np
from types import SimpleNamespace


def flatten_weights(weights):
    """Flattens the weights of a model, using only the last layer."""
    if not weights:
        return torch.tensor([])
    last_layer_key = list(weights.keys())[-1]
    return weights[last_layer_key].reshape(-1)


# def solve_s_for_subset(target_gradients_G, stale_gradients_Q_selected, lambda_i_list):
#     """Solves for the optimal coefficients 's' using per-client ridge regression."""
#     N, D = target_gradients_G.shape
#     k = stale_gradients_Q_selected.shape[0]
#     device = target_gradients_G.device
#
#     QQT = stale_gradients_Q_selected @ stale_gradients_Q_selected.T
#     GQT = target_gradients_G @ stale_gradients_Q_selected.T
#
#     s_optimal_list = []
#     for i in range(N):
#         lambda_i = lambda_i_list[i]
#         regularized_QQT = QQT + lambda_i * torch.eye(k, device=device)
#         try:
#             inv_matrix = torch.linalg.inv(regularized_QQT)
#         except torch.linalg.LinAlgError:
#             inv_matrix = torch.linalg.pinv(regularized_QQT)
#         s_i = GQT[i, :] @ inv_matrix
#         s_optimal_list.append(s_i)
#
#     return torch.stack(s_optimal_list)

# a stable version
def solve_s_for_subset(target_gradients_G, stale_gradients_Q_selected, lambda_i_list):
    """
    Solves for the optimal coefficients 's' using batched ridge regression.
    This version is numerically stable and computationally efficient.
    """
    N, D = target_gradients_G.shape
    k = stale_gradients_Q_selected.shape[0]
    device = target_gradients_G.device

    # These calculations remain the same
    QQT = stale_gradients_Q_selected @ stale_gradients_Q_selected.T  # Shape: (k, k)
    GQT = target_gradients_G @ stale_gradients_Q_selected.T  # Shape: (N, k)

    # --- Vectorized Solution ---

    # 1. Create a batched version of the regularization term.
    # Reshape lambdas to (N, 1, 1) for broadcasting
    lambda_tensor = torch.tensor(lambda_i_list, device=device).view(N, 1, 1)

    # Create a batch of k x k identity matrices
    identity_batch = torch.eye(k, device=device).unsqueeze(0).expand(N, -1, -1)

    # Batch of regularization matrices: lambda_i * I
    regularization_batch = lambda_tensor * identity_batch  # Shape: (N, k, k)

    # 2. Construct the batched matrix A for the system Ax=B.
    # QQT is broadcasted from (k, k) to (N, k, k)
    A_batch = QQT.unsqueeze(0) + regularization_batch  # Shape: (N, k, k)

    # 3. Construct the batched vector B.
    # We need to solve for s_i.T, so B is GQT.T
    B_batch = GQT.unsqueeze(2)  # Shape: (N, k, 1)

    # 4. Solve all N linear systems at once.
    # This solves A_i @ x_i = B_i for each i in the batch.
    s_optimal_T = torch.linalg.solve(A_batch, B_batch)  # Shape: (N, k, 1)

    # 5. Reshape the result to the desired (N, k) format.
    return s_optimal_T.squeeze(2)


def calculate_total_error(fresh_gradients, stale_gradients, selected_indices, coefficients, weights, Lambda):
    """Calculates the final regularized variance error for a given selection."""
    Q_selected = stale_gradients[selected_indices]
    approximated_G = coefficients @ Q_selected

    # ||G - sQ||^2 term, weighted per client
    residual_matrix = fresh_gradients - approximated_G
    error_term = torch.sum(weights.unsqueeze(1) * (residual_matrix ** 2))

    # Lambda * ||s||^2 term
    reg_term = Lambda * torch.sum(coefficients ** 2)

    return error_term + reg_term


def _run_omp_trial(initial_indices, k, target_gradients_G, stale_gradients_Q_pool, weights, lambda_i_values):
    """Helper function to run a single OMP trial from a given starting set."""
    M = stale_gradients_Q_pool.shape[0]
    N = target_gradients_G.shape[0]

    selected_indices = list(initial_indices)
    # Ensure remaining_indices are not in the initial set
    remaining_indices = [i for i in range(M) if i not in selected_indices]

    # If starting from a non-empty set, calculate the initial residual
    if selected_indices:
        Q_selected = stale_gradients_Q_pool[selected_indices]
        s_current = solve_s_for_subset(target_gradients_G, Q_selected, lambda_i_values)
        residual = target_gradients_G - (s_current @ Q_selected)
    else:
        residual = target_gradients_G.clone()

    s_final = None

    # Run greedy selection for the remaining number of clients
    for _ in range(k - len(selected_indices)):
        if not remaining_indices:
            break

        correlations = residual @ stale_gradients_Q_pool[remaining_indices].T
        scores = weights * (correlations ** 2)
        total_scores = scores.sum(dim=0)

        best_local_idx = torch.argmax(total_scores)
        best_global_idx = remaining_indices.pop(best_local_idx)
        selected_indices.append(best_global_idx)

        Q_selected = stale_gradients_Q_pool[selected_indices]
        s_current = solve_s_for_subset(target_gradients_G, Q_selected, lambda_i_values)
        residual = target_gradients_G - (s_current @ Q_selected)
        s_final = s_current

    # If k was smaller than initial set, calculate s_final now
    if s_final is None and selected_indices:
        Q_selected = stale_gradients_Q_pool[selected_indices]
        s_final = solve_s_for_subset(target_gradients_G, Q_selected, lambda_i_values)

    return selected_indices, s_final


def OMP(stale_gradients, fresh_gradients, d_list, p_list, k, Lambda, args):
    """
    Main OMP function with gradient normalization and output rescaling.
    """
    # --- 1. Data Preparation ---
    first_tensor = next(iter(stale_gradients[0].values()))
    device = first_tensor.device

    stale_flat_list = [flatten_weights(g).to(device) for g in stale_gradients]
    fresh_flat_list = [flatten_weights(g).to(device) for g in fresh_gradients]

    stale_gradients_Q_pool = torch.stack(stale_flat_list)
    target_gradients_G = torch.stack(fresh_flat_list)

    ## --- NORMALIZATION START --- ##
    # This is the new section for normalizing the inputs.

    # Calculate L2 norm for each gradient vector (row). keepdim=True helps with broadcasting.
    fresh_norms = torch.linalg.norm(target_gradients_G, dim=1, keepdim=True)
    stale_norms = torch.linalg.norm(stale_gradients_Q_pool, dim=1, keepdim=True)

    # Add a small epsilon to avoid division by zero for zero-norm vectors.
    epsilon = 1e-8

    # Create normalized versions of the gradient tensors.
    G_normalized = target_gradients_G / (fresh_norms + epsilon)
    Q_normalized = stale_gradients_Q_pool / (stale_norms + epsilon)

    # print(f"[INFO] Gradients have been normalized for OMP.")
    ## --- NORMALIZATION END --- ##

    d_list = torch.as_tensor(d_list, device=device, dtype=torch.float32)
    p_list = torch.as_tensor(p_list, device=device, dtype=torch.float32)

    M = Q_normalized.shape[0]
    k = min(k, M)
    if k <= 0:
        return [], np.array([])

    weights = (d_list ** 2) / (p_list + 1e-9)
    lambda_i_values = Lambda * (p_list + 1e-9) / (d_list ** 2 + 1e-9)

    # --- 2. Main Logic ---
    # The algorithm now runs on the NORMALIZED gradients.
    if args.three_sample:
        best_error = float('inf')
        best_selection = []
        best_coeffs_hat = torch.empty(0)  # These are coefficients for the normalized problem

        print(f"Running {args.num_trials} trials with 3-sample initialization...")
        for i in range(args.num_trials):
            initial_indices = np.random.choice(M, 3, replace=False)

            # Note: Pass normalized gradients to the trial runner
            selected_indices, s_hat_final = _run_omp_trial(initial_indices, k, G_normalized, Q_normalized,
                                                           weights.unsqueeze(1), lambda_i_values)

            if s_hat_final is not None and s_hat_final.numel() > 0:
                # Pass normalized gradients to the error calculation as well
                total_error = calculate_total_error(G_normalized, Q_normalized, selected_indices, s_hat_final, weights,
                                                    Lambda)
                print(f"  Trial {i + 1}: this error: {total_error:.6f}")

                if total_error < best_error:
                    best_error = total_error
                    best_selection = selected_indices
                    best_coeffs_hat = s_hat_final
                    print(f"  Trial {i + 1}: New best error found: {best_error:.6f}")

        selected_indices = best_selection
        s_final_hat = best_coeffs_hat

    else:
        # print("Running standard OMP...")
        selected_indices, s_final_hat = _run_omp_trial([], k, G_normalized, Q_normalized, weights.unsqueeze(1),
                                                       lambda_i_values)

    # --- 3. Format Output ---
    if s_final_hat is not None and s_final_hat.numel() > 0:
        ## --- RESCALING START --- ##
        # This new section rescales the coefficients back to the original gradient magnitudes.

        # Get the original norms of the selected stale gradients
        selected_stale_norms = stale_norms[selected_indices].view(1, -1)  # Shape (1, k)

        # Rescale coefficients: s_ij = s_hat_ij * (||G_i|| / ||q_j||)
        # We use broadcasting: (N, 1) / (1, k) -> (N, k) scaling matrix
        scaling_matrix = (fresh_norms + epsilon) / (selected_stale_norms + epsilon)
        s_rescaled = s_final_hat * scaling_matrix

        # print("[INFO] Output coefficients have been rescaled.")
        ## --- RESCALING END --- ##

        # Convert the rescaled tensor to a NumPy array
        coefficients_np = s_rescaled.detach().cpu().numpy()
    else:
        coefficients_np = np.array([])

    return selected_indices, coefficients_np


# def calculate_distr_var(G_mean, fresh_gradients, selected_indices, weights):
#     """Calculates the final regularized variance error for a given selection."""
#     Q_selected = stale_gradients[selected_indices]
#     G_current_mean = torch.mean(flat_gradients[selected_indices], dim=0)
#
#     # ||G - sQ||^2 term, weighted per client
#     residual_matrix = fresh_gradients - approximated_G
#     error_term = torch.sum(weights.unsqueeze(1) * (residual_matrix ** 2))
#
#     # Lambda * ||s||^2 term
#
#     return error_term






def random_improve(stale_gradients, fresh_gradients, d_list, p_list, k, Lambda, args):
    # --- 1. Data Preparation ---
    # (This section is identical to the OMP function)
    first_tensor = next(iter(stale_gradients[0].values()))
    device = first_tensor.device

    stale_flat_list = [flatten_weights(g).to(device) for g in stale_gradients]
    fresh_flat_list = [flatten_weights(g).to(device) for g in fresh_gradients]

    stale_gradients_Q_pool = torch.stack(stale_flat_list)
    target_gradients_G = torch.stack(fresh_flat_list)

    fresh_norms = torch.linalg.norm(target_gradients_G, dim=1, keepdim=True)
    stale_norms = torch.linalg.norm(stale_gradients_Q_pool, dim=1, keepdim=True)
    epsilon = 1e-8
    G_normalized = target_gradients_G / (fresh_norms + epsilon)
    Q_normalized = stale_gradients_Q_pool / (stale_norms + epsilon)

    d_list = torch.as_tensor(d_list, device=device, dtype=torch.float32)
    p_list = torch.as_tensor(p_list, device=device, dtype=torch.float32)

    weights = (d_list ** 2) / (p_list + 1e-9)
    lambda_i_values = Lambda * (p_list + 1e-9) / (d_list ** 2 + 1e-9)

    # --- 2. Uniform Initialization ---
    M = stale_gradients_Q_pool.shape[0]
    k = min(k, M)
    selected_indices = np.random.choice(M, k, replace=False).tolist()
    print(f"Starting with random subset: {selected_indices}")


    # --- 3. Iterative Greedy Swapping ---
    for step in range(args.num_swaps):  # Assuming args.num_swaps is defined, e.g., 5
        # First, calculate the error of the current selection to beat
        Q_selected_current = Q_normalized[selected_indices]
        s_current = solve_s_for_subset(G_normalized, Q_selected_current, lambda_i_values)
        # if args.distr_var is True:
        #     current_error = calculate_distr_var(G_true_mean, G_normalized, selected_indices, d_torch)
        # else:
        current_error = calculate_total_error(G_normalized, Q_normalized, selected_indices, s_current, weights, Lambda)

        best_gain = 0
        best_swap_pair = (None, None)  # Stores (client_to_remove, client_to_add)

        # Get the pool of clients that are not currently selected
        candidate_pool_to_add = [idx for idx in range(M) if idx not in selected_indices]

        # This is the computationally expensive part: O(k * (M-k))
        for client_to_remove in selected_indices:
            for client_to_add in candidate_pool_to_add:
                # Create a temporary candidate subset
                temp_selection = selected_indices.copy()
                temp_selection.remove(client_to_remove)
                temp_selection.append(client_to_add)

                # Solve for the optimal 's' and calculate error for this candidate
                # NOTE: This is the bottleneck, as it involves matrix inversions.
                Q_cand = Q_normalized[temp_selection]
                s_cand = solve_s_for_subset(G_normalized, Q_cand, lambda_i_values)
                cand_error = calculate_total_error(G_normalized, Q_normalized, temp_selection, s_cand, weights, Lambda)

                gain = current_error - cand_error

                if gain > best_gain:
                    best_gain = gain
                    best_swap_pair = (client_to_remove, client_to_add)

        # After checking all swaps, execute the best one if it's beneficial
        if best_swap_pair[0] is not None:
            client_out, client_in = best_swap_pair
            selected_indices.remove(client_out)
            selected_indices.append(client_in)
            print(
                f"  Step {step + 1}/{args.num_swaps}: Swapped out {client_out} for {client_in}. Gain: {best_gain:.4f}")
        else:
            # If no swap provides a positive gain, we've converged.
            print(f"  Step {step + 1}/{args.num_swaps}: No improvement found.")
            break

    # --- 4. Final Coefficient Calculation and Rescaling ---
    # After the final swap, calculate the coefficients for the final set
    Q_final_normalized = Q_normalized[selected_indices]
    s_final_hat = solve_s_for_subset(G_normalized, Q_final_normalized, lambda_i_values)

    # Rescale coefficients back to the original gradient magnitudes
    selected_stale_norms = stale_norms[selected_indices].view(1, -1)
    scaling_matrix = (fresh_norms + epsilon) / (selected_stale_norms + epsilon)
    s_rescaled = s_final_hat * scaling_matrix
    coefficients_np = s_rescaled.detach().cpu().numpy()

    print(f"Finished. Final subset: {selected_indices}")
    return selected_indices, coefficients_np

