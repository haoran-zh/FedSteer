import torch
import numpy as np


def OMP(stale_gradients, fresh_gradients, d_list, p_list, k, Lambda, lambda0=None):
    """
    Orthogonal Matching Pursuit for client selection and coefficient optimization.

    Args:
        stale_gradients: List of stale gradients (tensors) from candidate clients.
        fresh_gradients: List of fresh gradients (tensors) from all clients.
        d_list: List of weights d_i for each client.
        p_list: List of sampling probabilities p_i for each client.
        k: Number of clients to select.
        Lambda: Global regularization parameter.
        lambda0: Optional common regularization parameter for selection.

    Returns:
        selected_clients: Indices of selected stale gradients.
        coefficients: Coefficients s_i for each fresh gradient (list of tensors).
    """
    def flatten_weights(weights):
        last_layer_key = list(weights.keys())[-1]
        return weights[last_layer_key].reshape(-1)
        # return torch.cat([weights[k].reshape(-1) for k in sorted(weights.keys())])



    M = len(stale_gradients)
    N = len(fresh_gradients)

    # Handle edge cases
    if k <= 0:
        return [], []
    k = min(k, M)

    # Flatten all gradients
    stale_flat = [flatten_weights(g) for g in stale_gradients]
    fresh_flat = [flatten_weights(g) for g in fresh_gradients]
    d_list = torch.tensor(d_list, dtype=torch.float32)
    p_list = torch.tensor(p_list, dtype=torch.float32)

    # Set common regularization parameter
    if lambda0 is None:
        lambda0 = Lambda

    # Initialize data structures
    selected_clients = []
    Q_list = []
    A = None

    # OMP iterations
    for step in range(k):
        best_gain = -float('inf')
        best_v = None
        best_r = None
        best_b = None

        # Precompute current Q_S if not empty
        Q_S = torch.stack(Q_list, dim=1) if Q_list else None

        # Evaluate each candidate client
        for v in range(M):
            if v in selected_clients:  # if already selected, skip
                continue

            q_v = stale_flat[v]

            # Compute residual vector r
            if Q_S is None:
                r = q_v
                b = None
            else:
                b = Q_S.t() @ q_v
                a = torch.linalg.solve(A, b)
                r = q_v - Q_S @ a  # get the residual vector

            # Compute numerator: sum_i [ (d_i^2 / p_i) * (G_i^t · r)^2 ]
            numerator_val = 0.0
            for i in range(N):
                ip = torch.dot(fresh_flat[i], r)
                term = (d_list[i] ** 2 / p_list[i]) * (ip ** 2)
                numerator_val += term.item()

            # Compute denominator: ||r||^2 + lambda0
            denominator_val = torch.dot(r, r).item() + lambda0
            gain_v = numerator_val / denominator_val

            # Track best candidate
            if gain_v > best_gain:
                best_gain = gain_v
                best_v = v
                best_r = r
                best_b = b

        # Update selected set
        selected_clients.append(best_v)
        Q_list.append(stale_flat[best_v])

        # Update matrix A for next iteration
        if step == 0:
            A = torch.tensor([[torch.dot(Q_list[0], Q_list[0]) + lambda0]])
        else:
            new_size = len(selected_clients)
            new_A = torch.zeros((new_size, new_size))
            new_A[:new_size - 1, :new_size - 1] = A
            new_A[:new_size - 1, new_size - 1] = best_b
            new_A[new_size - 1, :new_size - 1] = best_b
            new_A[new_size - 1, new_size - 1] = torch.dot(Q_list[-1], Q_list[-1]) + lambda0
            A = new_A

    # Compute coefficients s_i for each client using client-specific regularization
    coefficients = []
    Q_S = torch.stack(Q_list, dim=1)  # d x k

    for i in range(N):
        lambda_i = Lambda * p_list[i] / (d_list[i] ** 2)
        A_i = Q_S.t() @ Q_S + lambda_i * torch.eye(len(selected_clients))
        b_i = Q_S.t() @ fresh_flat[i]
        s_i = torch.linalg.solve(A_i, b_i)
        coefficients.append(s_i)
    # convert coefficients to an array so I can use it as coefficients[i,:]
    coefficients = [c.detach().cpu().numpy() for c in coefficients]  # convert tensors to numpy arrays
    coefficients = np.array(coefficients)  # convert to numpy array for easier indexing

    return selected_clients, coefficients