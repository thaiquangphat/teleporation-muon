import torch
from torch import nn
import copy
import torch.backends.cuda

# This is a placeholder for the teleportation scheduler generation.
# The actual implementation might be different based on the project's needs.
def generate_tele_scheduler(tele_batch, number_of_batch, tele_opt, tele_cons):
    import random
    if tele_opt == 0:  # random
        tele_scheduler = [True] * tele_batch + [False] * (number_of_batch - tele_batch)
        random.shuffle(tele_scheduler)
    elif tele_opt == 1:  # consecutive
        tele_scheduler = [False] * number_of_batch
        count = 0
        interval = number_of_batch // ((tele_batch + tele_cons - 1) // tele_cons)
        for i in range(0, (tele_batch + tele_cons - 1) // tele_cons):
            for j in range(tele_cons):
                idx = i * interval + j
                if idx < number_of_batch and count < tele_batch:
                    tele_scheduler[idx] = True
                    count += 1
    return tele_scheduler

def get_module_by_name(model, name):
    """Gets a module from a model by its dot-separated name."""
    names = name.split('.')
    module = model
    for n in names:
        module = getattr(module, n)
    return module

def set_module_by_name(model, name, new_module):
    """Sets a module in a model by its dot-separated name."""
    names = name.split('.')
    parent_module = model
    for n in names[:-1]:
        parent_module = getattr(parent_module, n)
    setattr(parent_module, names[-1], new_module)

def group_action(U_weight, V_weight, U_bias, V_bias, h, T, activation):
    """
    Applies the group action transformation to a pair of weights (U, V) and biases.
    This transformation aims to preserve the output of the two-layer block.
    h must have shape (features, N).
    V_out = g * V
    U_out = U * activation(V*h) * pinv(activation(g*V*h))
    """
    g = torch.eye(T.shape[0], device=T.device) + T

    # Original pre-activations and activations
    Vh = torch.matmul(V_weight, h) + V_bias.unsqueeze(1)
    sigma_Vh = activation(Vh)

    # Transformed V
    gV_weight = torch.matmul(g, V_weight)
    gV_bias = torch.matmul(g, V_bias)

    # Transformed pre-activations and activations
    gVh = torch.matmul(gV_weight, h) + gV_bias.unsqueeze(1)
    sigma_gVh = activation(gVh)

    # Pseudo-inverse of the transformed activation
    try:
        # Shape of sigma_gVh is (intermediate_size, N), pinv is (N, intermediate_size)
        pinv_sigma_gVh = torch.linalg.pinv(sigma_gVh)
    except torch.linalg.LinAlgError:
        pinv_sigma_gVh = torch.linalg.pinv(sigma_gVh + 1e-6 * torch.eye(sigma_gVh.shape[-1], device=sigma_gVh.device))

    # Calculate the transformation matrix for U's weights
    # delta_U_weight shape: (intermediate_size, intermediate_size)
    delta_U_weight = torch.matmul(sigma_Vh, pinv_sigma_gVh)

    # Apply transformation to U's weight
    U_out_weight = torch.matmul(U_weight, delta_U_weight)
    
    # Based on the group action theory, U's bias remains unchanged to preserve the output.
    U_out_bias = U_bias

    return U_out_weight, gV_weight, U_out_bias, gV_bias


def teleport(model, samples, targets, criterion, args):
    """
    Performs teleportation by finding a transformation that maximizes the gradient norm
    while keeping the loss invariant. This is done via gradient ascent on the transformation matrix T.
    """
    model.eval()
    device = next(model.parameters()).device

    # 1. Identify layer pairs and create transformation matrices T
    layer_pairs = []
    if args.tele_att:
        for i in range(args.num_hidden_layer):
            # Pair: (Query, Key)
            # The transformation is applied on the hidden dimension
            q_name = f"vit.encoder.layer.{i}.attention.attention.query"
            k_name = f"vit.encoder.layer.{i}.attention.attention.key"
            # The input 'h' is the output of the layer norm before the attention block
            h_hook_handle_name = f"vit.encoder.layer.{i}.layernorm_before"
            # Activation is identity for Q/K transformation
            activation = nn.Identity()
            pair = {
                "U_name": q_name, "V_name": k_name, "h_hook": h_hook_handle_name,
                "T_size": args.d_model, "activation": activation
            }
            layer_pairs.append(pair)

    if args.tele_mlp:
        for i in range(args.num_hidden_layer):
            # Pair: (Intermediate, Output)
            # The transformation is applied on the intermediate dimension
            U_name = f"vit.encoder.layer.{i}.output.dense"
            V_name = f"vit.encoder.layer.{i}.intermediate.dense"
            # The input 'h' is the output of the attention block
            h_hook_handle_name = f"vit.encoder.layer.{i}.attention.output"
            # Activation is the MLP's activation function (e.g., GELU)
            activation = get_module_by_name(model, f"vit.encoder.layer.{i}.intermediate.intermediate_act_fn")
            pair = {
                "U_name": U_name, "V_name": V_name, "h_hook": h_hook_handle_name,
                "T_size": args.intermediate_size, "activation": activation
            }
            layer_pairs.append(pair)

    T_list = [torch.zeros(p["T_size"], p["T_size"], device=device, requires_grad=True) for p in layer_pairs]

    # --- Optimization loop for T ---
    # We use the sdp_kernel context manager to disable optimized attention kernels (like Flash Attention)
    # because they do not support the second-order derivatives needed for this algorithm.
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        for _ in range(args.tele_steps):
            # Create a temporary model to apply transformations
            temp_model = copy.deepcopy(model)
            temp_model.train() # Set to train for gradient calculation

            # --- Register forward hooks to capture intermediate activations 'h' ---
            activations = {}
            def get_activation(name):
                def hook(model, input, output):
                    activations[name] = output
                return hook

            hooks = []
            for pair in layer_pairs:
                hook_name = pair["h_hook"]
                module_to_hook = get_module_by_name(temp_model, hook_name)
                hooks.append(module_to_hook.register_forward_hook(get_activation(hook_name)))
            
            _ = temp_model(samples)
            for hook in hooks:
                hook.remove()

            # --- Monkey-patch the model to use transformed weights ---
            transformed_weights = []
            original_forwards = {}

            for i, pair in enumerate(layer_pairs):
                T = T_list[i]
                U_module = get_module_by_name(temp_model, pair["U_name"])
                V_module = get_module_by_name(temp_model, pair["V_name"])
                h = activations[pair["h_hook"]].detach()
                h_transposed = h.view(-1, h.shape[-1]).t()

                U_w_new, V_w_new, U_b_new, V_b_new = group_action(
                    U_module.weight, V_module.weight, U_module.bias, V_module.bias,
                    h_transposed, T, pair["activation"]
                )
                transformed_weights.extend([U_w_new, V_w_new, U_b_new, V_b_new])

                original_forwards[pair["U_name"]] = U_module.forward
                original_forwards[pair["V_name"]] = V_module.forward
                
                def make_forward(weight, bias):
                    return lambda x: nn.functional.linear(x, weight, bias)

                U_module.forward = make_forward(U_w_new, U_b_new)
                V_module.forward = make_forward(V_w_new, V_b_new)

            # --- Calculate the objective: ||dL/dW||^2 ---
            outputs = temp_model(pixel_values=samples).logits
            loss = criterion(outputs, targets)
            
            # Get gradients dL/dW for the TRANSFORMED weights
            grad_params = torch.autograd.grad(loss, transformed_weights, create_graph=True, allow_unused=True)
            
            # Calculate the squared norm of the gradients
            grad_norm_sq = sum(g.pow(2).sum() for g in grad_params if g is not None)

            # --- Gradient ascent step on T ---
            dT_list = torch.autograd.grad(grad_norm_sq, T_list, allow_unused=True)

            # Update T
            with torch.no_grad():
                for i in range(len(T_list)):
                    if dT_list[i] is not None:
                        T_list[i] += args.tele_lr * dT_list[i]

            # Restore original forward methods for the next loop iteration
            for name, func in original_forwards.items():
                module = get_module_by_name(temp_model, name)
                module.forward = func

    # --- End of optimization loop ---
    
    # 3. Apply the final optimized transformation to the original model
    print("Teleportation finished. Applying final transformation to the model.")
    with torch.no_grad():
        # Re-register hooks on the original model to get fresh activations
        activations = {}
        hooks = []
        for pair in layer_pairs:
            hook_name = pair["h_hook"]
            module_to_hook = get_module_by_name(model, hook_name)
            hooks.append(module_to_hook.register_forward_hook(get_activation(hook_name)))
        
        _ = model(samples)
        for hook in hooks:
            hook.remove()

        for i, pair in enumerate(layer_pairs):
            T = T_list[i]
            U_module = get_module_by_name(model, pair["U_name"])
            V_module = get_module_by_name(model, pair["V_name"])
            h = activations[pair["h_hook"]].detach()
            h_transposed = h.view(-1, h.shape[-1]).t()

            U_w_new, V_w_new, U_b_new, V_b_new = group_action(
                U_module.weight, V_module.weight, U_module.bias, V_module.bias,
                h_transposed, T, pair["activation"]
            )
            U_module.weight.data = U_w_new
            V_module.weight.data = V_w_new
            U_module.bias.data = U_b_new
            V_module.bias.data = V_b_new
            
    model.train()
