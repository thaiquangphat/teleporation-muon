import copy
import random

import torch
from torch import nn


def calculate_grad_L2(net):
    total_L2 = 0.0
    for param in net.parameters():
        if param.grad is not None:
            total_L2 += param.grad.norm(2).item() ** 2
    return total_L2 ** 0.5


def generate_tele_scheduler(tele_batch, number_of_batch, tele_opt, tele_cons):
    if tele_opt == 0:
        tele_scheduler = [True] * tele_batch + [False] * (number_of_batch - tele_batch)
        random.shuffle(tele_scheduler)
    elif tele_opt == 1:
        tele_scheduler = [False] * number_of_batch
        count = 0
        for i in range(0, (tele_batch + tele_cons - 1) // tele_cons):
            for j in range(tele_cons):
                start = (i * number_of_batch) // ((tele_batch + tele_cons - 1) // tele_cons)
                idx = start + j
                if idx < number_of_batch and count < tele_batch:
                    tele_scheduler[idx] = True
                    count += 1
    else:
        tele_scheduler = [False] * number_of_batch
    return tele_scheduler


def gen_inv_matrix(n_head, d_head, tele_high, tele_low, tele_sign):
    size = d_head
    dtype = torch.float32
    M = torch.eye(size, dtype=dtype)
    M *= torch.rand(size, dtype=dtype) * (tele_high - tele_low) + tele_low
    if tele_sign == 1:
        mask = torch.randint(0, 2, (size,), dtype=dtype) * 2 - 1
        M[range(size), range(size)] *= mask
    indices = torch.randperm(size)
    M = M[indices, :]
    indices = torch.randperm(size)
    M = M[:, indices]
    M_inv = torch.transpose(M, -1, -2)
    M_inv = torch.where(M_inv != 0, 1.0 / M_inv, M_inv)
    M = torch.block_diag(*[M] * n_head)
    M_inv = torch.block_diag(*[M_inv] * n_head)
    return M, M_inv


def teleport_mlp(model, transforms):
    linear_layers = [m for m in model.net.modules() if isinstance(m, nn.Linear)]
    for idx, (M, M_inv) in transforms.items():
        current = linear_layers[idx]
        next_linear = linear_layers[idx + 1]
        with torch.no_grad():
            M = M.to(device=current.weight.device, dtype=current.weight.dtype)
            M_inv = M_inv.to(device=next_linear.weight.device, dtype=next_linear.weight.dtype)
            current.weight.data = torch.matmul(M, current.weight.data)
            if current.bias is not None:
                current.bias.data = torch.matmul(M, current.bias.data.unsqueeze(-1)).squeeze(-1)
            next_linear.weight.data = torch.matmul(next_linear.weight.data, M_inv)


def try_teleportation(model, samples, targets, criterion, args, high, low, sign=0):
    model.eval()
    base_model = copy.deepcopy(model)
    base_outputs = base_model(samples)
    base_loss = criterion(base_outputs, targets)
    base_loss.backward()
    grad_l2_base = calculate_grad_L2(base_model)
    max_grad_l2 = grad_l2_base
    max_transforms = None
    count_large_grad = 0

    linear_layers = [module for module in model.net.modules() if isinstance(module, nn.Linear)]
    if args.tele_layer == "all":
        layer_indices = range(len(linear_layers) - 1)
    elif args.tele_layer == "first":
        layer_indices = [0]
    else:
        layer_indices = [len(linear_layers) - 2]

    for _ in range(args.n_teleport - 1):
        model_i = copy.deepcopy(model)
        transforms = {
            idx: gen_inv_matrix(
                n_head=1,
                d_head=linear_layers[idx].out_features,
                tele_high=high,
                tele_low=low,
                tele_sign=sign,
            )
            for idx in layer_indices
        }
        teleport_mlp(model_i, transforms)
        outputs_i = model_i(samples)
        loss_i = criterion(outputs_i, targets)
        loss_i.backward()
        grad_l2_i = calculate_grad_L2(model_i)
        if grad_l2_i > grad_l2_base:
            count_large_grad += 1
        if grad_l2_i > max_grad_l2:
            max_grad_l2 = grad_l2_i
            max_transforms = transforms

    if count_large_grad > args.n_teleport / 2 and max_transforms is not None:
        teleport_mlp(model, max_transforms)
    model.train()
