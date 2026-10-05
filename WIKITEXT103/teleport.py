import torch
import random
import copy
def calculate_grad_L2(net):
    total_L2 = 0.0
    for name, param in net.named_parameters():
        if param.grad is not None:
            total_L2 += param.grad.norm(2).item() ** 2
    return total_L2 ** 0.5

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
def gen_rope_inv_matrix(n_head, d_head, tele_high, tele_low, tele_sign):
    if d_head % 2 != 0: raise ValueError(f"d_head must be even for RoPE structure, got d_head={d_head}")
    dtype = torch.float32
    two_pi = 2.0 * torch.pi
    def make_head_blocks():
        blocks = []
        inv_blocks = []
        # sample all thetas and rhos for this head
        thetas = torch.rand(d_head // 2, dtype=dtype) * two_pi
        rhos = torch.rand(d_head // 2, dtype=dtype) * (tele_high - tele_low) + tele_low
        if tele_sign == 1:
            signs = (torch.randint(0, 2, (d_head // 2,)) * 2 - 1).to(dtype)
            rhos = rhos * signs
        cos = torch.cos(thetas)
        sin = torch.sin(thetas)
        # build 2x2 blocks
        for c, s, r in zip(cos, sin, rhos):
            # M_block = rho * R(theta)
            M_block = torch.tensor([[r * c, -r * s], [r * s,  r * c]], dtype=dtype)
            # M_inv_block = (1/rho) * R(-theta) = (1/rho) * [[c, s], [-s, c]]
            rinv = 1.0 / r
            M_inv_block = torch.tensor([[rinv * c,  rinv * s], [-rinv * s, rinv * c]], dtype=dtype)
            blocks.append(M_block)
            inv_blocks.append(M_inv_block)
        # stitch the 2x2 blocks into a d_head x d_head block-diagonal for this head
        return torch.block_diag(*blocks), torch.block_diag(*inv_blocks)
    per_head_M = []
    per_head_Minv = []
    for _ in range(n_head):
        Mh, Minvh = make_head_blocks()
        per_head_M.append(Mh)
        per_head_Minv.append(Minvh)
    M = torch.block_diag(*per_head_M)
    M_inv = torch.block_diag(*per_head_Minv)
    return M, M_inv

def generate_tele_scheduler(tele_batch, number_of_batch, tele_opt, tele_cons):
    if tele_opt == 0:  # random
        tele_scheduler = [True] * tele_batch + [False] * (number_of_batch - tele_batch)
        random.shuffle(tele_scheduler)
    elif tele_opt == 1:  # consecutive
        tele_scheduler = [False] * number_of_batch
        count = 0
        for i in range(0, (tele_batch + tele_cons - 1) // tele_cons):
            for j in range(tele_cons):
                if ((i * number_of_batch) // ((tele_batch + tele_cons - 1) // tele_cons)) +  j < number_of_batch and count < tele_batch:
                    tele_scheduler[((i * number_of_batch) // ((tele_batch + tele_cons - 1) // tele_cons)) + j] = True
                    count += 1
    return tele_scheduler

def teleportation_att(mem_transformer_model, M, M_inv, tele_layer):
    # Navigate to decoder layers
    layers = mem_transformer_model.layers
    if tele_layer == "all":
        for i, block in enumerate(layers):
            q_linear = block.dec_attn.q_net
            # Note: kv_net contains both key and value, we need to handle this carefully
            kv_linear = block.dec_attn.kv_net
            with torch.no_grad():
                M = M.to(q_linear.weight.device)
                M_inv = M_inv.to(kv_linear.weight.device)
                # Apply teleportation to query
                q_linear.weight.data = torch.matmul(M, q_linear.weight.data)
                # Note: q_net has bias=False, so no bias transformation needed
                # Apply inverse teleportation to key part of kv_net
                # kv_net outputs [key, value] concatenated, so key is first half
                kv_weight = kv_linear.weight.data
                key_dim = kv_weight.shape[0] // 2  # Assuming key and value have same dim
                # Transform only the key part (first half of the output features)
                kv_weight[:key_dim, :] = torch.matmul(torch.transpose(M_inv, -1, -2), kv_weight[:key_dim, :])
                kv_linear.weight.data = kv_weight
                # Note: kv_net has bias=False, so no bias transformation needed
                
    else:
        if tele_layer == "first":
            block = layers[0]
        else:
            block = layers[-1]
        q_linear = block.dec_attn.q_net
        kv_linear = block.dec_attn.kv_net
        with torch.no_grad():
            M = M.to(q_linear.weight.device)
            M_inv = M_inv.to(kv_linear.weight.device)
            # Apply teleportation to query
            q_linear.weight.data = torch.matmul(M, q_linear.weight.data)
            # Apply inverse teleportation to key part of kv_net
            kv_weight = kv_linear.weight.data
            key_dim = kv_weight.shape[0] // 2  # Assuming key and value have same dim
            # Transform only the key part (first half of the output features)
            kv_weight[:key_dim, :] = torch.matmul(torch.transpose(M_inv, -1, -2), kv_weight[:key_dim, :])
            kv_linear.weight.data = kv_weight
def teleportation_mlp(mem_transformer_model, M, M_inv):
    # Navigate to decoder layers
    layers = mem_transformer_model.layers
    
    for i, block in enumerate(layers):
        # Access the MLP layers in pos_ff.CoreNet
        intermediate_linear = block.pos_ff.CoreNet[0]  # First Linear layer (128 -> 2048)
        output_linear = block.pos_ff.CoreNet[3]        # Second Linear layer (2048 -> 128)
        
        with torch.no_grad():
            M = M.to(intermediate_linear.weight.device)
            M_inv = M_inv.to(output_linear.weight.device)
            # Apply teleportation to intermediate layer
            intermediate_linear.weight.data = torch.matmul(M, intermediate_linear.weight.data)
            intermediate_linear.bias.data = torch.matmul(intermediate_linear.bias.data, torch.transpose(M, -1, -2))
            # Apply inverse teleportation to output layer
            output_linear.weight.data = torch.matmul(output_linear.weight.data, M_inv)
def try_teleportation(model, data, target,mems, high, low, sign, args):
    model.eval()
    # Base forward pass
    base_model = copy.deepcopy(model)
    ret = base_model(data, target, *copy.deepcopy(mems), carry_over_fast_weight=args.carry_over_fast_weight)
    base_loss, base_mems = ret[0], ret[1:]
    base_loss = base_loss.float().mean().type_as(base_loss)
    base_loss.backward()
    grad_l2_base = calculate_grad_L2(base_model)
    max_grad_l2 = grad_l2_base
    max_M_att = None
    max_M_mlp = None
    max_M_att_inv = None
    max_M_mlp_inv = None
    count_large_grad = 0
    for _ in range(args.n_teleport - 1):
        model_i = copy.deepcopy(model)
        if args.tele_att:
            if args.attn_type == 123:
                M_att, M_att_inv = gen_rope_inv_matrix(
                    n_head=args.n_head,
                    d_head=args.d_head,
                    tele_high=high,
                    tele_low=low,
                    tele_sign=sign
                ) 
            else:
                M_att, M_att_inv = gen_inv_matrix(
                    n_head=args.n_head,
                    d_head=args.d_head,
                    tele_high=high,
                    tele_low=low,
                    tele_sign=sign
                )
            teleportation_att(model_i, M_att, M_att_inv, args.tele_layer)
        if args.tele_mlp:
            M_mlp, M_mlp_inv = gen_inv_matrix(
                n_head=1,
                d_head=args.d_inner,
                tele_high=high,
                tele_low=low,
                tele_sign=sign
            )
            teleportation_mlp(model_i, M_mlp, M_mlp_inv)
        
        ret_i = model_i(data, target, *copy.deepcopy(mems), carry_over_fast_weight=args.carry_over_fast_weight)
        loss_i, mems_i = ret_i[0], ret_i[1:]
        loss_i = loss_i.float().mean().type_as(loss_i)
        loss_i.backward()
        grad_l2_i = calculate_grad_L2(model_i)
        
        if (grad_l2_i > grad_l2_base):
            count_large_grad += 1
        if grad_l2_i > max_grad_l2:
            max_grad_l2 = grad_l2_i
            if args.tele_att:
                max_M_att = M_att
                max_M_att_inv = M_att_inv
            if args.tele_mlp:
                max_M_mlp = M_mlp
                max_M_mlp_inv = M_mlp_inv
                
    if (count_large_grad > args.n_teleport / 2):
        if max_M_att is not None and max_M_att_inv is not None:
            teleportation_att(model, max_M_att, max_M_att_inv, args.tele_layer)
        if max_M_mlp is not None and max_M_mlp_inv is not None:
            teleportation_mlp(model, max_M_mlp, max_M_mlp_inv)
        print(max_grad_l2 / grad_l2_base)
        model.train()