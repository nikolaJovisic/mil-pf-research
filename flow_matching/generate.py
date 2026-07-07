import torch
from setflow import SetFlow

mu_it1, std_it1 = 47.6014, 28.1936
mu_it0, std_it0 = 2.3055, 0.6760

def generate(
    model,
    num_bags_y0=7856,
    num_bags_y1=439,
    feature_dim=1152,
    num_steps=200,
    device="cpu",
):
    model.eval()

    B = num_bags_y0 + num_bags_y1

    time_steps = torch.linspace(0.0, 1.0, num_steps, device=device)

    global_bag_ids = torch.arange(B, device=device)
    y_bag = (global_bag_ids >= num_bags_y0).long()

    n_it1 = torch.clamp(
        torch.normal(mu_it1, std_it1, size=(B,), device=device),
        min=10
    ).long()

    n_it0 = torch.clamp(
        torch.normal(mu_it0, std_it0, size=(B,), device=device),
        min=2
    ).long()

    bag_sizes = n_it0 + n_it1
    n_total = bag_sizes.sum().item()

    instance_type = torch.cat([
        torch.cat([
            torch.zeros(n0, dtype=torch.long),
            torch.ones(n1, dtype=torch.long),
        ])
        for n0, n1 in zip(n_it0.cpu(), n_it1.cpu())
    ]).to(device)

    group = torch.repeat_interleave(
        torch.arange(B, device=device),
        bag_sizes
    )

    x_t = torch.randn(n_total, feature_dim, device=device)

    y_inst = torch.repeat_interleave(y_bag, bag_sizes)

    with torch.no_grad():
        for i in range(len(time_steps) - 1):
            x_t = model.step(
                x_t=x_t,
                t_start=time_steps[i],
                t_end=time_steps[i + 1],
                y=y_inst.unsqueeze(1),
                group=group,
                instance_type=instance_type,
            )

    return x_t.cpu(), y_inst.cpu(), group.cpu(), instance_type.cpu()
