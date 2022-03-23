import torch
from steptaker import steptaker


def fdq_rejection(fake, critic_list, steps, args):
    """Flows fake and a perturbed fake forward under TTC, and computes
    the finite difference quotient. Discards some portion of the batch based on
    sorting these finite difference quotients/
    Inputs
    - fake; minibatch of noise
    - critic_list; list of trained critics
    - steps; steps for TTC
    - args; args to ttc_eval.py. The important ones here are epsilon (magnitude of noise)
    keep_rate (percentage of images to keep), and crop_size (resolution of centre crop for fdq comp)
    Outputs
    - fake; the first bs * keep_rate images of fake, sorted according to the fdq
    """
    midway = fake.size(3) // 2
    lower = midway - args.crop_size // 2
    upper = midway + args.crop_size // 2

    init_fake = fake.detach().clone()

    for b_idx in range(2):
        if b_idx > 0:
            noise = args.epsilon * torch.randn_like(fake)
            noise_mags = torch.norm(noise[:, :, lower:upper, lower:upper].flatten(1), dim=1) ** (-1)
            noise_mags = noise_mags.cuda()
            fake = (init_fake + noise).detach().clone()

        for i in len(critic_list):  # apply the steps of TTC
            eps = torch.tensor(steps[i]).cuda()
            fake = steptaker(fake, critic_list[i], eps)

        if b_idx == 0:
            base_point = fake.detach().clone()
            cropped_base = base_point[:, :, lower:upper, lower:upper]
        else:
            cropped_fake = fake[:, :, lower:upper, lower:upper]
            finite_differences = torch.norm((cropped_base - cropped_fake).flatten(1), dim=1) * noise_mags

    order = torch.argsort(finite_differences)
    order = order[:int(args.bs * args.keep_rate)]

    return init_fake[order, :, :, :]