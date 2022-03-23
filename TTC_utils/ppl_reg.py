import torch
from steptaker import steptaker

def ppl_reg(init_fake, critic, args):
    """Computes ppl regularization
    Inputs
    - init_fake; prepared batch of images with steps of TTC applied
    - critic; current critic
    - args; args to TTC, including parameters for this reg.
    Outputs
    - Returns norm of finite difference quotient of critic
    """
    eps = torch.ones([1])
    if torch.cuda.is_available():
        eps = eps.cuda()

    noise = args.epsilon * torch.randn_like(init_fake)
    noise_mags = torch.norm(noise.flatten(1), dim=1) ** (-1)

    perturbed_fake = (init_fake + noise).detach().clone()

    perturbed_fake = steptaker(perturbed_fake, critic, eps)
    init_fake = steptaker(init_fake, critic, eps)

    fdq = torch.norm((perturbed_fake - init_fake).flatten(1), dim=1) * noise_mags  # finite difference quotient

    return args.lambda_ppl * torch.mean(fdq)
