import torch
from steptaker import grad_calc

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

    perturbed_fake = init_fake + noise

    perturbed_fake = grad_calc(perturbed_fake, critic, create_graph=True)
    init_fake = grad_calc(init_fake, critic, create_graph=True)

    fdq = torch.norm((perturbed_fake - init_fake).flatten(1), dim=1) * noise_mags  # finite difference quotient

    return args.lambda_ppl * torch.mean(fdq)
