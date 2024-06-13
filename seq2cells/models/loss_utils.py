import torch


def log_minus_exp(a, b, epsilon=1.e-6):
    return a + torch.log1p(-torch.exp(b - a) + epsilon)


def get_logits_from_logistic_pars(loc, log_scale, num_classes = 10):
    loc = loc.unsqueeze(-1)
    log_scale = log_scale.unsqueeze(-1)

    inv_scale = (-log_scale + 2.0).exp()
    
    bin_width = 2.0 / (num_classes - 1)
    bin_centers = torch.linspace(-1.0, 1.0, num_classes).to(loc.device)
    for dim in range(bin_centers.ndim - 1):
        bin_centers = torch.unsqueeze(bin_centers, dim=0)
    bin_centers = bin_centers - loc
    
    # equivalent implementation
    # log_cdf_min = -torch.log1p((-inv_scale * (bin_centers - 0.5 * bin_width)).exp())
    # log_cdf_plus = -torch.log1p((-inv_scale * (bin_centers + 0.5 * bin_width)).exp())
    log_cdf_min = torch.nn.LogSigmoid()(inv_scale * (bin_centers - 0.5 * bin_width))
    log_cdf_plus = torch.nn.LogSigmoid()(inv_scale * (bin_centers + 0.5 * bin_width))

    logits = log_minus_exp(log_cdf_plus, log_cdf_min)
    return logits


if __name__ == "__main__":
    loc = torch.randn(5, 1024).clip(-1, 1)
    log_scale = torch.randn(5, 1024)

    num_classes = 2
    logits = get_logits_from_logistic_pars(loc, log_scale, num_classes)
    print(logits.shape)