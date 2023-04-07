import torch
import torch.nn.functional as F



# Loss function and optimizer
def entropy(p):
    """ We compute the entropy """
    if (len(p.size())) == 2:
        return - torch.sum(p * torch.log(p + 1e-18)) / float(len(p))
    elif (len(p.size())) == 1:
        return - torch.sum(p * torch.log(p + 1e-18))
    else:
        raise NotImplementedError


def Compute_entropy(net, x):
    """ We compute the conditional entropy H(Y|X) and the entropy H(Y) """
    p = F.softmax(net(x), dim=1)
    p_ave = torch.sum(p, dim=0) / len(x)
    return entropy(p), entropy(p_ave)



def entropy_2d(p, dim=0, keepdim=False):
    """ 
    We compute the entropy along the first dimension, for each value of the tensor
    :param p: tensor of probabilities
    :param dim: dimension along which we want to compute entropy (the sum across this dimension must be equal to 1)
    """
    entrop = - torch.sum(p * torch.log(p + 1e-18), dim=dim, keepdim=keepdim)
    return entrop


def compute_aver_entropy_2d(prob_input, entropy_dim=1, aver_entropy_dim=[1, 2]):
    tot_entropy = entropy_2d(prob_input, dim=entropy_dim)
    aver_entropy = torch.mean(tot_entropy, dim=aver_entropy_dim)
    return aver_entropy


def compute_entropy_aver_2d(prob_input, p_ave_dim=[2, 3], entropy_aver_dim=1):
    p_ave = torch.mean(prob_input, dim=p_ave_dim)
    entropy_aver = entropy_2d(p_ave, dim=entropy_aver_dim)
    return entropy_aver




def JSD(prob_dists, alpha=0.5, p_ave_dim=0, entropy_aver_dim=0, entropy_dim=1, aver_entropy_dim=0):
    """
    JS divergence JSD(p1, .., pn) = H(sum_i_to_n [w_i * p_i]) - sum_i_to_n [w_i * H(p_i)], where w_i is the weight given to each probability
                                  = Entropy of average prob. - Average of entropy

    :param prob_dists: probability tensors (shape: B, C, H, W)
    :param alpha: weight on terms of the JSD
    """
    entropy_mean = compute_entropy_aver_2d(prob_dists, p_ave_dim=p_ave_dim, entropy_aver_dim=entropy_aver_dim)
    mean_entropy = compute_aver_entropy_2d(prob_dists, entropy_dim=entropy_dim, aver_entropy_dim=aver_entropy_dim)
    jsd = alpha * entropy_mean - (1 - alpha) * mean_entropy
    
    return jsd