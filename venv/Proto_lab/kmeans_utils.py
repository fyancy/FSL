import torch


def compute_logits(cluster_centers, data):
    """Computes the logits of being in one cluster, squared Euclidean.
    Args:
    cluster_centers: [B, K, D] Cluster center representation.
    data: [B, N, D] Data representation.
    Returns:
    log_prob: [B, N, K] logits.
    """
    # cluster_centers = cluster_centers.unsqueeze(dim=0)  # [1, K, D]
    cluster_centers = cluster_centers.unsqueeze(dim=1)  # [B, 1, K, D]
    # data = data.unsqueeze(dim=1)  # [N, 1, D]
    data = data.unsqueeze(dim=2)  # [B, N, 1, D]
    neg_dist = -torch.sum(torch.pow(data - cluster_centers, 2), dim=-1)  # [B, N, K]
    # neg_dist = -torch.mean(torch.pow(data - cluster_centers, 2), dim=-1)  # [N, K] 使用mean效果差
    return neg_dist


def assign_cluster(cluster_centers, data):
    """Assigns data to cluster center, using K-Means.return the probability.
  Args:
    cluster_centers: [B, K, D] Cluster center representation.
    data: [B, N, D] Data representation.
  Returns:
    prob: [B, N, K] Soft assignment.
  """
    n_data = data.shape[1]
    # data = data.contiguous().view(-1, data.shape[-1])  # [B*N, D]
    logits = compute_logits(cluster_centers, data)  # [B, N, K]
    # bsize = logits.shape[0]
    # ncluster = logits.shape[2]
    # logits = logits.view(-1, ncluster)
    prob = torch.nn.functional.softmax(logits, dim=-1)
    # prob = prob.view(bsize, n_data, -1)  # [B, N, K]
    return prob


def update_cluster(data, prob):
    """Updates cluster center based on assignment, standard K-Means.
  Args:
    data: [B, N, D]. Data representation.
    prob: [B, N, K]. Cluster assignment soft probability.
  Returns:
    cluster_centers: [B, K, D]. Cluster center representation.
  """
    # Normalize accross N.
    # data = data.view(-1, data.shape[-1])  # [B*N, D]
    # prob = prob.view(-1, prob.shape[-1])  # [B*N, K]
    prob_sum = torch.sum(prob, dim=1, keepdim=True)  # [B, 1, K]
    # prob_sum = torch.sum(prob, dim=0, keepdim=True)  # [1, K]
    prob_sum += torch.eq(prob_sum, 0.0).float()
    prob2 = prob / prob_sum  # [B, N, K]/[B, 1, K] ==> [B, N, K]  broadcast mechanism

    # cluster_centers = torch.sum(data.unsqueeze(dim=1) * prob2.unsqueeze(dim=2), dim=0)
    # [B, N, 1, D]*[B, N, K, 1]==>[B, N, K, D]==>[B, K, D]
    # print('---------prob2---------\n', prob2.cpu().detach().numpy())
    # cluster_centers = torch.sum(data.unsqueeze(dim=2) * prob2.unsqueeze(dim=3), dim=1)
    b = prob.shape[0]
    k = prob.shape[-1]
    d = data.shape[-1]
    cluster_centers = data.unsqueeze(dim=2) * prob2.unsqueeze(dim=3)
    cluster_centers = cluster_centers.view(-1, k, d)  # [B*N, K, D]
    cluster_centers = torch.sum(cluster_centers, dim=0, keepdim=True).expand(b, k, d)
    # print('---------cluster---------\n', cluster_centers.cpu().detach().numpy())

    return cluster_centers
