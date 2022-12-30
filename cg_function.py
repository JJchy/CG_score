import numpy as np
import torch

from collections import defaultdict
from functools import reduce

def calc_cg_score(
    _dataset, device, *args, 
    rep_num=1, unbalance_ratio=1, 
    sub_term=False
):
"""
Calculate CG-score and its sub-terms.

Args:
    _dataset: torch.utils.data.Dataset
        Dataset to calculate CG-score.
    device: torch.device
        Device to calculate CG-score.
    rep_num: int
        Number of repetitions.
    unbalance_ratio: float
        Ratio of unbalanced data. (1:unbalance_ratio)
    sub_term: bool
        If True, return sub-terms of CG-score.
    
Returns:
    vi_base: np.array or dict
        If sub_term is true, return CG-score only
        Otherwise, return CG-score and its sub-terms.
"""
    vi_base = { # a = first term of Equation 6, b = second term of Equation 6
        "vi": np.zeros((len(_dataset))),
        "ab": np.zeros((len(_dataset))), 
        "a2": np.zeros((len(_dataset))),
        "b2": np.zeros((len(_dataset))),
        "times": np.zeros((len(_dataset)))
    }

    with torch.no_grad():
        # Repeting calculation
        for _ in range(rep_num):
            dataset = defaultdict(list)
            data_idx = defaultdict(list)

            # Load and normalize data
            for j in range(len(_dataset)):
                data, label = _dataset[j]
                data_unnormed = torch.flatten(data).unsqueeze(0).type(torch.DoubleTensor)
                data_normed = data_unnormed/torch.norm(data_unnormed)
                label = label.item() if torch.is_tensor(label) else label

                dataset[label].append(torch.flatten(data_normed).unsqueeze(0))
                data_idx[label].append(j)

            new_dataset = {}
            for key, data_list in dataset.items():
                new_dataset[key] = np.array(data_list, dtype=object)
                data_idx[key] = np.array(data_idx[key])
            dataset = new_dataset

            # Calculate CG-score in each class
            for curr_label, data_list in dataset.items():
                curr_num = len(data_list)

                chosen_curr_idx = np.random.choice(range(len(data_list)), curr_num, replace=False)
                chosen_curr_list = data_list[chosen_curr_idx]

                # Sub-sample another class examples
                another_labels = [label for label in dataset if label != curr_label]
                another_list = reduce(
                    lambda acc, idx: np.concatenate((acc, dataset[idx])), another_labels, np.array([]))
                another_num = min(int(curr_num * unbalance_ratio), len(another_list))

                chosen_another_list = another_list[
                    np.random.choice(range(len(another_list)), another_num, replace=False)]

                # Make gram matrix H^\infty
                a = torch.cat(
                        list(np.concatenate((chosen_curr_list, chosen_another_list))), 0
                    ).type(torch.DoubleTensor).to(device)
                y = torch.Tensor(
                        [1 for _ in range(curr_num)] + [-1 for _ in range(another_num)]
                    ).type(torch.DoubleTensor).to(device)

                H_inner = torch.matmul(a, a.transpose(0, 1))
                del a
                H = H_inner*(np.pi-torch.acos(H_inner))/(2*np.pi)
                del H_inner
                H.fill_diagonal_(1/2)

                invH = torch.inverse(H)
                del H

                original_error = y@(invH@y)

                vi_class = defaultdict(list)

                # Calculate CG-score at each example
                for k in range(curr_num):
                    A_with_row = torch.cat((invH[:, :k], invH[:, (k+1):]), axis=1)
                    A = torch.cat((A_with_row[:k, :], A_with_row[(k+1):, :]), axis=0)
                    B = A_with_row[k, :].unsqueeze(0)
                    del A_with_row
                    D = invH[k, k]

                    invH_mi = A - (B.T@B)/D
                    y_mi = torch.cat((y[:k], y[(k+1):]))

                    vi_class['vi'].append((original_error - y_mi@(invH_mi@y_mi)).item())
                    y_i = y[k]
                    if sub_term:
                        vi_class['ab'].append((y_i*B@y_mi).item())
                        vi_class['a2'].append((((B@y_mi)**2)/D).item())
                        vi_class['b2'].append((((y_i)**2)*D).item())

                    del A, B, invH_mi, y_mi

                for keys, values in vi_class.items():
                    vi_base[keys][data_idx[curr_label][chosen_curr_idx]] += np.array(values) \
                                                    if (keys == "vi" or sub_term) else 0

                vi_base["times"][data_idx[curr_label][chosen_curr_idx]] += 1

                del invH
            
                torch.cuda.empty_cache()

    vi = {
        "vi": np.zeros((len(_dataset))),
        "ab": np.zeros((len(_dataset))),
        "a2": np.zeros((len(_dataset))),
        "b2": np.zeros((len(_dataset))),
    }
    for keys, values in vi_base.items():
        if keys == "times":
            continue
        vi[keys] = values / np.where(vi_base["times"] > 0, vi_base["times"], 1)

    return vi if sub_term else vi["vi"]
