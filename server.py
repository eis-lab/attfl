import copy
import torch
from sklearn.metrics.pairwise import cosine_similarity

def Update_Global_Model(weights, grouped_info, idxs_users):
    idxs_users = list(idxs_users)
    w_avg = copy.deepcopy(weights[0])
    for key in w_avg.keys():
        if any(sub in key for sub in ['Local', 'SubGlobal', 'Global', 'conn']):
            w_avg[key] += sum(weights[i][key] for i in range(1, len(weights)))
            w_avg[key] = torch.div(w_avg[key], len(weights))

    global_weight = w_avg

    groups = []
    for loop_idx, sublist in enumerate(grouped_info):
        if len(sublist) != 0:
            sublist.append(idxs_users[loop_idx]) 
        new_group = set(sublist)
        
        for group in groups:
            if not new_group.isdisjoint(group):
                new_group |= group
                groups.remove(group)
        if new_group:
            groups.append(new_group)
            
    ungrouped_users = idxs_users
    for gorup in groups:
        ungrouped_users = list(set(ungrouped_users) - group)
    
    updated_local_weights = [[]] * len(idxs_users)
    
    if len(ungrouped_users) == len(idxs_users) or len(ungrouped_users) == 0:
        updated_local_weights = [global_weight] * (len(idxs_users))
        
        return updated_local_weights, global_weight
    
    else:
        groups.append(set(ungrouped_users))
        
        for group in groups:
            list_group = list(group)
            if len(list_group) != 1:
                w_base = copy.deepcopy(weights[idxs_users.index(list_group[0])])

                for key in w_avg.keys():
                    if any(sub in key for sub in ['Local', 'SubGlobal', 'Global', 'conn']):

                        for loop_idx, uid in enumerate(list_group):
                            if loop_idx != 0:
                                w_base[key] += weights[idxs_users.index(uid)][key]
                        w_base[key] = torch.div(w_base[key], len(list_group))

                for uid in list_group:
                    updated_local_weights[idxs_users.index(uid)] = w_base
                    
            elif len(list_group) == 1:
                updated_local_weights[idxs_users.index(list_group[0])] = global_weight

        return updated_local_weights, global_weight
                

def Update_Local_Models(weights, user_list, user_model_list):
    
    w_avg = copy.deepcopy(weights[0])
    for key in w_avg.keys():
        
        if any(sub in key for sub in ['Local', 'SubGlobal', 'Global', 'conn']):
            for loop_idx, user_idx in enumerate(user_list):
                user_model_list[user_idx].state_dict()[key] = weights[loop_idx][key]

    return user_model_list


def weight_groupping(idxs_users, local_saf, subglobal_saf, global_saf, participants):
    local_matrix, subglobal_matrix, global_matrix = [], [], []
    for idx in idxs_users:
        local_matrix.append(local_saf[idx].reshape(-1))
        subglobal_matrix.append(subglobal_saf[idx].reshape(-1))
        global_matrix.append(global_saf[idx].reshape(-1))

    local_cosine_matrix     = [1-cosine_similarity(local_matrix, local_matrix)][0]
    subglobal_cosine_matrix = [1-cosine_similarity(subglobal_matrix, subglobal_matrix)][0]
    global_cosine_matrix    = [1-cosine_similarity(global_matrix, global_matrix)][0]
    total_cosine_matrix     = (local_cosine_matrix + subglobal_cosine_matrix + global_cosine_matrix)/3
    
    total_grouped_info_uid     = [[] for i in range(0, participants)]
    total_grouped_info_sim     = [[] for i in range(0, participants)]
    
    search_idx = 1
    for participant_row_idx in range(0, participants):
        for participant_col_idx in range(search_idx, participants):
            
            if total_cosine_matrix[participant_row_idx][participant_col_idx] >= 0.5:
                total_grouped_info_uid[participant_row_idx].append(idxs_users[participant_col_idx])
                total_grouped_info_sim[participant_row_idx].append(total_cosine_matrix[participant_row_idx][participant_col_idx])
                
        uid_tmp = total_grouped_info_uid[participant_row_idx]
        sim_tmp = total_grouped_info_sim[participant_row_idx]
        
        search_idx += 1
    
    return total_grouped_info_uid