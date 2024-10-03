import pandas as pd
import os, argparse

search_graph = {}
search_space = {
    'learning_rate': [1e-5, 3e-5, 9e-5, 27e-5],
    'num_steps': [512, 2048, 8192],
    'batch_size': [8, 16],
}

start_params = []

def add_config(node, searched_set = set(), recursive_iter = 1):
    print(f"|{'------'*recursive_iter} searching {node}")
    all_configs = []
    searched_set.add(node)
    for value in search_space[node]:
        node_config = {node: value}
        
        new_configs = [node_config]
        if (node, value) in search_graph:
            for next_node in search_graph[node, value]:
                next_configs, new_searched_set = add_config(next_node, searched_set, recursive_iter + 1)
    
                searched_set = searched_set.union(new_searched_set)
                
                temporary_configs = []
                for next_config in next_configs:
                    for temporary_config in new_configs:
                        temporary_configs.append({**temporary_config, **next_config})
                
                new_configs = temporary_configs
                
            del search_graph[(node, value)]
                
        else:
            new_configs = [node_config]
    
            
        all_configs.extend(new_configs) 
    print(f"|{'------'*recursive_iter} new configs number {len(all_configs)}")
    return all_configs, searched_set

configs, searched_set = [dict()], set()
for node in start_params:
    print("--------"*3)
    new_configs, new_searched_set = add_config(node)
    
    if configs:
        temporary_configs = []
        for new_config in new_configs:

            for config in configs:
                temporary_configs.append({**config, **new_config})
                
        configs = temporary_configs
    else:
        configs = new_configs
            
    searched_set = new_searched_set.union(searched_set)
print("--------"*3)
print(f'| new configs number {len(configs)}')
    
for param in searched_set:
    del search_space[param]
    
if search_graph: 
    print('Unsearched parameters: ', search_graph.keys())
    raise ValueError("Some parameters are not searched!")

for param in search_space:
    temporary_configs = []
    print("--------"*3)
    print(f"| searching {param}")
    for value in search_space[param]:
        for config in configs:
            temporary_configs.append({**config, **{param: value}})
    configs = temporary_configs
            
    print(f"| new configs number {len(temporary_configs)}")
    
print(f"Configs number {len(configs)}")
            
final_configs = []
# check constraints
for config in configs:
    # if config['select_states_dim'] < config['updatable_states_dim']: continue
    # if config['select_channels_dim'] < config['updatable_channels_dim']: continue
    # if config['ssm_method'] != 'ours' and config['lp_method'] == 'ours': continue
    # if config['ssm_method'] == 'lora' and config['lp_method'] != 'lora': continue
    # if config['ssm_method'] == 'lora' and config['lp_method'] == 'lora': 
    #     if config['ssm_lora_rank'] != config['lp_lora_rank']: continue
    final_configs.append(config)
    
print("Remove redundant configs, config number ", len(final_configs))

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    # head: true or false
    parser.add_argument('--head', type=int, default=0, choices = [0,1])
    args = parser.parse_args()
    
    new_configs = pd.DataFrame(final_configs)
    if args.head: new_configs = new_configs.head(1)

    if (not args.head) and os.path.exists('tabpfn_configs.csv'):
        user_input = input("Old configs detected. Do you want to directly overwrite the original tabpfn configs? (y/n): ")
        if user_input.lower() in ['y', 'yes']:
            new_configs.to_csv('tabpfn_configs.csv', index=False, header=True)
            pd.DataFrame(new_configs.index).to_csv('tabpfn_configs_index.csv', index=False, header=False)
            exit(0)
        
        origin_configs = pd.read_csv('tabpfn_configs.csv', header=0)
        
        # Find additional configs
        additional_configs = new_configs.merge(origin_configs, how='left', indicator=True).loc[lambda x : x['_merge']=='left_only'].drop('_merge', axis=1).reset_index(drop=True)
        additional_configs.to_csv('additional_tabpfn_configs.csv', index=False, header=True)
        pd.DataFrame(additional_configs.index).to_csv('additional_tabpfn_configs_index.csv', index=False, header=False)
        
        while 1:
            print('Get additional configs.')
            # Ask the user if they want to overwrite the original configs
            user_input = input("Do you want to overwrite the original tabpfn configs? (y/n): ")
            if user_input.lower() in ['y', 'yes']:
                new_configs.to_csv('tabpfn_configs.csv', index=False, header=True)
                pd.DataFrame(new_configs.index).to_csv('tabpfn_configs_index.csv', index=False, header=False)
                print('Data overwritten!')
                break
            elif user_input.lower() in ['n', 'no']:
                print('Sure!')
                break
            else:
                print("Invalid input!")
    else:
        new_configs.to_csv('tabpfn_configs.csv', index=False, header=True)
        pd.DataFrame(new_configs.index).to_csv('tabpfn_configs_index.csv', index=False, header=False)