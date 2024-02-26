from __future__ import print_function
import ni_nfvo_client
import ni_mon_client
import re
from datetime import datetime, timedelta
from ni_nfvo_client.rest import ApiException as NfvoApiException
from ni_mon_client.rest import ApiException as NimonApiException
from config import cfg
from server.models.sfc_info import SFCInfo
from create_dashboard import create_dashboard

import random
import numpy as np
import time
import datetime as dt
from pprint import pprint
import subprocess
import csv

# Parameters
# OpenStack Parameters
openstack_network_id = "2d8bb3fc-fd89-49dc-84ea-5d304e372c87" # Insert OpenStack Network ID to be used for creating SFC
sample_user_data = "#cloud-config\n password: %s\n chpasswd: { expire: False }\n ssh_pwauth: True\n manage_etc_hosts: true\n runcmd:\n - sysctl -w net.ipv4.ip_forward=1"

#ni_nfvo_client_api
ni_nfvo_client_cfg = ni_nfvo_client.Configuration()
ni_nfvo_client_cfg.host=cfg["ni_nfvo"]["host"]
ni_nfvo_vnf_api = ni_nfvo_client.VnfApi(ni_nfvo_client.ApiClient(ni_nfvo_client_cfg))
ni_nfvo_sfc_api = ni_nfvo_client.SfcApi(ni_nfvo_client.ApiClient(ni_nfvo_client_cfg))
ni_nfvo_sfcr_api = ni_nfvo_client.SfcrApi(ni_nfvo_client.ApiClient(ni_nfvo_client_cfg))

#ni_monitoring_api
ni_mon_client_cfg = ni_mon_client.Configuration()
ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
ni_mon_api = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))

# <Important!!!!> parameters for Reinforcement Learning (Q-learning in this codes)
learning_rate = 0.01         # Learning rate
discount_factor = 0.98       # Discount factor
initial_epsilon = 0.10       # epsilon value of e-greedy algorithm
num_episode = 3000           # Number of iteration for Q-learning


def ssh_keygen(ip):
    ssh_command = "sshpass -p %s ssh -o stricthostkeychecking=no %s@%s "
    traffic_controller = (cfg["traffic_controller"]["password"], cfg["traffic_controller"]["username"], cfg["traffic_controller"]["ip"])

    inner_command =  "sudo ssh-keygen -f '/home/ubuntu/.ssh/known_hosts' -R %s"
    inner_command = (inner_command) % ip
    command = (ssh_command + inner_command) % traffic_controller     

    try:
        response = subprocess.check_output(command, shell=True).strip().decode("utf-8")
    except:
        print("error occured in ssh_keygen")
        return False
    return True


def setup_stress_for_test(ip):

    ssh_keygen(ip)
    
    ssh_command = "sshpass -p %s ssh -o stricthostkeychecking=no %s@%s "
    traffic_controller = (cfg["traffic_controller"]["password"], cfg["traffic_controller"]["username"], cfg["traffic_controller"]["ip"])    
    target_vnf = (cfg["instance"]["password"], cfg["traffic_controller"]["username"],ip)
    
    inner_command = "\"nohup stress-ng -c 0 -l 100 > /dev/null 2>&1 &\""
    inner_command = (ssh_command + inner_command) % target_vnf
    command = (ssh_command + inner_command) % traffic_controller

    limit = 10
    for i in range(0, limit):
        time.sleep(10)
        try:
            response = subprocess.Popen(command, shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
            print("passed")
            return
        except:
            print("waiting for server trun on")
        
        if limit == 9:
            print("stress-ng failed")
            return
            
    return 


def setup_env_for_test():

    response = "Cannot find test-sfcrs for test"
    deployed_sfcrs = ni_nfvo_sfcr_api.get_sfcrs()

    for sfcr in deployed_sfcrs:
        if sfcr.name.startswith("test-sfc"):
            if get_sfc_by_name(sfcr.name):
                continue
            else:
                print("building environment...")
                response = build_env_for_test(sfcr)  
        else:
            continue
            
    return response


def build_env_for_test(sfcr):

    #Test environment
    target_nodes_0 = ["ni-compute-181-155","ni-compute-181-158","ni-compute-181-156","ni-compute-181-155","ni-compute-181-158"]
    target_nodes_1 = ["ni-compute-181-155","ni-compute-181-158","ni-compute-181-156","ni-compute-181-203","ni-compute-181-156"]
    target_nodes_2 = ["ni-compute-181-203","ni-compute-181-156","ni-compute-181-158","ni-compute-181-155","ni-compute-181-156"]

    free_nodes_0 = [1, 1, 2, 2, 1]
    free_nodes_1 = [3, 2, 3, 4, 2]
    free_nodes_2 = [1, 3, 3, 4, 4]  

    target_nodes = [target_nodes_0, target_nodes_1, target_nodes_2]
    free_nodes = [free_nodes_0, free_nodes_1, free_nodes_2]

    mult = 5
    
    #Check name is 0, 1, or 2
    idx = int(re.search(r'\d+$', sfcr.name).group())
    
    target_node = target_nodes[idx]
    free_node = free_nodes[idx]
    
    target_name = sfcr.name + cfg["instance"]["prefix_splitter"]
    target_type = sfcr.nf_chain
    sfc_in_instance = []     

    for j in range(0, len(target_type)):
        type_instance = []
        for m in range(0, mult):
            print("{} {} {}".format(target_type[j], target_node[j], sfcr.name + cfg["instance"]["prefix_splitter"] + target_type[j] + cfg["instance"]["prefix_splitter"] +str(m) ))
            vnf_spec = set_vnf_spec(target_type[j], target_node[j], sfcr.name + cfg["instance"]["prefix_splitter"] + target_type[j] + cfg["instance"]["prefix_splitter"] +str(m) )
            vnf_id = deploy_vnf(vnf_spec)

            limit = 500 
            for i in range(0, limit):
                time.sleep(2)

                # Success to create VNF instance
                if check_active_instance(vnf_id):
                    type_instance.append(ni_mon_api.get_vnf_instance(vnf_id))
                    break
                elif i == (limit-1):
                    print("destroy vnf")
                    destroy_vnf(vnf_id)
                    
        sfc_in_instance.append(type_instance)


    for j in range(0, len(target_type)):
        for m in range(0, mult):
            if m == free_node[j]:
                continue
            else:
                #setup_stress_for_test(get_ip_from_id(sfc_in_instance[j][m]))
                setup_stress_for_test(get_ip_from_id(sfc_in_instance[j][m].id))
                
    
    AS_mydashboard_url = create_dashboard(sfc_in_instance,"SFC-VNF")
    
    return ("Target sfc : {} ML grafana dashboard : {}".format(sfcr.name, AS_mydashboard_url))




# get_sfcr_by_name(sfcr_name): get sfcr information by using sfcr_name from NFVO module
# Input: sfcr name
# Output: sfcr_info
def get_sfcr_by_name(sfcr_name):
#    print("9")
    query = ni_nfvo_sfcr_api.get_sfcrs()

    sfcr_info = [ sfcri for sfcri in query if sfcri.name == sfcr_name ]
    sfcr_info = sfcr_info[-1]

    return sfcr_info



# get_sfc_by_name(sfc_name): get sfc information by using sfc_name from NFVO module
# Input: sfc name
# Output: sfc_info
def get_sfc_by_name(sfc_name):
#    print("11")

    query = ni_nfvo_sfc_api.get_sfcs()

    sfc_info = [ sfci for sfci in query if sfci.sfc_name == sfc_name ]

    if len(sfc_info) == 0:
        return False

    sfc_info = sfc_info[-1]

    return sfc_info

def set_vnf_spec(vnf_type, node_name, vnf_name):
    vnf_spec = get_nfvo_vnf_spec2(vnf_type)
    vnf_spec.vnf_name = vnf_name
    vnf_spec.image_id = cfg["image"][vnf_type] #client or server
    vnf_spec.node_name = node_name

    return vnf_spec 


def get_nfvo_vnf_spec2(flavor_name):

    t = ni_nfvo_client.ApiClient(ni_nfvo_client_cfg)

    ni_nfvo_vnf_spec = ni_nfvo_client.VnfSpec(t)
    ni_nfvo_vnf_spec.flavor_id = cfg["flavor"][flavor_name]
    ni_nfvo_vnf_spec.user_data = sample_user_data % cfg["instance"]["password"]

    return ni_nfvo_vnf_spec



def deploy_vnf(vnf_spec):

    api_response = ni_nfvo_vnf_api.deploy_vnf(vnf_spec)
    print("check deployed")
    print(vnf_spec)
    print(api_response)

    return api_response

def destroy_vnf(id):

    api_response = ni_nfvo_vnf_api.destroy_vnf(id)

    return api_response


# check_active_instance(id): Check an instance whether it's status is ACTIVE
# Input: instance id
# Output: True or False
def check_active_instance(id):
    status = ni_mon_api.get_vnf_instance(id).status

    if status == "ACTIVE":
        return True
    else:
        return False



# get_ip_from_vm(vm_id):
# Input: vm instance id
# Output: port IP of the data plane
def get_ip_from_id(vm_id):

    query = ni_mon_api.get_vnf_instance(vm_id)

    ## Get ip address of specific network
    ports = query.ports
    network_id = openstack_network_id

    for port in ports:
        if port.network_id == network_id:
            return port.ip_addresses[-1]


##Why we should specify sfc_vnf??? It does not need at all!!..

# get_vnf_info(sfcr_name, sfc_vnfs): get each VNF instance ID and information from monitoring module
# Input: Prefix of VNF instance name, SFC order tuple [example] ("client", "firewall", "dpi", "ids", "proxy")
# Output: Dict. object = {'vnfi_info': vnfi information, 'num_vnf_type': number of each vnf type}
def get_vnf_info(sfcr_name):

    sfc_vnfs = get_sfcr_by_name(sfcr_name).nf_chain

    # Get information of VNF instances which are used for SFC
    query = ni_mon_api.get_vnf_instances()
    
    print("test1 : ", sfc_vnfs)
    #print("test : ", [ vnfi.name for vnfi in query for vnf_type in sfc_vnfs if vnfi.name.startswith(sfcr_name + cfg["instance"]["prefix_splitter"]) ])
    #selected_vnfi = [ vnfi for vnfi in query for vnf_type in sfc_vnfs if vnfi.name.startswith(sfcr_name + cfg["instance"]["prefix_splitter"]) ]
    
    print("test : ", [ vnfi.name for vnfi in query if vnfi.name.startswith(sfcr_name + cfg["instance"]["prefix_splitter"]) ])
    selected_vnfi = [ vnfi for vnfi in query if vnfi.name.startswith(sfcr_name + cfg["instance"]["prefix_splitter"]) ]    
    
    
    node_ids = [ vnfi.node_id for vnfi in selected_vnfi ]
    node_ids = list(set(node_ids))

    vnfi_list = []
    num_vnf_type = []
    temp = []

    # Sort VNF informations for creating states
    for vnf_type in sfc_vnfs:
        i =  sfc_vnfs.index(vnf_type)

        temp.append([])

        temp[i] = [ vnfi for vnfi in selected_vnfi if vnfi.name.startswith(sfcr_name + cfg["instance"]["prefix_splitter"] + vnf_type) ]
        temp[i].sort(key=lambda vnfi: vnfi.name)

        for vnfi in temp[i]:
            vnfi.node_id = node_ids.index(vnfi.node_id)

        vnfi_list = vnfi_list + temp[i]
        num_vnf_type.append(len(temp[i]))

    return {'vnfi_list': vnfi_list, 'num_vnf_type': num_vnf_type}



# get_vnf_resources(vnfi_list): get resources info. of VNF instance from the monitoring module
# Input: VNF instance list
# Output: Resource array -> [(CPU utilization, Memory utilization, Physical location), (...), ...]
def get_vnf_resources(vnfi_list):

    # In this codes, we regard CPU utilization, Memory utilization, Physicil node location
    resource_type = ("cpu_usage___value___gauge", "memory_free___value___gauge")

    # Create an initial resource table initialized by 0
    resources = np.zeros((len(vnfi_list), len(resource_type)+1))

    # Query to get resource data
    for vnfi in vnfi_list:
        i = vnfi_list.index(vnfi)
        for type in resource_type:
            j = resource_type.index(type)

            vnf_id = vnfi.id
            measurement_type = type
            end_time = dt.datetime.now()
            start_time = end_time - dt.timedelta(seconds = 10)

            if str(end_time)[-1]!='Z':
                 end_time = str(end_time.isoformat())+ 'Z'
            if str(start_time)[-1]!='Z':
                 start_time = str(start_time.isoformat()) + 'Z'

            response = ni_mon_api.get_measurement(vnf_id, measurement_type, start_time, end_time)
            #pprint(len(response))

            # Calculate CPU utilization as persent
            if j == 0:
                resources[i, j] = resources[i, j]

            # Calculate Memory utilization as percent
            elif j == 1:
                flavor_id = vnfi_list[i].flavor_id
                memory_query = ni_mon_api.get_vnf_flavor(flavor_id)
                memory_total = 1000000 * memory_query.ram_mb
                resources[i, j] = (resources[i, j]/memory_total)*100

        # Additionally, insert vnf location
        resources[i, -1] = vnfi.node_id

    return resources



# get_vnf_type(current_state, num_vnf_type): get vnf type showing vnf order of SFC
# Input: current state number, number of each vnf instance
# Output: vnf type (the order which is index number of vnf in sfc)
def get_vnf_type(current_state, num_vnf_type):

    index = len(num_vnf_type)
    pointer = num_vnf_type[0]

    for i in range (0, index):
        if current_state < pointer:
            return i
        else:
            pointer = pointer + num_vnf_type[i+1]



# get_action(current_state, Q, epsilon, pi_0): decide action from current state
# Input: current state, Q-value, epsilon for -greedy, action policy
# Output: action from current state
def get_action(current_state, Q, epsilon, pi_0):

    [state, action] = Q.shape

    # Decide action
    # Choose random action with probability
    if np.random.rand() < epsilon:
        next_action = np.random.choice(action, p=pi_0[current_state, :])
    # Choose the action maximizing Q-value
    else:
        next_action = np.nanargmax(Q[current_state, :])

    return next_action



# get_next_state(current_state, current_action, num_vnf_type): move from current state to next state after doing action
# Input: currrent_state, current_action, num_vnf_type
# Output: next_state (if negative value, it means no next state)
def get_next_state(current_state, current_action, num_vnf_type):

    current_vnf_type = get_vnf_type(current_state, num_vnf_type)
    last_vnf_type = len(num_vnf_type) - 1

    next_state = 0

    for i in range (0, current_vnf_type+1):
        next_state = next_state + num_vnf_type[i]

    next_state = next_state + current_action

    return next_state


def create_sfc(sfcr, instance_id_list):

    sfc_spec =ni_nfvo_client.SfcSpec(sfc_name=sfcr.name,
                                 sfcr_ids=[sfcr.id],
                                 vnf_instance_ids=instance_id_list,
                                 is_symmetric=False)


    api_response = ni_nfvo_sfc_api.set_sfc(sfc_spec)
    print("Success to pass for creating sfc")
    return api_response



# set_action_policy(theta): define initial action policy
# Input: (number of states, number of actions, number of each vnf instance)
# Output: initial action policy
def set_action_policy(theta):

    [m, n] = theta.shape
    pi = np.zeros((m, n))

    for i in range(0, m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])

    pi = np.nan_to_num(pi)  # Change nan to 0

    return pi



# set_initial_policy(vnfi_info): create initial policy array
# Input: vnfi_info
# Output: initial policy array
def set_initial_policy(vnfi_info):

    # Number of each VNF type
    # Count states and actions for RL (exclude final states)
    num_vnf_type = vnfi_info["num_vnf_type"]
    num_state = len(vnfi_info["vnfi_list"]) - num_vnf_type[-1]
    num_action = max(num_vnf_type)

    policy = np.zeros((num_state, num_action))

    final_type = len(num_vnf_type)-1

    for i in range (0, num_state):
        nan_list = []

        vnf_type = get_vnf_type(i, num_vnf_type)

        # Is it final states?
        if vnf_type == final_type:
            for j in range(0, num_action):
                nan_list.append(np.nan)

        else:
            for j in range(0, num_action):
                if j < num_vnf_type[vnf_type+1]:
                    nan_list.append(1)
                else:
                    nan_list.append(np.nan)

        policy[i] = nan_list

    return policy



# is_final_state(state, num_vnf_type): check whether input state is final state or not
# Input: currrent_state, num_vnf_type
# Output: true or false
def is_final_state(state, num_vnf_type):

    vnf_type = get_vnf_type(state, num_vnf_type)
    last_vnf_type = len(num_vnf_type) - 1

    if vnf_type == last_vnf_type:
        return True
    else:
        return False




# Q_learning(current_state, current_action, r, next_state, Q, eta, gamma): Q-leraning algorithm to updeate Q-value
# Input: current_state, current_action, rewords, next_state, Q-value, Discount factor
# Output: updated Q-value
def Q_learning(current_state, current_action, r, next_state, Q, eta, gamma):

    if next_state == -1: # Arriving final state
        Q[current_state, current_action] = Q[current_state, current_action] + eta * (r)
    else:
        Q[current_state, current_action] = Q[current_state, current_action] + eta * (r + gamma * np.nanmax(Q[next_state,: ]) - Q[current_state, current_action])

    return Q



# sfc_path_selection(Q, epsilon, eta, gamma, pi, resources, vnfi_list, num_vnf_type): decide sfc path by Q-learning
# Input: Q-value, epsilon of -greedy , learning rate, discount factor, action policy, resources, vnf instance info, number of each vnf type
# Output: history, Q-value
def sfc_path_selection(Q, epsilon, eta, gamma, pi, resources, vnfi_list, num_vnf_type):

    current_state = 0  # Starting state
    s_a_history = [[0, np.nan]]  # Define list to track history of (agent action, state)

    while (1):  # Unitl deciding SFC path
        current_action = get_action(current_state, Q, epsilon, pi)  # Choosing an action

        s_a_history[-1][1] = current_action  # Adding current state (-1 because no entry in the list now)

        next_state = get_next_state(current_state, current_action, num_vnf_type) # Get next state

        # Reward calculatoin
        ## CPU Utilization
        if resources[next_state, 0] < 1:
            r_cpu = 1
        else:
            r_cpu = 1/(100 * resources[next_state, 0])

        ## Memory Utilization
        if resources[next_state, 1] < 1:
            r_memory = 1
        else:
            r_memory = resources[next_state, 1]/100

        ## VNF location
        if resources[current_state, 2] == resources[next_state, 2]: # if exist in the same physical node
            r_location = 1
        else:
            r_location = 0

        ## Give different weights whether it is CPU intensive of Memory intensive
        vnf_type = get_vnf_type(next_state, num_vnf_type)
        vnfi_name = vnfi_list[next_state].name

        if "firewall" in vnfi_name or "flowmonitor" in vnfi_name or "proxy" in vnfi_name:
            r = (0.35*r_cpu) + (0.15*r_memory) + (0.5*r_location) ## CPU intensive (weights: CPU 0.35, Memory 0.15, Location 0.5)
        elif "dpi" in vnfi_name or "ids" in vnfi_name:
            r = (0.15*r_cpu) + (0.35*r_memory) + (0.5*r_location) ## Memory intensive (weights: CPU 0.15, Memory 0.35, location 0.5)
        else:
            r = (0.40*r_cpu) + (0.30*r_memory) + (0.30*r_location) ## Others

        s_a_history.append([next_state, np.nan]) # Adding next state into the history

        # Update Q-value
        if is_final_state(next_state, num_vnf_type):
            Q = Q_learning(current_state, current_action, r, -1, Q, eta, gamma)
        else:
            Q = Q_learning(current_state, current_action, r, next_state, Q, eta, gamma)

        # Check wheter final state or not
        if is_final_state(next_state, num_vnf_type):
            break
        else:
            current_state = next_state

    return [s_a_history, Q]










# q_based_sfc(sfcr_name, sfc_vnfs, sfc_name): create sfc by using Q-leraning
# Input: JSON sfc_info (flowclassifier name, sfc vnfs, sfc name)
# Output: flow classifier id, sfc id
def q_based_sfc(sfc_info):

    eta = learning_rate       # Learning rate
    gamma = discount_factor     # Discount factor
    epsilon = initial_epsilon   # epsilon value of -greedy algorithm
    episode = num_episode   # Number of iteration for Q-learning

    ## Step 1: Get VNF instance Info
    vnfi_info = get_vnf_info(sfc_info.sfcr_name)

    vnfi_list = vnfi_info["vnfi_list"]
    num_vnf_type = vnfi_info["num_vnf_type"]

    vnf_resources = get_vnf_resources(vnfi_list)

    ## Step 2: initialize Q-value and action policy
    pi_0 = set_initial_policy(vnfi_info)
    pi_0 = set_action_policy(pi_0)
    Q = pi_0 * 0

    ## Step 3: Q-Leraning
    for i in range(0, episode):

        # Decrese epsilon value
        epsilon = 1./((i / 100) + 1)

        # Finding SFC path
        [s_a_history, Q] = sfc_path_selection(Q, epsilon, eta, gamma, pi_0, vnf_resources, vnfi_list, num_vnf_type)

    ## Step 4: Print final sfc path decision regarding to the latest Q-value
    sfc_path = []
    instance_id_list = []
    epsilon = 0

    [s_a_history, Q] = sfc_path_selection(Q, epsilon, eta, gamma, pi_0, vnf_resources, vnfi_list, num_vnf_type)

    for i in range(0, len(s_a_history)):
        sfc_path.append(vnfi_list[s_a_history[i][0]].name)
        instance_id_list.append([vnfi_list[s_a_history[i][0]].id])

    pprint(sfc_path)

    #Find sfcr_id using sfcr_name
    sfcr = get_sfcr_by_name(sfc_info.sfcr_name)
    sfc_id = create_sfc(sfcr, instance_id_list)
    #sfc_id = set_sfc(sfcr_id, sfc_info.sfc_name, sfc_path, vnfi_list)

    response = { "sfcr_id": sfcr.id,
                 "sfc_id": sfc_id,
                 "sfc_path": sfc_path }

    return response


# random_sfc(sfcr_name, sfc_vnfs, sfc_name):
# Input: JSON sfc_info (flowclassifier name, sfc vnfs, sfc name)
# Output: flow classifier id, sfc id
def random_sfc(sfc_info):

    ## Step 1: Get VNF instance Info
    vnfi_info = get_vnf_info(sfc_info.sfcr_name)

    vnfi_list = vnfi_info["vnfi_list"]
    num_vnf_type = vnfi_info["num_vnf_type"]

    # Random sfc path selection
    start = 0
    end = 0
    sfc_path = []
    instance_id_list = []

    for i in range(0, len(num_vnf_type)):
        end = start + num_vnf_type[i]
        sfc_path.append(random.choice(vnfi_list[start:end]).name)
        instance_id_list.append([random.choice(vnfi_list[start:end]).id])
        start = start + num_vnf_type[i]

    pprint(sfc_path)



    sfcr = get_sfcr_by_name(sfc_info.sfcr_name)
    sfc_id = create_sfc(sfcr, instance_id_list)
    #sfc_id = set_sfc(sfcr_id, sfc_info.sfc_name, sfc_path, vnfi_list)
    

    response = { "sfcr_id": sfcr.id,
                 "sfc_id": sfc_id,
                 "sfc_path": sfc_path }

    return response




