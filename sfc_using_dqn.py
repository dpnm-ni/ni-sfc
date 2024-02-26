import ni_mon_client, ni_nfvo_client
from ni_mon_client.rest import ApiException
from ni_nfvo_client.rest import ApiException
from datetime import datetime, timedelta
from config import cfg
from torch_dqn import *

import numpy as np

import datetime as dt
import math
import os
import time
import subprocess
from pprint import pprint
import random
import json

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


# <Important!!!!> parameters for Reinforcement Learning (DQN in this codes)
learning_rate = 0.01            # Learning rate
gamma         = 0.98            # Discount factor
buffer_limit  = 10000           # Maximum Buffer size
batch_size    = 16              # Batch size for mini-batch sampling
num_neurons = 64               # Number of neurons in each hidden layer
epsilon = 0.99                  # epsilon value of e-greedy algorithm
required_mem_size = 24        # Minimum number triggering sampling
print_interval = 24             # Number of iteration to print result during DQN

# Global values

sfc_update_flag = True
training_list = []

# get_all_flavors(): get all flavors information
# Input: null
# Output: flavors information
def get_all_flavors():
    query = ni_mon_api.get_vnf_flavors()

    return query


# destroy_vnf(id): destory VNF instance in OpenStack environment
# Inpurt: ID of VNF instance
# Output: API response
def destroy_vnf(id):

    api_response = ni_nfvo_vnf_api.destroy_vnf(id)

    return api_response


# get_vnf_info(sfcr_name, sfc_vnfs): get each VNF instance information from monitoring module
# Input: Prefix of VNF instance name, SFC order tuple or list [example] ("client", "firewall", "dpi", "ids", "proxy")
# Output: VNF information list
def get_vnf_info(sfcr_name):
    query = ni_mon_api.get_vnf_instances()

    sfc_vnfs = get_sfcr_by_name(sfcr_name).nf_chain

    selected_vnfi = [ vnfi for vnfi in query if vnfi.name.startswith(sfcr_name + cfg["instance"]["prefix_splitter"]) ]

    vnfi_list = []

    # Sort VNF informations for creating states
    for vnf_type in sfc_vnfs:
        i =  sfc_vnfs.index(vnf_type)

        vnfi_list.append([])

        vnfi_list[i] = [ vnfi for vnfi in selected_vnfi if vnfi.name.startswith(sfcr_name + cfg["instance"]["prefix_splitter"] + vnf_type) ]
        vnfi_list[i].sort(key=lambda vnfi: vnfi.name)

    return vnfi_list


# get_sfcr_by_name(sfcr_name): get sfcr information by using sfcr_name from NFVO module
# Input: sfcr name
# Output: sfcr_info
def get_sfcr_by_name(sfcr_name):
#    print("9")
    query = ni_nfvo_sfcr_api.get_sfcrs()

    sfcr_info = [ sfcri for sfcri in query if sfcri.name == sfcr_name ]
    sfcr_info = sfcr_info[-1]

    return sfcr_info



# get_specific_vnf_info(sfcr_name, id): get specific VNF instance information from monitoring module
# Input: VNF instance ID
# Output: VNF information
def get_specific_vnf_info(id):
    query = ni_mon_api.get_vnf_instance(id)

    return query


# set_flow_classifier(sfcr_name, sfc_ip_prefix, nf_chain, source_client): create flow classifier in the testbed
# Input: flowclassifier name, flowclassifier ip prefix, list[list[each vnf id]], flowclassifier VM ID
# Output: response
def set_flow_classifier(sfcr_name, src_ip_prefix, nf_chain, source_client):
    sfcr_spec = ni_nfvo_client.SfcrSpec(name=sfcr_name,
                                 src_ip_prefix=src_ip_prefix,
                                 nf_chain=nf_chain,
                                 source_client=source_client)

    api_response = ni_nfvo_sfcr_api.add_sfcr(sfcr_spec)

    return api_response


def get_sfc_by_name(sfc_name):
#    print("11")

    query = ni_nfvo_sfc_api.get_sfcs()

    sfc_info = [ sfci for sfci in query if sfci.sfc_name == sfc_name ]

    if len(sfc_info) == 0:
        return False

    sfc_info = sfc_info[-1]

    return sfc_info


# set_sfc(sfcr_id, sfc_name, sfc_path, vnfi_list): create sfc in the testbed
# Input: flowclassifier name, sfc name, sfc path, vnfi_info
# Output: response
def set_sfc(sfcr_id, sfc_name, inst_in_sfc):

    del inst_in_sfc[0]
    instIDs = []

    for inst in inst_in_sfc:
        instIDs.append([ inst.id ])

    sfc_spec = ni_nfvo_client.SfcSpec(sfc_name=sfc_name,
                                   sfcr_ids=[ sfcr_id ],
                                   vnf_instance_ids=instIDs)

    api_response = ni_nfvo_sfc_api.set_sfc(sfc_spec)

    return api_response



def create_sfc(sfcr, instance_id_list):

    sfc_spec =ni_nfvo_client.SfcSpec(sfc_name=sfcr.name,
                                 sfcr_ids=[sfcr.id],
                                 vnf_instance_ids=instance_id_list,
                                 is_symmetric=False)


    api_response = ni_nfvo_sfc_api.set_sfc(sfc_spec)
    print("Success to pass for creating sfc")
    return api_response


# get_instance_info(instance, flavor): create sfc in the testbed
# Input: flowclassifier name, sfc name, sfc path, vnfi_info
# Output: response
def get_instance_info(instance, flavor):
    resource_type = ["cpu_usage___value___gauge",
                     "memory_free___value___gauge"]

    info = { "id": instance.id, "cpu" : 0.0, "memory": 0.0}

    # Set time-period to get resources
    end_time = dt.datetime.now() #+ dt.timedelta(hours=24)
    start_time = end_time - dt.timedelta(seconds = 10)

    if str(end_time)[-1]!='Z':
         end_time = str(end_time.isoformat())+ 'Z'
    if str(start_time)[-1]!='Z':
         start_time = str(start_time.isoformat()) + 'Z'


    for resource in resource_type:
        query = ni_mon_api.get_measurement(instance.id, resource, start_time, end_time)
        value = 0

        for response in query:
            value = value + response.measurement_value

        value = value/len(query) if len(query) > 0 else 0

        if resource.startswith("cpu"):
            info["cpu"] = value
        elif resource.startswith("memory"):
            memory_ram_mb = flavor.ram_mb
            memory_total = 1000000 * memory_ram_mb
            info["memory"] = 100*(1-(value/memory_total)) if len(query) > 0 else 0

    return info


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



def get_hops_in_topology(src_node, dst_node):
    #print("-------------------")
    #print(src_node, dst_node)
    nodes = [ "ni-compute-181-155", "ni-compute-181-156", "ni-compute-181-157", "ni-compute-181-158", "ni-compute-181-203", "ni-compute-181-162", "ni-compute-kisti", "ni-compute-181-154"]
    hops = [[1, 2, 4, 4, 4, 6, 8, 10],
            [2, 1, 4, 4, 4, 6, 8, 10],
            [4, 4, 1, 2, 2, 6, 8, 10],
            [4, 4, 2, 1, 2, 6, 8, 10],
            [4, 4, 2, 2, 1, 6, 8, 10],
            [6, 6, 6, 6, 6, 1, 8, 10],
            [8, 8, 8, 8, 8, 8, 1, 10],
            [10, 10, 10, 10, 10, 10, 10, 1]]

    return hops[nodes.index(src_node)][nodes.index(dst_node)]

# get_node_info(): get all node information placed in environment
# Input: null
# Output: Node information list
def get_node_info(flavor):
    query = ni_mon_api.get_nodes()

    response = [ node_info for node_info in query if node_info.type == "compute" and node_info.status == "enabled"]
    response = [ node_info for node_info in response if not (node_info.name).startswith("NI-Compute-82-9")]
    response = [ node_info for node_info in response if node_info.n_cores_free >= flavor.n_cores and node_info.ram_mb >= flavor.ram_mb]

    return response


# get_nfvo_vnf_spec(): get ni_nfvo_vnf spec to interact with a nfvo module
# Input: null
# Output: nfvo moudle's vnf spec
def get_nfvo_vnf_spec():

    nfvo_client_cfg = ni_nfvo_client.Configuration()

    nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
    ni_nfvo_vnf_spec = ni_nfvo_client.VnfSpec(ni_nfvo_client.ApiClient(nfvo_client_cfg))
    ni_nfvo_vnf_spec.user_data = sample_user_data % cfg["instance"]["password"]

    return ni_nfvo_vnf_spec

# deploy_vnf(vnf_spec): deploy VNF instance in OpenStack environment
# Input: VnFSpec defined in nfvo client module
# Output: API response
def deploy_vnf(vnf_spec):
    instID = ni_nfvo_vnf_api.deploy_vnf(vnf_spec)
    
    print(vnf_spec)
    print(instID)
    
    limit = 500 
    for i in range (0, limit):
        time.sleep(2)

        if check_active_instance(instID):
            return get_specific_vnf_info(instID)
        elif i == (limit-1):
            print("destroy vnf")
            destroy_vnf(instID)

    return ""


# check_active_instance(id): Check an instance whether it's status is ACTIVE
# Input: instance id
# Output: True or False
def check_active_instance(id):
    status = ni_mon_api.get_vnf_instance(id).status

    if status == "ACTIVE":
        return True
    else:
        return False


def reward_calculator(src, dst):
    cost = 1.25
    resTime = 0

    for port in src.ports:
        if port.network_id == openstack_network_id:
            src_ip = port.ip_addresses[-1]
            break

    for port in dst.ports:
        if port.network_id == openstack_network_id:
            dst_ip = port.ip_addresses[-1]
            break

    for i in range (0, 15):
        time.sleep(2)

        command = ("sshpass -p %s ssh -o stricthostkeychecking=no %s@%s ./test_ping_e2e.sh %s %s %s %s" % (cfg["traffic_controller"]["password"],
                                                                               cfg["traffic_controller"]["username"],
                                                                               cfg["traffic_controller"]["ip"],
                                                                               src_ip,
                                                                               cfg["instance"]["username"],
                                                                               cfg["instance"]["password"],
                                                                               dst_ip))
        print(command)
        command = command + " | grep avg | awk '{split($4,a,\"/\");print a[2]}'"

        resTime = subprocess.check_output(command, shell=True).strip().decode("utf-8")

        if resTime != "":
            resTime = float(resTime)/1000.0
            reward = -math.log(1.0+resTime)*cost

            return reward


    return 10


def dqn_training(sfc_info):
    epsilon_value = epsilon
    n_epi = 0
    training_list.append(sfc_info.sfc_name)

    # Q-network, Target Q-network, remplay memory
    q = Qnet(3, 2, 32) # State 3, Action 2, Neuron 32
    q_target = Qnet(3, 2, 32) # State 3, Action 2, Neuron 32
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    memory = ReplayBuffer(buffer_limit)

    flavor_info = get_all_flavors()

    while True:
        time.sleep(10)
        vnf_info = get_vnf_info(sfc_info.sfcr_name)
        sfc_vnfs = get_sfcr_by_name(sfc_info.sfcr_name).nf_chain

        # Insert Traffic classifer instance
        deployedInst = []
        instInSFC = []
        src = ni_mon_api.get_vnf_instance(get_sfcr_by_name(sfc_info.sfcr_name).source_client)
        instInSFC.append(src)
        #instInSFC.append(vnf_info[0][-1]) #######Shouldfixed!

        # Create State (Except for Traffic classifer)
        for vnf in vnf_info:
            cpuUtil = 0
            memUtil = 0
            placement = 0
            instSize = len(vnf)
            resourceInfo = []

            # Measure mean values per each VNF type
            for inst in vnf:
                flavor = [ flavor for flavor in flavor_info if flavor.id == inst.flavor_id ][-1]
                inst_resUtil = get_instance_info(inst, flavor)

                cpuUtil = cpuUtil + (inst_resUtil["cpu"]/instSize)
                memUtil = memUtil + (inst_resUtil["memory"]/instSize)
                placement = placement + get_hops_in_topology(instInSFC[-1].node_id, inst.node_id)/instSize
                resourceInfo.append({"id": inst.id, "cpu": inst_resUtil["cpu"], "memory": inst_resUtil["memory"], "placement": get_hops_in_topology(instInSFC[-1].node_id, inst.node_id)})

            # Create state
            state = np.array([cpuUtil, memUtil, placement])
            epsilon_value = max(0.50, epsilon_value*0.99)
            action = q.sample_action(torch.from_numpy(state).float(), epsilon_value)["action"]


            if action == 0: # Select an instance
                print("select")
                resourceInfo.sort(key=lambda info: info["placement"])
                resourceInfo.sort(key=lambda info: info["cpu"])
                instInSFC.append([ inst for inst in vnf if inst.id == resourceInfo[0]["id"] ][-1])

            elif action == 1: # Deploy a new instance
                print("Deploy")
                node_info = get_node_info(flavor)
                node_info = [ {"id": node.id, "distance": get_hops_in_topology(instInSFC[-1].node_id, node.id) } for node in node_info if node.id != "ni-compute-kisti"]
                node_info.sort(key=lambda info: info["distance"])

                vnf_spec = get_nfvo_vnf_spec()
                vnf_type = sfc_vnfs[vnf_info.index(vnf)]
                vnf_spec.vnf_name = sfc_info.sfcr_name + cfg["instance"]["prefix_splitter"] +vnf_type + " " + dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                vnf_spec.image_id = cfg["image"][vnf_type]
                vnf_spec.flavor_id = flavor.id
                vnf_spec.node_name = node_info[0]["id"]

                # After successful deployment, add information of the deployed instance
                newInst = deploy_vnf(vnf_spec)

                if newInst != "":
                    deployedInst.append(newInst)
                    instInSFC.append(newInst)
                else:
                    instInSFC.append(random.choice(vnf))


            # Reward calculation
            time.sleep(10)
            length = len(instInSFC)
            #print(instInSFC)
            #print(length)

            reward = reward_calculator(instInSFC[length-2], instInSFC[length-1])
            print(reward) #for debugging
            # Create new state
            flavor = [ flavor for flavor in flavor_info if flavor.id == instInSFC[-1].flavor_id ][-1]
            inst_resUtil = get_instance_info(instInSFC[-1], flavor)
            new_cpuUtil = ((cpuUtil*instSize)+inst_resUtil["cpu"])/(instSize+1)
            new_memUtil = ((memUtil*instSize)+inst_resUtil["memory"])/(instSize+1)
            new_placement = ((placement*instSize)+get_hops_in_topology(instInSFC[-2].node_id, instInSFC[-1].node_id))/(instSize+1)
            nextState = np.array([new_cpuUtil, new_memUtil, new_placement])

            # Store in Replay memory
            transition = (state, action, reward, nextState, 1.0)
            memory.put(transition)

            if memory.size() > required_mem_size:
                train(q, q_target, memory, optimizer, gamma, batch_size)

            if n_epi % print_interval==0 and n_epi != 0:
                q_target.load_state_dict(q.state_dict())

            n_epi = n_epi+1

            if len(instInSFC) == len(sfc_vnfs):
                for inst in deployedInst:
                    destroy_vnf(inst.id)

            # Finish
            if sfc_info.sfc_name not in training_list:
                q.save_model("./dqn_models/"+sfc_info.sfc_name)

                for inst in deployedInst:
                    destroy_vnf(inst.id)

                print("[Training finish] " + sfc_info.sfc_name)


def dqn_based_sfc(sfc_info):

    q = Qnet(3, 2, 32) # State 3, Action 2, Neuron 32
    q.load_state_dict(torch.load("./dqn_models/" + sfc_info.sfc_name))

    flavor_info = get_all_flavors()
    vnf_info = get_vnf_info(sfc_info.sfcr_name)
    sfc_vnfs = get_sfcr_by_name(sfc_info.sfcr_name).nf_chain

    # Insert Traffic classifer instance
    instInSFC = []
    instance_id_list = []

    src = ni_mon_api.get_vnf_instance(get_sfcr_by_name(sfc_info.sfcr_name).source_client)
    instInSFC.append(src)
    #instInSFC.append(vnf_info[0][-1]) #######Shouldfixed!

    # Create state
    for vnf in vnf_info:
        cpuUtil = 0
        memUtil = 0
        placement = 0
        instSize = len(vnf)
        resourceInfo = []

        # Measure mean values
        for inst in vnf:
            flavor = [ flavor for flavor in flavor_info if flavor.id == inst.flavor_id ][-1]
            inst_resUtil = get_instance_info(inst, flavor)

            cpuUtil = cpuUtil + (inst_resUtil["cpu"]/instSize)
            memUtil = memUtil + (inst_resUtil["memory"]/instSize)
            placement = placement + get_hops_in_topology(instInSFC[-1].node_id, inst.node_id)/instSize
            resourceInfo.append({"id": inst.id, "cpu": inst_resUtil["cpu"], "memory": inst_resUtil["memory"], "placement": get_hops_in_topology(instInSFC[-1].node_id, inst.node_id)})

        print("resourceInfo : ", resourceInfo)
        # Create state
        state = np.array([cpuUtil, memUtil, placement])
        action = q.sample_action(torch.from_numpy(state).float(), 0)["action"]


        if action == 0: # Select
            resourceInfo.sort(key=lambda info: info["placement"])
            resourceInfo.sort(key=lambda info: info["cpu"])
            instInSFC.append([ inst for inst in vnf if inst.id == resourceInfo[0]["id"] ][-1])
            instance_id_list.append([ [ inst.id for inst in vnf if inst.id == resourceInfo[0]["id"] ][-1] ])


        #It should consider only SFC selection... why it deploy??????

        elif action == 1: # Deploy
            node_info = get_node_info(flavor)
            node_info = [ {"id": node.id, "distance": get_hops_in_topology(instInSFC[-1].node_id, node.id) } for node in node_info if node.id != "ni-compute-kisti"]
            node_info.sort(key=lambda info: info["distance"])

            vnf_spec = get_nfvo_vnf_spec()
            vnf_type = sfc_vnfs[vnf_info.index(vnf)]
            vnf_spec.vnf_name = sfc_info.sfcr_name + cfg["instance"]["prefix_splitter"] +vnf_type + " " + dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            vnf_spec.image_id = cfg["image"][vnf_type]
            vnf_spec.flavor_id = flavor.id
            vnf_spec.node_name = node_info[0]["id"]

            # Deployment success
            newInst = deploy_vnf(vnf_spec)

            if newInst != "":
                instInSFC.append(newInst)
                instance_id_list.append([newInst.id])
            else:
                instInSFC.append(random.choice(vnf))
                instance_id_list.append([random.choice(vnf).id])

    del instInSFC[0]

    # Create SFC
    sfcr = get_sfcr_by_name(sfc_info.sfcr_name)

    #sfcID = set_sfc(sfcrID, sfc_info.sfc_name, instInSFC)
    sfc_id = create_sfc(sfcr, instance_id_list)
    sfcPath = [ inst.name for inst in instInSFC ]

    response = { "sfcr_id": sfcr.id,
                 "sfc_id": sfc_id,
                 "sfc_path": sfcPath }

    pprint(sfcPath)

    return response
    
    
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
    


def test_measure_response_time():
    
    try:
        response = "Cannot find test-sfcrs for test"
        deployed_sfcrs = ni_nfvo_sfcr_api.get_sfcrs()
        for sfcr in deployed_sfcrs:
            if sfcr.name.startswith("test-sfc"):
                print(sfcr.name)
                if get_sfc_by_name(sfcr.name):
                    target_sfcr = sfcr
                    #continue

    except:
        return "There is no target sfcr for sfc evaluation"
            
    src_ip = (target_sfcr.src_ip_prefix).split('/')[0]
    dst_ip = (target_sfcr.dst_ip_prefix).split('/')[0]
    
  
    cnd_path = os.path.dirname(os.path.realpath(__file__))

    command = ("sshpass -p %s ssh -o stricthostkeychecking=no %s@%s ./test_http_e2e.sh %s %s %s %s %s" % (cfg["traffic_controller"]["password"],
                                                                            cfg["traffic_controller"]["username"],
                                                                            cfg["traffic_controller"]["ip"],
                                                                            src_ip,
                                                                            cfg["instance"]["username"],
                                                                            cfg["instance"]["password"],
                                                                            cfg["traffic_controller"]["num_requests"],
                                                                            dst_ip))

    command = command + " | grep 'Time per request' | head -1 | awk '{print $4}'"


    print(command)
    # Wait until web server is running
    start_time = dt.datetime.now()


    while True:
        #print("19-while loop")
        time.sleep(1)
        response = subprocess.check_output(command, shell=True).strip().decode("utf-8")
        if response != "":
            response = float(response) * 10
            #print("if")
            pprint("[Test] %s" % (response))
            f = open("test_monitor.txt", "a+", encoding='utf-8')
            f.write(str(response)+'\n')
            f.close()
            print("write done")
            return float(response)
        elif (dt.datetime.now() - start_time).seconds > 60:
            #print("elif")
            return -1    
    
