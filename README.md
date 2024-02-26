# ni-sfc-path-module
NI-SFC-Path-Module creates SFC by selecting/creating VNFs running from the OpenStack testbed. 

(This module is private and already configured to be used to DPNM testbed)

## Main Responsibilities
Random or RL-based SFC path selection module.
- Provide APIs to create random SFCs
- Provide APIs to create SFCs by using Q-learning
- Provide APIs to create SFC by using Deep Q-network (DQN)

## Requirements
```
Python 3.6.5+
```

Please install pip3 and requirements by using the command as below.
```
sudo apt-get update
sudo apt-get install python3-pip
pip3 install -r requirements.txt
```

## Configuration
This module runs as web server to handle an SFC request that describes required data for an SFC.
To use a web UI of this module or send an SFC request to the module, a port number can be configured (a default port number is 8001)

```
# server/__main__.py

def main():
    app = connexion.App(__name__, specification_dir='./swagger/')
    app.app.json_encoder = encoder.JSONEncoder
    app.add_api('swagger.yaml', arguments={'title': 'NI SFC Sub-Module Service'})
    app.run(port=8001) ### Port number configuration
```

This module interacts with ni-mano to create SFC in OpenStack environment.
To communicate with ni-mano, this module should know URI of ni-mano.
In ni-mano, ni_mon and ni_nfvo are responsible for interacting with this module so their URI should be configured as follows.
If you use DQN-based SFC composition function, it needs to determine either selecting VNF instnace or creating a new VNF instances by measuring response time of VNF instances.
To measure the response time, we assume that there is a monitoring instance that has a public IP addresee in OpenStack environment. 
Thus, the SFC module accesses to the monitoring instance for measuring response time.
If the monitoring instance fails to measure the response time due to missing script files, please move shell files locateed in scripts directory to the instance. 
To make a SSH connection between the SFC module and the monitoring instance, SSH information should be configured as follows. 
Finally, the image IDs are used to create a new VNF instances. 

```
# config/config.yaml

ni_mon:
  host: http://<ni_mon_ip>:<ni_mon_port>      # Configure here to interact with a monitoring module
ni_nfvo:
  host: http://<ni_nfvo_ip>:<ni_nfvo_port>    # Configure here to interact with an NFVO module
instance:                                   
  monitor: <IP of a monitoring instance>      # IP of traffic generator (i.e., VM instance that has a public IP address)
  id: <ssh_id of a monitoring instance>       # SSH ID of new VNF instance
  password: <ssh_id of a monitoring instance> # SSH ID of new VNF instance
image:                                        # Image IDs used by OpenStack
  firewall: <OpenStack Image ID>
  flowmonitor: <OpenStack Image ID>
  dpi: <OpenStack Image ID>
  ids: <OpenStack Image ID>
  proxy: <OpenStack Image ID>
```

Before running this module, OpenStack network ID should be configured because VNF instances in OpenStack can have multiple network interfaces.
This module uses *openstack_network_id* value to identify a network interface used to create an SFC.
Moreover, Q-learning or DQN hyper-parameters can be configured as follows (they have default values).

```
# sfc_path_selection.py

# Parameters
# OpenStack Parameters
openstack_network_id = ""    # Insert OpenStack Network ID to be used for creating SFC

# <Important!!!!> parameters for Reinforcement Learning (Q-learning in this codes)
learning_rate = 0.10         # Learning rate
discount_factor = 0.60       # Discount factor
initial_epsilon = 0.90       # epsilon value of e-greedy algorithm
num_episode = 3000           # Number of iteration for Q-learning
```

```
# sfc_using_dqn.py

# Parameters
# OpenStack Parameters
openstack_network_id = ""       # Insert OpenStack Network ID to be used for creating SFC

# <Important!!!!> parameters for Reinforcement Learning (DQN in this codes)
learning_rate = 0.01            # Learning rate
gamma         = 0.98            # Discount factor
buffer_limit  = 10000           # Maximum Buffer size
batch_size    = 16              # Batch size for mini-batch sampling
num_neurons = 64                # Number of neurons in each hidden layer
epsilon = 0.99                  # epsilon value of e-greedy algorithm
required_mem_size = 24          # Minimum number triggering sampling
print_interval = 24             # Number of iteration to print result during DQN
```

## Usage

After installation and configuration of this module, you can run this module by using the command as follows.

```
python3 -m server
```

This module provides web UI based on Swagger:

```
http://<host IP running this module>:<port number>/ui/
```

To create an SFC in OpenStack testbed, this module processes a HTTP POST message including SFCInfo data in its body.
You can generate an SFC request by using web UI or using other library creating HTTP messages.
If you create and send a HTTP POST message to this module, the destination URI is as follows.

```
# Choose and create an SFC path randomly 
# [HTTP POST]
http://<host IP running this module>:<port number>/path_selection/dqn

# Choose and create an optimal SFC path using Q-learning
# [HTTP POST]
http://<host IP running this module>:<port number>/path_selection/q_learning

# Choose and create an SFC path randomly 
# [HTTP POST]
http://<host IP running this module>:<port number>/path_selection/random
```

Required data to create SFC is defined in SFCInfo model that is JSON format data.
The SFCInfo model consists of 4 data as follows.

- **sfc_name**: a name of SFC identified by OpenStack (or DQN model name if you request DQN)
- **sfc_prefix**: a prefix to identify instances which can be components of an SFC from OpenStack
- **sfc_vnfs**: a string array including a flow classifier name and name of each VNF instance in order
- **sfcr_name**: a name of flow classifier identified by OpenStack

For example, if an SFC request includes SFCInfo data as follows, this module identifies an instance of which name is *test-client* as a flow classifier and VNF instances of which name starts with *test-firewall* and *test-dpi* to create an SFC.

```
    {
      “sfc_name”: "sample-sfc",
      “sfc_prefix”: “test-”,
      “sfc_vnfs”: [
        “client”, “firewall”, “dpi”
      ],
      “sfcr_name”: “sample-sfcr”
    }
```

In addition, DQN-based SFC composition method requires a DQN model to determine actions, such as creating VNF instances or selecting VNF instances. 
For this, SFC module provides URIs as follows. 

```
# Create DQN model training process
# [HTTP POST]
http://<host IP running this module>:<port number>/path_selection/dqn_training

# Stop DQN model training process
# [HTTP DELETE]
http://<host IP running this module>:<port number>/path_selection/del_dqn_training/{id}

# Get a ID list of all DQN model training processes 
# [HTTP GET]
http://<host IP running this module>:<port number>/path_selection/get_training_process
```
