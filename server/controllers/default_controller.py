import connexion
import six

from server import util
import sfc_path_selection as sfc
import sfc_using_dqn as sfc_dqn
import threading
import time
from server.models.sfc_info import SFCInfo, SFCInfo_DQN

def measure_response_time():

    for i in range(0, 5):
        sfc_dqn.test_measure_response_time()

    return "sucess"


def build_test_environment():

    response = sfc.setup_env_for_test()

    return response


# Determine an SFC path by Q-learning
def q_learning_sfc(body):
    if connexion.request.is_json:
        body = SFCInfo.from_dict(connexion.request.get_json())
        response = sfc.q_based_sfc(body)

    return response

# Determine an SFC path randomly
def random_sfc(body):
    if connexion.request.is_json:
        body = SFCInfo.from_dict(connexion.request.get_json())
        response = sfc.random_sfc(body)

    return response


# Determine an SFC path by Deep Q-network (DQN)
def dqn_sfc(body):
    if connexion.request.is_json:
        body = SFCInfo.from_dict(connexion.request.get_json())
        response = sfc_dqn.dqn_based_sfc(body)

    return response

# List DQN training threads
def get_training_process():
    response = sfc_dqn.training_list

    return response

# Remove one of the running DQN training processes
def del_dqn_training(id):
    if id in sfc_dqn.training_list:
        sfc_dqn.training_list.remove(id)
        response = "Delete " + id
    else:
        response = "Fail to match a training id: " + id

    return response

# Create a thread for training a DQN model
def dqn_training(body):
    if connexion.request.is_json:
        body = SFCInfo.from_dict(connexion.request.get_json())

        threading.Thread(target=sfc_dqn.dqn_training, args=(body,)).start()

    return "Training start!"
