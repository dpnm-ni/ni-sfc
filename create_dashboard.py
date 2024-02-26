import copy
import datetime
import json
import requests
from string import ascii_uppercase

# Function to create a Grafana panel
def generate_grafana_panel(refIds, aliases, measurements, title, gridPos, id):
    panel = ""
    with open("grafana-dashboard-template.json") as infile:
        dash_json = json.load(infile)

        # Use panel from template file as starting point and replace content.
        panel = copy.deepcopy(dash_json["dashboard"]["panels"][0])
    target_template = panel["targets"][0]
    targets = []
    for i in range(0, len(refIds)):
        # Start with target template and replace relevant fields.
        targets.append(copy.deepcopy(target_template))
        targets[i]["refId"] = refIds[i]
        targets[i]["alias"] = aliases[i]
        targets[i]["measurement"] = measurements[i]
        targets[i]["select"][0][1]["type"] = "distinct"
    panel["targets"] = targets
    panel["title"] = title
    panel["gridPos"] = gridPos
    panel["id"] = id
    return panel

### This script gets VNF Instance information from the monitoring module and creates a Grafana Dashboard

# Get input parameters from the input.json file
with open("input.json") as infile:
    input = json.load(infile)
    query_url = input["module"]["monitoring_url"] + "/vnfinstances"
    tag = input["dashboard"]["prefix"]
    sub_tag = input["dashboard"]["vnfs"]
    network_id = input["openstack_conf"]["network_id"]
    dashboard_id = input["dashboard"]["id"]
    dashboard_pw = input["dashboard"]["pw"]
    dashboard_url = input["dashboard"]["url"]


def create_dashboard(vnf_instances, dashboard_name="Test"):

    # Get VNF information from the monitoring module
    response = requests.get(query_url)
    res_json = response.json()
    n_tags = len(sub_tag)


    for i in range(0, n_tags):
        temp_list = [vnfi for vnfi in res_json if vnfi["name"].startswith(tag + sub_tag[i]) and vnfi["status"] == "ACTIVE"]
        temp_list.sort(key=lambda vnfi: vnfi["name"])
        vnf_instances.append(temp_list)

    n_vnfs = len(vnf_instances)
    panels = []

    # Create panel for each VNF
    for i in range(0, n_tags):
        vnfs = vnf_instances[i]

        #print("vnfs : ", vnfs)
        n_vnfs = len(vnfs)

        ## CPU panel
        refIds_cpu = [x for x in ascii_uppercase[0:n_vnfs]]
        #print("refIds_cpu : ", refIds_cpu)
        aliases_cpu = [vnfi.name for vnfi in vnfs]
        #print("aliases_cpu : ", aliases_cpu)
        measurements_cpu = ["%s___cpu_usage___value___gauge" % (vnfi.id) for vnfi in vnfs]
        #print("measurements_cpu : {}".format(measurements_cpu))
        cpu_panel = generate_grafana_panel(refIds_cpu, aliases_cpu, measurements_cpu, "CPU Usage - " + sub_tag[i], {"h": 9, "w": 6, "x": 0, "y": 9*i}, 1+4*i)

        panels.append(cpu_panel)

        ## Memory panel
        refIds_memory = [x for x in ascii_uppercase[0:n_vnfs]]
        aliases_memory = [vnfi.name for vnfi in vnfs]
        measurements_memory = ["%s___memory_free___value___gauge" % (vnfi.id) for vnfi in vnfs]
        #print("measurements_memory : {}".format(measurements_memory))
        memory_panel = generate_grafana_panel(refIds_memory, aliases_memory, measurements_memory, "Memory Free - " + sub_tag[i], {"h": 9, "w": 6, "x": 6, "y": 9*i}, 2+4*i)

        panels.append(memory_panel)

        ## Disk panel
        aliases_read = [vnfi.name + " (disk read)" for vnfi in vnfs]
        measurements_read = ["%s___vda___disk_octets___read___derive" % (vnfi.id) for vnfi in vnfs]

        aliases_write = [vnfi.name + " (disk write)" for vnfi in vnfs]
        measurements_write = ["%s___vda___disk_octets___write___derive" % (vnfi.id) for vnfi in vnfs]

        refIds_disk = [x for x in ascii_uppercase[0:n_vnfs*2]]
        aliases_disk = aliases_read + aliases_write
        measurements_disk = measurements_read + measurements_write
        #print("measurements_disk : {}".format(measurements_disk))
        disk_panel = generate_grafana_panel(refIds_disk, aliases_disk, measurements_disk, "Disk Operation Read/Write - " + sub_tag[i], {"h": 9, "w": 6, "x": 12, "y": 9*i}, 3+4*i)

        panels.append(disk_panel)

        ## Traffic panel
        aliases_rx = [vnfi.name + " (packet rx)" for vnfi in vnfs]
        measurements_rx = ["%s___tap%s___if_packets___rx___derive" % (vnfi.id, port.port_id[0:11]) for vnfi in vnfs for port in vnfi.ports if port.network_id == network_id]

        aliases_tx = [vnfi.name + " (packet tx)" for vnfi in vnfs]
        measurements_tx = ["%s___tap%s___if_packets___tx___derive" % (vnfi.id, port.port_id[0:11]) for vnfi in vnfs for port in vnfi.ports if port.network_id == network_id]

        refIds_traffic = [x for x in ascii_uppercase[0:n_vnfs*2]]
        aliases_traffic = aliases_rx + aliases_tx
        measurements_traffic = measurements_rx + measurements_tx
        #print("measurements_traffic : {}".format(measurements_traffic))
        traffic_panel = generate_grafana_panel(refIds_traffic, aliases_traffic, measurements_traffic, "Packet TX/RX - " + sub_tag[i], {"h": 9, "w": 6, "x": 18, "y": 9*i}, 4+4*i)

        panels.append(traffic_panel)

    # Create a Grafana dashboard config. file to result.json
    with open("grafana-dashboard-template.json") as infile:
        dash_json = json.load(infile)
        dash_json["dashboard"]["panels"] = panels
        dash_json["dashboard"]["title"] = dashboard_name

    # Create a Grafana dashboard on web dashboard
    headers = {'Content-Type': 'application/json'}
    dashboard_query_url = "http://" + dashboard_id + ":" + dashboard_pw + "@" + dashboard_url + "/api/dashboards/import"
    common = "http://" + dashboard_id + ":" + dashboard_pw + "@" + dashboard_url

    response = requests.get(common+"/api/search?query="+dashboard_name, headers=headers)
    
    if response.status_code == 200:
        if len(response.json())>0:
            dashboard_uid = response.json()[0]['uid']
            response = requests.delete(common+"/api/dashboards/uid/"+dashboard_uid, headers=headers)
        response = requests.post(dashboard_query_url, headers=headers, data=json.dumps(dash_json))
    else :    
        response = requests.post(dashboard_query_url, headers=headers, data=json.dumps(dash_json))

    if response.status_code == 200:
        response = requests.get(common+"/api/search?query="+dashboard_name, headers=headers)
        dashboard_uid = response.json()[0]['uid']
        mydashboard_url = common +"/d/"+dashboard_uid+'/tg?refresh=5s&orgId=1'
        print("[Log] dashboard is created! Check your dashboard here: " + mydashboard_url)
    else:
        print("[Error] please check parameters.")

    return mydashboard_url
    

