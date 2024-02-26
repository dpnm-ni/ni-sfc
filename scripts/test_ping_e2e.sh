#!/bin/bash

# Get library to use functions
source library
requirement=(sshpass)

isSet_hostname # Solve the hostname warning
isExist_package # Check pacages and install them

# Variables for default setting
target_server=127.0.0.1
test_script=test_ping.sh

if [[ $1 == "-h"  ]]; then
  echo "[Help] Please enter the command: ./test_ping_e2e.sh {source_server} {ssh_id} {ssh_pw} {target_server}"
  echo "[Help] example: ./test_ping_e2e.sh 127.0.0.1 ubuntu_id ubuntu_pw 8.8.8.8"
else
  echo "[Log] ready to send packets by traceroute"
  sshpass -p $3 scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $test_script library $2@$1:~/
  sshpass -p $3 ssh -T -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $2@$1 ./$test_script $4
fi
