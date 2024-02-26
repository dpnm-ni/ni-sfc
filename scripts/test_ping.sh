#!/bin/bash

# Get library to use functions
source library

isSet_hostname # Solve the hostname warning

if [[ $1 == "-h"  ]]; then
  echo "[Help] Please enter the command: ./test_ping.sh {target_server}"
  echo "[Help] example: ./test_ping.sh 8.8.8.8"
else
  if [[ ! -z $1 ]]; then
    ping $1 -c 4
  else
    echo "[Error] Please enter a target_server"
  fi
fi
