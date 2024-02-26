#!/bin/bash

conn=80
max=40

for ((;;))
do

        for ((i=60; i<=$conn; i++))
        do
                sudo ab -c $((i*10)) -n $((i*1000)) -d -q http://10.10.20.85:8888/ > ab_result.txt
                sleep 0.1
        done

        for ((i=$conn; i>60; i--))
        do
                sudo ab -c $((i*10)) -n $((i*1000)) -d -q http://10.10.20.85:8888/ > ab_result.txt
                sleep 0.1
        done

        for ((i=60; i<=$conn; i++))
        do
                if [ $i -ge $max ];then
                        sudo ab -c $((max*10)) -n $((max*1000)) -d -q http://10.10.20.85:8888/ > ab_result.txt
                else
                        sudo ab -c $((i*10)) -n $((i*1000)) -d -q http://10.10.20.85:8888/ > ab_result.txt
                fi

                sleep 0.1
        done

        for ((i=$conn; i>=60; i--))
        do
                if [ $i -ge $max ];then
                        sudo ab -c $((max*10)) -n $((max*1000)) -d -q http://10.10.20.85:8888/ > ab_result.txt
                else
                        sudo ab -c $((i*10)) -n $((i*1000)) -d -q http://10.10.20.85:8888/ > ab_result.txt
                fi

                sleep 0.1
        done

        for ((i=60; i<=$conn; i++))
        do
                sudo ab -c $((i*10)) -n $((i*500)) -d -q http://10.10.20.85:8888/ > ab_result.txt

                sleep 0.1
        done

        for ((i=$conn; i>60; i--))
        do
                sudo ab -c $((i*10)) -n $((i*500)) -d -q http://10.10.20.85:8888/ > ab_result.txt
        done
done

