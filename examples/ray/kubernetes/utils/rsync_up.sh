#!/bin/bash

# Example: ./rsync_up.sh cluster-name ~/repos/theflow

cluster_name="${1:-$CLUSTER_NAME}"
theflow_local_dir=$2

pods=$(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep ${cluster_name}-)
script_full_path=$(dirname "$0")

echo "Rsync to head and workers..."
for pod in $pods
do
    echo "Rsync to pod: $pod"
    ${script_full_path}/krsync.sh -a --progress --stats ${theflow_local_dir}/theflow/ ${pod}:/home/ray/anaconda3/lib/python3.7/site-packages/theflow
done
