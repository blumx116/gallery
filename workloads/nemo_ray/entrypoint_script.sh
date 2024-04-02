# ! /bin/bash

mkdir -p /workspace

wait_all_success_or_exit() {
  # https://www.baeldung.com/linux/background-process-get-exit-code
  local pids=("$@")
  while [[ ${#pids[@]} -ne 0 ]]; do
    all_success="true"
    for pid in "${pids[@]}"; do
      code=$(non_blocking_wait "$pid")
      if [[ $code -ne 127 ]]; then
        if [[ $code -ne 0 ]]; then
          echo "PID $pid failed with exit code $code"
          exit "$code"
        fi
      else
        all_success="false"
      fi
    done
    if [[ $all_success == "true" ]]; then
      echo "All pids succeeded"
      break
    fi
    sleep 5
  done
}

non_blocking_wait() {
  # https://www.baeldung.com/linux/background-process-get-exit-code
  local pid=$1
  local code=127 # special code to indicate not-finished
  if [[ ! -d "/proc/$pid" ]]; then
    wait "$pid"
    code=$?
  fi
  echo $code
}

echo "Launching Torch distributed as node rank $NODE_RANK out of $NNODES nodes"
for ((LOCAL_RANK=0; LOCAL_RANK <= $((GPUS_PER_NODE - 1)); LOCAL_RANK++)); do
  RANK=$((8*$NODE_RANK + $LOCAL_RANK))

  OMP_NUM_THREADS=12 RANK=$RANK LOCAL_RANK=$LOCAL_RANK \
  /usr/bin/python3 $TORCH_DISTRIBUTED_TARGET \
    --config-path="/etc/workload-configuration" \
    --config-name="nemo-configuration.yaml" \
    +trainer.num_nodes="$NNODES" \
    +exp_manager.version="$JOB_IDENTIFIER" \
    ${workload_arguments[@]} &

    PID=$!
    PIDS+=($PID)
done

wait_all_success_or_exit "${PIDS[@]}"