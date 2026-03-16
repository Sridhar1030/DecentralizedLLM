#!/bin/bash
set -e

echo "Starting Ray worker, connecting to head at ${RAY_HEAD_ADDRESS}..."
ray start --address="${RAY_HEAD_ADDRESS}" --block
