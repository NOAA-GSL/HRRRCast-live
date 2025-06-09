#!/bin/bash
export CONDA_OVERRIDE_CUDA="12.0"
conda env create -f environment.yaml
