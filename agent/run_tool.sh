#!/bin/bash
source /home/chaima-eddib/anaconda3/etc/profile.d/conda.sh
conda activate glassbox-agent
exec python3 /home/chaima-eddib/GlassBox-AutoML-Agent/agent/tool.py
