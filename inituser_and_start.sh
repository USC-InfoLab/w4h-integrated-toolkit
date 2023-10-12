#!/bin/bash
python init_user.py &

python stream_sim.py &

streamlit run viz.py