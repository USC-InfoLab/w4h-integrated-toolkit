#!/bin/bash


cp "conf/db_config_sample.yaml" "conf/db_config.yaml"

python init_user.py &

python stream_sim.py &

streamlit run viz.py


wget
lsb_release
gnupg
service postgresql start
/etc/postgresql/{version}/main/pg_hba.conf