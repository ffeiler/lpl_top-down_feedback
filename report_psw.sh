#! /bin/bash

# T=(0 1 2 3 4 5)
# for t in "${T[@]}"
# do
#     python lpl_main.py --topdown --distance_top_down 1 --max_steps 5000 --experiment_name "ext_detached_cbe_ag" --error_correction --error_nb_updates $t
# done


td_dists=(1 2 3)

for td_dist in "${td_dists[@]}"
do
    python lpl_main.py --topdown --distance_top_down $td_dist --max_steps 5000 --experiment_name "ext_psw" 
    python lpl_main.py --topdown --distance_top_down $td_dist --max_steps 5000 --experiment_name "ext_psw" --symmetric_topdown
    python lpl_main.py --topdown --distance_top_down $td_dist --max_steps 5000 --experiment_name "ext_psw" --topdown_cross_branch
    python lpl_main.py --topdown --distance_top_down $td_dist --max_steps 5000 --experiment_name "ext_psw" --topdown_cross_branch --symmetric_topdown
done