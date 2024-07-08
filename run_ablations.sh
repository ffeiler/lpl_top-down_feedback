#! /bin/bash

T=(0 1 2 3 4 5)
for t in "${T[@]}"
do
    python lpl_main.py --topdown --distance_top_down 1 --max_steps 5000 --experiment_name "ext_detached_cbe_ag" --error_correction --error_nb_updates $t
done


# alpha_errors=(0.0 0.1 0.2 0.5 1.0)
# for alpha in "${alpha_errors[@]}"
# do
    # python lpl_main.py --topdown --distance_top_down 1 --max_steps 5000 --experiment_name "ext_detached_abs" --error_correction --alpha_error $alpha
# done