#! /bin/bash

T=(7 6 5 4 3 2 1 0)
rnd_seeds=(7 42 69 420 3407)

for rnd in "${rnd_seeds[@]}"
do
    for t in "${T[@]}"
    do
        python lpl_main.py --topdown --distance_top_down 1 --max_steps 5000 --experiment_name "ext_detached_cbe_ag" --error_correction --error_nb_updates $t --random_seed $rnd --alpha_error 2.0
    done
done

# alpha=(5.0 10.0 20.0)
# for a in "${alpha[@]}"
# do
#     python lpl_main.py --topdown --distance_top_down 1 --max_steps 5000 --experiment_name "ext_detached_cbe" --error_correction --error_nb_updates 1 --random_seed 69 --alpha_error $a
# done