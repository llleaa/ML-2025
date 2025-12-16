#!/bin/sh

for depth in 1 2 3 4; do
    for first_layer_size in 4 8 16 32 64; do
        for sample in 80 50; do
            for annotation in 0.9 0.84 0.77 0.67 0.57; do
                python main.py -d $depth -fls $first_layer_size -e 1000 -s $sample -a $annotation -r 
            done
        done
    done
done
