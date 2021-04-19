# continual-learning-mas-cloning-injection-molding
Code to reproduce the results from the paper "Continual Learning of Neural Networks for Quality Prediction in Production using Memory Aware Synapses and Weight Transfer" by Tercan et al., submitted to the Journal of Intelligent Manufacturing

The script runs MAS-Cloning (described in the paper) for a fixed sequence of tasks (lego bricks) over several test runs. In order to run the script, several parameters have to be provided in the command, such as the task list and the MAS hyperparameters lambda and gamma. 

As an example, you can execute the following command to run the MAS-Cloning with an MLP with two hidden layers (20 neurons per layer) over a 30 sequences of parts with 16 tasks per sequence.  

```
user@machine:~$ python sequence_experiments.py -use_cuda false -hidden_dims 20 20 -batch_size 16 -lr 0.01 -save_models false -lambda 1000. -init_outputs cloning -n_base 1 -n_inc 15 -n_shuffles 5 -n_sequences 30 -exp_name mas_cloning_30_sequences -part_filter 6x1_Lego 4x2_Lego 4x1_Lego 4x1_Lego_flach 3x1_Lego 3x2_Lego 8x2_Lego 8x1_Lego 2x2_Lego 3x2_Lego_flach 6x2_Lego 6x1_Lego_flach 3x1_Lego_flach 2x1_Lego 6x2_Lego_flach 4x2_Lego_flach