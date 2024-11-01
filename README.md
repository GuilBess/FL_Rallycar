# Federated learning on the rallycar game

The goal of this project was to implement federated learning on different training data datasets ont the rallycar minigame.

Here is a short description of the different files in this github repo:

## data
This is where all our datasets are stored, separated by groups. We will have to extract and preprocess this data later on.

## autopilote.py
This is the part of the code used to do the inference based on the current state of the game. The data is retrieved and we compute what the next move should be using a specific model we load from the files

## best_FL_model.pt
This is the final trained model's weights, saved to our computer. We save it that way so we can load it elsewhere.

## fl.ipynb
This is a notebook used for loading and preprocessing the data, as well as creating the neural networks and training them using the federated learning ideas. We start with all the data and then we train a base model. Then we have as many worker NNs as sub datasets working in different processes. We train them for a certain number of epoch, averaging the weights at each epoch. Finally, int the last round, we average the weights one last time and save them to "best_FL_model.pt". You can read the commented notebook for more infos.