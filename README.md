# Weight agnostic neural network

This repo is an approach to reproduce WANN concept described here: https://weightagnostic.github.io/ . The purpose is not to outperform the original NEAT-based algorithm but to learn low-level NN meachanics and TensorFlow API. Therefore I don't advice to use for any other reason than mentioned.

# How it works

The main object is a Model class. It holds information about nodes and connections between them, so basically, this is a kind of graph. From model you can create a TensorFlow Graph, which can be ran in tf.Session.

There is a GymAgent class, which provides somewhat easy way to test the project. Just make an instance of it and run its 'start_training' and then 'stop_training' methods. This was tested in CarPole and LunarLander environments.
```
a = GymAgent("LunarLanderContinuous-v2", num_workers=8)
a.start_training()
...
a.stop_training()
```
