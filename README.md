# Weight agnostic neural network

This repo is an approach to reproduce WANN concept described here: https://weightagnostic.github.io/ . The purpose is not to outperform the original NEAT-based algorithm but to learn low-level NN meachanics and TensorFlow API. Therefore I don't advice to use for any other reason than mentioned.

# How it works

The main object is a Model class. It holds information about nodes and connections between them, so basically, this is a kind of graph. From model you can create a TensorFlow Graph, which can be ran in tf.Session.

There is a GymAgent class, which provides somewhat easy way to test the project. Just make an instance of it and run its 'start_training' method. 
```
a = GymAgent("LunarLanderContinuous-v2", num_workers=8)
a.start_training()
```
And you'll see some logs about learning progress. When you wish to stop the process, run 'stop_training' method:
```
a.stop.training()
```
And if you like to see how the agent plays, run 'a.play()'. This was tested in CarPole and LunarLander environments.
