# Worker functions for training

# These functions are generators for alteration objects
# They all share the purpose but differ in how they generate modifications or what modifications they create.

# How it works:
# 1. Main process starts a child with one of these methods as target
# 2. The child creates an alteration (this what differs one method from another)
# 3. The child runs short testing over it
# 4. If it matches the target score, add the alteration object to the out queue
# 5. Go back to step 2

# The mentioned out que is processed by the main process
# 1. Each object added to the queue is evaluated with a pack of different seeds
#    This helps to avoid cases when an alteration just was lucky in short test
# 2. If the performance in general is greater than with original model, the alteration is then applied to the model
# 3. Main process shut down workers
# 4. Main process restart workers with updated model


def random(queue, data):
    # Purely random search
    # data arg is a complex dict object, containing all possible information that can be helpful in search.

    # TensorFlow import was necessary with multiprocessing module. TODO double-check this.
    import tensorflow as tf
    import random
    import time
    import gym

    env_id = data['env_id']
    model = data['model']

    while True:
        print("sleeping")
        time.sleep(10)
        x = random.uniform(0, 1)
        y = tf.multiply(x, 2)
        queue.put(y)
