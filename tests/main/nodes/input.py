import tensorflow as tf
import numpy as np
from src.main.nodes import Input

tf.compat.v1.disable_eager_execution()
node = Input('input', 5)
test_array = np.ones(shape=(5,1))
graph = tf.Graph()
node.to_graph(graph)
with tf.compat.v1.Session(graph=graph) as sess:
    x = sess.run(node.outputs, feed_dict={node.graph_name: test_array})
print(f"{test_array} \n    ||\n    ||\n    \\/")
print(x)
