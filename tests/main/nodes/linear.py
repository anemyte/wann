import numpy as np
import tensorflow as tf
from src.main.nodes.io import Input
from src.main.nodes.linear import Linear


def test1():
    tf.compat.v1.disable_eager_execution()
    node = Input('input', (2, 2))
    test_array = np.array([[1., 2.], [3., 4.]])
    graph = tf.Graph()
    node.to_graph(graph)
    with graph.as_default():
        tf.constant([2.,2.], name="shared_weight_2")
    linear = Linear("lin", node.outputs[0:2])
    linear.to_graph(graph)
    with tf.compat.v1.Session(graph=graph) as sess:
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        x = sess.run(linear.outputs, feed_dict={node.graph_name: test_array})
    print(x)


def test_relu():
    tf.compat.v1.disable_eager_execution()
    node = Input('input', (2, 2))
    test_array = np.array([[1., 2.], [3., 4.]])
    graph = tf.Graph()
    node.to_graph(graph)
    with graph.as_default():
        tf.constant([2., 2.], name="shared_weight_2")
    linear = Linear("lin", node.outputs[0:2], 'relu')
    linear.to_graph(graph)
    with tf.compat.v1.Session(graph=graph) as sess:
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        x = sess.run(linear.outputs, feed_dict={node.graph_name: test_array})
    print(x)


def test_sigmoid():
    tf.compat.v1.disable_eager_execution()
    node = Input('input', (2, 2))
    test_array = np.array([[1., 2.], [3., 4.]])
    graph = tf.Graph()
    node.to_graph(graph)
    with graph.as_default():
        tf.constant([2., 2.], name="shared_weight_2")
    linear = Linear("lin", node.outputs[0:2], 'sigmoid')
    linear.to_graph(graph)
    with tf.compat.v1.Session(graph=graph) as sess:
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        x = sess.run(linear.outputs, feed_dict={node.graph_name: test_array})
    print(x)