import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
import tensorflow as tf

x=tf.placeholder(tf.float32,[None,784])
#x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值
# 。我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。我们用2维的浮点数张量来表示这些图，
# 这个张量的形状是[None，784 ]。（这里的None表示此张量的第一个维度可以是任何长度的。）

#设置权重 和 偏移量
#tensorflow用更好的办法来表示他们：variable.一个variable可以代表一个可修改的张量
w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
# W的维度是[784，10]，因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量，
# 每一位对应不同数字类。b的形状是[10]，所以我们可以直接把它加到输出上面。
y=tf.nn.softmax(tf.matmul(x,w)+b)#实现一个模型。











