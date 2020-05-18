import sys
import tensorflow as tf

from binascii import b2a_hex,a2b_hex

#创建一个op，产生一个1*2的矩阵，这个op被作为一个节点加到默认的图中
#构造器的返回值代表op常量的返回值
matrix1=tf.constant([[3.,3.]])
#创建另外一个常量op,产生一个2*1的矩阵
matrix2=tf.constant([[2.],[2.]])
#创建一个矩阵乘法matmul op ，把 matrix1和matrix2作为输入，返回值product代表矩阵乘法的结果
product =tf.matmul(matrix1,matrix2)

#创建Session对象，如没有任何参数，则启动默认图
sess=tf.Session()
# 调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数.
# 上面提到, 'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回矩阵乘法 op 的输出.
# 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的.
# 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.
# 返回值 'result' 是一个 numpy `ndarray` 对象.
result=sess.run(product)
print(result)
sess.close()

#此处with代码段功能跟上边的代码功能一样。实现sess的自动关闭、释放资源。
with tf.Session() as sess:
    result=sess.run([product])
    print(result)
#====================================================================================================
#如果机器上有超过一个可用的 GPU, 除第一个外的其它 GPU 默认是不参与计算的. 为了让 TensorFlow 使用这些 GPU,
#你必须将 op 明确指派给它们执行. with...Device 语句用来指派特定的 CPU 或 GPU 执行操作:
with tf.Session() as sess:
    with tf.device("/gpu:1"):
        matrix1 = tf.constant([[3., 3.]])
        matrix2 = tf.constant([[2.], [2.]])
        product = tf.matmul(matrix1, matrix2)
        result=sess.run([product])
        print(result)
#====================================================================================================
#为了便于使用Ipytho这类的Python交互环境，可以使用interactiveSession代替Session类，使用tensor.eval()
#和operation.run()方法代替session.run().这样可以避免使用一个变量来持有会话’
sess=tf.InteractiveSession();
x=tf.Variable([1.0,2.0])
a=tf.constant([3.0,3.0])
#使用初始化器的run方法初始化x
x.initializer.run()
#增加一个减法op，
sub=tf.subscribe(x,a)
add=tf.add()
print(sub)

#====================================================================================================
#Tensor
#TensorFlow 程序使用 tensor 数据结构来代表所有的数据, 计算图中, 操作间传递的数据都是 tensor.
#你可以把 TensorFlow tensor 看作是一个 n 维的数组或列表. 一个 tensor 包含一个静态类型 rank, 和 一个 shape
#变量
#Variables for more details. 变量维护图执行过程中的状态信息。使用变量实现一个简单的计数器功能
#创建一个变量，初始化为标量0
state =tf.Variable(0,name="counter")
# 创建一个op，其作用是使得state加1
one=tf.constant(1)
new_value=tf.add(state,one)
update =tf.assign(state,new_value)#assign是图所描绘的表达式的一部分

#启动图后，变量必须先经过初始化 op初始化，首先必须增加一个 初始化op到图中
init_op=tf.initialize_all_variables()
#启动图，运行op
with tf.Session() as sess:
    #运行init op
    sess.run(init_op)
    #打印state的初始值
    print(sess.run(state))
    #运行OP，更新state，并打印state
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

#==============================================================================================================
#fetch
#为了取回操作的输出内容，可以在使用Session对象的run调用执行图时候，传入一些tensor，这些tensor会帮助你取回结果，
#可以取回单个节点state,但是你可以取回多个tensor：
input1=tf.constant(3.0)
input2=tf.constant(2.0)
input3=tf.constant(5.0)
intermed=tf.add(input2,input3)
mul=tf.multiply(input1,intermed)
with tf.Session as sess:
    result=sess.run([mul,intermed])
    print(result)
#要获取多个tensor值，在op的一次运行中一起获取。
#===============================================================================================================
#feed
#TensorFlow 还提供了 feed 机制, 该机制 可以临时替代图中的任意操作中的 tensor 可以对图中任何操作提交补丁,
# 直接插入一个 tensor
input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)
output=tf.multiply(input1,input2)

with tf.Session as sess:
    print(sess.run([output],feed_dict={input1:[7.],input2:[2.]}))
    







