import tensorflow as tf
biases=tf.Variable(tf.zeros([2,3]))#定义一个2x3的全0矩阵
sess=tf.InteractiveSession()#使用InteractiveSession函数
biases.initializer.run()#使用初始化器 initializer op 的 run() 方法初始化 'biases'
print(sess.run(biases))