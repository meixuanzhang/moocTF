# -*- coding: UTF-8 -*-

# 引入 TensorFlow 库
import tensorflow as tf

# 创建一个常量 Operation（操作）
hw = tf.constant("Hello WOrld ! I love TensorFlow !")

# 启动一个 TensorFLow 的 Session
sess = tf.Session()

# 运行 Graph （计算图）
print sess.run(hw)

# 关闭Session（会话）
sess.close()
