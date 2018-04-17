# -*- coding:UTF-8 -*-

# 引入tensorflow
import tensorflow as tf

# 创建两个常量 Tensor
# const1 是一行两列 ，const2 是两行一列
const1 = tf.constant([[2,2]])

const2 = tf.constant([[4],[4]])

multiple = tf.matmul(const1,const2)

print(multiple)

# 创建了Session（会话）对象
sess = tf.Session()

# 用Session的run方法来实际运行multiple这个矩阵的乘法操作
# 并把操作执行的结果赋值给result
result = sess.run(multiple)

# 用print来打印运行的结果
print(result)

if const1.graph is tf.get_default_graph():
    print("const1所在图（Graph）是当前上下文默认的图")

# 关闭已用完的Session（会话）
sess.close()


# 第二种方法创建和关闭Session

with tf.Session() as sess:
    result2 = sess.run(multiple)
    print("Multiple的结果是 %s "% result2 )

