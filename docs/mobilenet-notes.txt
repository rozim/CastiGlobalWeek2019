
last layer: conv_preds

x = Conv2D(classes, (1, 1),
                   padding='same', name='conv_preds')(x)

mymodel.endpoints
(88) ["input_1", "conv1", "conv1_bn", "conv1_relu", "conv_dw_1", "conv_dw_1_bn", "conv_dw_1_relu", "conv_pw_1", "conv_pw_1_bn", "conv_pw_1_relu", "conv_dw_2", "conv_dw_2_bn", "conv_dw_2_relu", "conv_pw_2", "conv_pw_2_bn", "conv_pw_2_relu", "conv_dw_3", "conv_dw_3_bn", "conv_dw_3_relu", "conv_pw_3", "conv_pw_3_bn", "conv_pw_3_relu", "conv_dw_4", "conv_dw_4_bn", "conv_dw_4_relu", "conv_pw_4", "conv_pw_4_bn", "conv_pw_4_relu", "conv_dw_5", "conv_dw_5_bn", "conv_dw_5_relu", "conv_pw_5", "conv_pw_5_bn", "conv_pw_5_relu", "conv_dw_6", "conv_dw_6_bn", "conv_dw_6_relu", "conv_pw_6", "conv_pw_6_bn", "conv_pw_6_relu", "conv_dw_7", "conv_dw_7_bn", "conv_dw_7_relu", "conv_pw_7", "conv_pw_7_bn", "conv_pw_7_relu", "conv_dw_8", "conv_dw_8_bn", "conv_dw_8_relu", "conv_pw_8", "conv_pw_8_bn", "conv_pw_8_relu", "conv_dw_9", "conv_dw_9_bn", "conv_dw_9_relu", "conv_pw_9", "conv_pw_9_bn", "conv_pw_9_relu", "conv_dw_10", "conv_dw_10_bn", "conv_dw_10_relu", "conv_pw_10", "conv_pw_10_bn", "conv_pw_10_relu", "conv_dw_11", "conv_dw_11_bn", "conv_dw_11_relu", "conv_pw_11", "conv_pw_11_bn", "conv_pw_11_relu", "conv_dw_12", "conv_dw_12_bn", "conv_dw_12_relu", "conv_pw_12", "conv_pw_12_bn", "conv_pw_12_relu", "conv_dw_13", "conv_dw_13_bn", "conv_dw_13_relu", "conv_pw_13", "conv_pw_13_bn", "conv_pw_13_relu", "global_average_pooling2d_1", "reshape_1", "dropout", "conv_preds", "act_softmax", "reshape_2"]

mymodel.infer(canvas, "conv_preds")

e {isDisposedInternal: false, size: 1000, shape: Array(4), dtype: "float32", strides: Array(3), …}
dataId: {}
dtype: "float32"
id: 1269
isDisposed: (...)
isDisposedInternal: false
rank: (...)
rankType: "4"
shape: (4) [1, 1, 1, 1000]
size: 1000
strides: (3) [1000, 1000, 1000]
__proto__: Object

maybe here ("unfrozen")
https://github.com/tensorflow/tfjs-examples/blob/master/simple-object-detection/train.js
buildNewHead()

---
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

