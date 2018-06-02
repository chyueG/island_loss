# island_loss
Island loss implement by tensorflow
Refer to the paper "Island Loss for Learning Discriminative Features in Facial Expression"
The basic code is referred to https://github.com/EncodeTS/TensorFlow_Center_Loss,
besides I read the relative part in https://github.com/davidsandberg/facenet/blob/master/src/facenet.py#L76-L88,
if i use the latter one which use tensorflow auto grad computation,the center points in tensorboard will appear only
one point while i use the tf.control_depencies[update_op] in the primer ,the center points will appear several points 
correspond to the center points' numbers.The picture will be attached latter.
About the center point initializer,the center loss uses zeros_initializer,but it will cause grad NAN in island loss,
so gaussain initializer instead of the original one.
About the converage,more tests need to be tried.
