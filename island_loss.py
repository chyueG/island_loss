import tensorflow as tf
import itertools
import numpy as np

def get_center_loss(features, labels, alpha,alpha1, num_classes):

    len_features = features.get_shape()[1]

    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)

    labels = tf.reshape(labels, [-1])

    centers_batch = tf.gather(centers, labels)

    loss_part1 = tf.nn.l2_loss(features - centers_batch)

    ####add new code
    index = np.arange(num_classes)
    combination = itertools.permutations(index,2)

    pairwise_grad_val = {}
    for idx,item in enumerate(combination):
        index = idx/(num_classes-1)
        lc_grad = pairwise_grad(fea,item)
        if idx%(num_classes-1) == 0:
            if index in cate:
                pairwise_grad_val[index] += lc_grad
            else:
                pairwise_grad_val[index] = lc_grad

    grad_pairwise = pairwise_grad_val[0]
    for idx in range(1,num_classes):
        grad_pairwise = tf.stack([grad_pairwise,pairwise_grad_val[idx]])

    grad_pairwsise_batch = tf.gather(grad_pairwise,labels)



    #pair_distance_loss,pairwise_grad
    pair_distance_loss = [].append(map(lambda x:pairwise_distance(centers_batch,x),combination))

    pair_distance_loss = tf.reduce_sum(pair_distance_loss)

    loss = loss_part1 + alpha1*tf.reduce_sum(pair_distance_loss)

    ####new code end

    diff = centers_batch - features

    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = diff + alpha1*(num_classes-1)*grad_pairwsise_batch

    diff = alpha * diff



    centers_update_op = tf.scatter_sub(centers, labels, diff)

    return loss, centers, centers_update_op



def pairwise_distance(fea,com):
    dims = len(fea.get_shape())
    if dims == 2:
        fea_k = fea[com[0],:]
        fea_j = fea[com[1],:]
    else:
        print ("Please check the feature dimensions")
        return
    k_l2norm = tf.nn.l2_normalize(fea_k)
    j_l2norm = tf.nn.l2_normalize(fea_j)
    loss_term = k_l2norm*j_l2norm + 1
    grad_term = k_l2norm/tf.norm(fea_j,ord=2) - k_l2norm*tf.square(j_l2norm)/tf.norm(fea_j,ord=2)
    return loss_term,grad_term

def pairwise_grad(fea,com):
    dims = len(fea.get_shape())
    if dims == 2:
        fea_k = fea[com[0],:]
        fea_j = fea[com[1],:]
    else:
        print ("Please check the feature dimensions")
        return
    k_l2norm = tf.nn.l2_normalize(fea_k)
    j_l2norm = tf.nn.l2_normalize(fea_j)
    grad_term = k_l2norm/tf.norm(fea_j,ord=2) - k_l2norm*tf.square(j_l2norm)/tf.norm(fea_j,ord=2)
    return grad_term


if __name__ == "__main__":
    feature = tf.placeholder(dtype=tf.float32,shape=[64,1024],name="feature")
    label   = tf.placeholder(dtype=tf.int32,shape=[64],name="label")
    alpha   = 0.4
    alpha2  = 0.5
    num_classes = 7
    get_center_loss(feature,label,alpha,alpha2,num_classes)
