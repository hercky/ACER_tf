# a sample net 
import numpy as np
import tensorflow as tf
import sonnet as snt




class AdvantageValueNet(snt.AbstractModule):
    """
    to model  f(.|\phi_{\theta'}(x_i))
    parameters denoted by \theta'
    
    Input: the observed state, x
    
    Output: a distribution over action ( a normal distribution with fixed diagonal co-variance)
    
    """
    def __init__(self, 
                 hidden_size,
                 val_layer_size = 1, # just one flly connected layer for now
                 adv_layer_size = 1,
                 n = 5,
                 name="adv_val_net"):
        """
        hidden_size = size of the neural net
        output size = dimensionality of the action space
        n: no of samples in u_n
        """
        super(AdvantageValueNet, self).__init__(name=name)
        self._hidden_size = hidden_size
        self._val_layer_size = val_layer_size
        self._adv_layer_size = adv_layer_size
        self._n = n
        self._name = name
        
    def _build(self, x_t, a_t, u):
        """
        x_t: shape: 1 x INPUT_DIM
        a_t: shape:
        u: input form policy network, returns a distribution
        n: no of samples
        """        
        feature_layer = snt.Linear(self._hidden_size, name="x_feature_layer")
        val_layer = snt.Linear(self._val_layer_size, name="val_layer")
        adv_layer = snt.Linear(self._adv_layer_size, name="adv_layer")
        
        # get the shared feature / can be removed and taken directly as input 
        phi_x = tf.nn.relu(feature_layer(x_t))
        
        # value function estimate
        V_x = val_layer(phi_x)
        
        # sample action from policy distribution 
        u_n = u.sample([ self._n ])
        
        #  get A(x_t,a_t)
        xa = tf.concat([phi_x, a_t], 1) # merge columns
        
        A_xa = adv_layer(xa)
        
        # get \Sum[A(x_t,u_i)]
        copy_phi_x = tf.tile(tf.reshape(phi_x, [1, -1 ,self._hidden_size]),[self._n,1,1])
        
        xu = tf.concat([copy_phi_x,u_n] , 2)
        
        # apply batch apply here now
        A_xu_i = snt.BatchApply(adv_layer)(xu)
        
        # reshape to batch x val x n 
        A_xu_i = tf.reshape(A_xu_i, [-1, self._n])
        
        A_xu_mean = tf.reduce_mean(A_xu_i, 1, keep_dims=True)
        
        Q_xa = V_x + A_xa - A_xu_mean
        
        return V_x, A_xa
    
    def update_local_params_op(self, target_name):
        """
        to copy the ops within a net
        
        returns the copy op
        """
        t_vars = tf.trainable_variables()
        
        local_vars = [var for var in t_vars if self._name in var.name]
        target_vars = [var for var in t_vars if target_name in var.name]
        
        copy_op = []
        
        for l_var,t_var in zip(local_vars,target_vars):
            copy_op.append(l_var.assign(t_var))

        return copy_op
    
    def local_params(self):
        """
        return the local params
        """
        t_vars = tf.trainable_variables()
        
        local_vars = [var for var in t_vars if self._name in var.name]
        
        return local_vars
