import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D,Conv2D,Flatten,Dense,SeparableConv2D
from tensorflow.keras.layers import BatchNormalization,Activation,Reshape
from tensorflow import keras

class DepthLocalization(tf.keras.layers.Layer):
    def __init__(self, 
                 filters_1: int, 
                 filters_2: int, 
                 fc_units: int, 
                 kernel_size=(3,3),
                 pool_size=(2,2),
                 name='depthlocalization', 
                 **kwargs):
        super(DepthLocalization, self).__init__(**kwargs)
        self.filters_1 = filters_1
        self.filters_2 = filters_2
        self.fc_units = fc_units
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.network = keras.Sequential(
            [
                MaxPooling2D(pool_size=pool_size, name=name+'_mp_1_depth'),
                Conv2D(filters=filters_1, 
                       kernel_size=kernel_size, 
                       padding='same', 
                       strides=1,
                       kernel_initializer="he_normal", 
                       name=name+'_c_1'),
                # SeparableConv2D(filters=filters_1, 
                #        kernel_size=kernel_size, 
                #        padding='same', 
                #        strides=1,
                #        depthwise_initializer="he_normal", 
                #        pointwise_initializer="he_normal",
                #        name=name+'_c_1_depth'),
                BatchNormalization(axis=-1),
                Activation('relu'),
                MaxPooling2D(pool_size=pool_size, name=name+'_mp_2_depth'),
                Conv2D(filters=filters_2, 
                       kernel_size=kernel_size, 
                       padding='same', 
                       strides=1, 
                       kernel_initializer="he_normal",
                       name=name+'_c_2'),
                # SeparableConv2D(filters=filters_2, 
                #        kernel_size=kernel_size, 
                #        padding='same', 
                #        strides=1,
                #        depthwise_initializer="he_normal", 
                #        pointwise_initializer="he_normal",
                #        name=name+'_c_2_depth'),
                BatchNormalization(axis=-1),
                Activation('relu'),
                MaxPooling2D(pool_size=pool_size, name=name+'_mp_3_depth'),
                Flatten(name=name+'_fl_depth'),
                Dense(fc_units, activation='relu',kernel_initializer="he_normal", name=name+'_d_1_depth'),
                Dense(6, activation=None, 
                      bias_initializer=tf.keras.initializers.constant([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), 
                      kernel_initializer='zeros',
                      name=name+'_d_2_depth'),
            ]
        )

    def build(self, input_shape):
        pass

    def compute_output_shape(self, input_shape):
        return [None, 6]

    def call(self, inputs):
        theta = self.network(inputs)
        theta = Reshape((2, 3))(theta)
        return theta

    def get_config(self):
        config = super(DepthLocalization, self).get_config()
        config.update({
            'filters_1': self.filters_1,
            'filters_2': self.filters_2,
            'fc_units': self.fc_units,
            'kernel_size': self.kernel_size,
            'pool_size': self.pool_size,
        })
        return config
     
class DepthBilinearInterpolation(tf.keras.layers.Layer):
    def __init__(self, height=48, width=48,name='depthbilinlearinterpolation'):
        super(DepthBilinearInterpolation, self).__init__()
        self.height = height
        self.width = width

    def compute_output_shape(self, input_shape):
        return [None, self.height, self.width, 1]

    def get_config(self):
        return {
            'height': self.height,
            'width': self.width,
        }
    
    def build(self, input_shape):
        pass

    def advance_indexing(self, inputs, x, y):
        '''
        Utility function to get pixel value for coordinate
        vectors x and y from a  4D tensor image.
        '''        
        shape = tf.shape(inputs)
        batch_size, _, _ = shape[0], shape[1], shape[2]
        
        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, self.height, self.width))
        indices = tf.stack([b, y, x], 3)
        return tf.gather_nd(inputs, indices)

    def call(self, inputs):
        images, theta = inputs
        sampling_grid = self.grid_generator(batch=tf.shape(images)[0])
        return self.interpolate(images, sampling_grid, theta)

    def grid_generator(self, batch):
        '''
        This function returns a sampling grid, which when
        used with the bilinear sampler on the input feature
        map, will create an output feature map that is an
        affine transformation of the input feature map.
        '''
        # create normalized 2D grid
        x = tf.linspace(-1, 1, self.width)
        y = tf.linspace(-1, 1, self.height)
        # x and y are selected in the range of -1 to 1 so the the transformation happens considering the center
        # of the image as the origin. The images will be later scaled up.
        xx, yy = tf.meshgrid(x, y)
            
        # flatten
        xx = tf.reshape(xx, (-1,))
        yy = tf.reshape(yy, (-1,))

        # reshape to [x_t, y_t , 1] - (homogeneous form)
        homogenous_coordinates = tf.stack([xx, yy, tf.ones_like(xx)])
        # # repeat grid num_batch times
        # sampling_grid = np.resize(sampling_grid, (num_batch, 3, H*W))
        # repeat grid num_batch times
        homogenous_coordinates = tf.expand_dims(homogenous_coordinates, axis=0)
        homogenous_coordinates = tf.tile(homogenous_coordinates, tf.stack([batch, 1, 1]))
        # homogenous_coordinates = tf.tile(homogenous_coordinates, [batch, 1, 1])

        # cast to float32 (required for matmul)
        homogenous_coordinates = tf.cast(homogenous_coordinates, dtype=tf.float32)

        return homogenous_coordinates
    
    def interpolate(self, images, grid, theta):
        '''
        Performs bilinear sampling of the input images according to the
        normalized coordinates provided by the sampling grid. Note that
        the sampling is done identically for each channel of the input.
        '''

        with tf.name_scope("Transformation"):
            # transform the sampling grid - batch multiply
            transformed = tf.matmul(theta, grid)
            # batch grid has shape (num_batch, 2, H*W)

            # reshape to (num_batch, H, W, 2)
            transformed = tf.transpose(transformed, perm=[0, 2, 1])
            transformed = tf.reshape(transformed, [-1, self.height, self.width, 2])
                
            x_transformed = transformed[:, :, :, 0]
            y_transformed = transformed[:, :, :, 1]
                
            # rescale x and y to [0, W-1/H-1]
            x = ((x_transformed + 1.) * tf.cast(self.width, dtype=tf.float32)) * 0.5
            y = ((y_transformed + 1.) * tf.cast(self.height, dtype=tf.float32)) * 0.5

        with tf.name_scope("VariableCasting"):
            # grab 4 nearest corner points for each (x_i, y_i)
            x0 = tf.cast(tf.math.floor(x), dtype=tf.int32)
            x1 = x0 + 1
            y0 = tf.cast(tf.math.floor(y), dtype=tf.int32)
            y1 = y0 + 1

            # clip to range [0, H-1/W-1] to not violate img boundaries
            x0 = tf.clip_by_value(x0, 0, self.width-1)
            x1 = tf.clip_by_value(x1, 0, self.width-1)
            y0 = tf.clip_by_value(y0, 0, self.height-1)
            y1 = tf.clip_by_value(y1, 0, self.height-1)
            x = tf.clip_by_value(x, 0, tf.cast(self.width, dtype=tf.float32)-1.0)
            y = tf.clip_by_value(y, 0, tf.cast(self.height, dtype=tf.float32)-1)

        with tf.name_scope("AdvanceIndexing"):
            # get pixel value at corner coords
            Ia = self.advance_indexing(images, x0, y0)
            Ib = self.advance_indexing(images, x0, y1)
            Ic = self.advance_indexing(images, x1, y0)
            Id = self.advance_indexing(images, x1, y1)

        with tf.name_scope("Interpolation"):
            # recast as float for delta calculation
            x0 = tf.cast(x0, dtype=tf.float32)
            x1 = tf.cast(x1, dtype=tf.float32)
            y0 = tf.cast(y0, dtype=tf.float32)
            y1 = tf.cast(y1, dtype=tf.float32)
                            
            # calculate deltas
            wa = (x1-x) * (y1-y)
            wb = (x1-x) * (y-y0)
            wc = (x-x0) * (y1-y)
            wd = (x-x0) * (y-y0)

            # add dimension for addition
            wa = tf.expand_dims(wa, axis=3)
            wb = tf.expand_dims(wb, axis=3)
            wc = tf.expand_dims(wc, axis=3)
            wd = tf.expand_dims(wd, axis=3)
                        
        return tf.math.add_n([wa*Ia + wb*Ib + wc*Ic + wd*Id])

def depthwise_stn(x,dims,kernel_size=(3,3),stage=1):
    n_f1,n_f2,n_units = dims
    theta = DepthLocalization(n_f1,n_f2,n_units,kernel_size=kernel_size,name='depthlocalization_stage_'+str(stage))(x)
    h,w = [x.shape[1],x.shape[2]]
    return DepthBilinearInterpolation(height=h,width=w,name='depthbilinearinterpolation_stage_'+str(stage))([x, theta]) 