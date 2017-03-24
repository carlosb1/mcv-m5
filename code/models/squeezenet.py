#Keras imports
# reference: https://github.com/rcmalli/keras-squeezenet
from keras.models import Model
#from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation,  Dropout, GlobalAveragePooling2D, warnings, merge
#from keras.layers import concatenate
#from keras.engine.topology import get_source_inputs
#from keras.utils import get_file
#from keras.utils import layer_utils

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

WEIGHTS_PATH = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5"

def fire_module(x, fire_id, squeeze=16, expand=64):
    x = Convolution2D(squeeze, 1, 1, border_mode='valid')(x)
    x = Activation('relu')(x)

    left = Convolution2D(expand, 1, 1, border_mode='valid')(x)
    left = Activation('relu')(left)

    right = Convolution2D(expand, 3, 3, border_mode='same')(x)
    right = Activation('relu')(right)

    #x = K.concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    x = merge([left, right], concat_axis=1)
    return x


def build_squeezenet(img_shape=(3, 224, 224), n_classes=1000, n_layers=16, l2_reg=0.,
                load_pretrained=False, freeze_layers_from='base_model'):
	
    if load_pretrained:
        weights = 'imagenet'
    else:
        weights = None

    img_input = Input(shape=img_shape)

# Modular function for Fire Node

    x = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='valid')(img_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3),  strides=(2, 2))(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x = Dropout(0.5)(x)

    x = Convolution2D(n_classes, 1, 1, border_mode='valid')(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax')(x)
    
    model = Model(input=img_input,output=out)

    # load weights
    if weights == 'imagenet':
        weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        model.load_weights(weights_path)
    
    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == 'base_model':
            print ('   Freezing base model layers')
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for i, layer in enumerate(model.layers):
                print(i, layer.name)
            print ('   Freezing from layer 0 to ' + str(freeze_layers_from))
            for layer in model.layers[:freeze_layers_from]:
               layer.trainable = False
            for layer in model.layers[freeze_layers_from:]:
               layer.trainable = True
			      
    return model



