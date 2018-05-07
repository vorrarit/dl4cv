from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
import tensorflow as tf
from keras import backend as K

class ShallowNet:
	@staticmethod
	def build(width, height, depth, classes):
		config = tf.ConfigProto(intra_op_parallelism_threads=1,
		                        inter_op_parallelism_threads=1, allow_soft_placement=True, device_count={'CPU': 1}
								)
		session = tf.Session(config=config)
		K.set_session(session)
		
		model = Sequential()
		inputShape = (height, width, depth)

		if (K.image_data_format() == 'channels_first'):
			inputShape = (depth, height, width)

		model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(Flatten())
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model
