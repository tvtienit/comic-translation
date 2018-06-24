import tensorflow as tf
import embedding.constant as constant

flags = tf.app.flags
FLAGS = flags.FLAGS

# Define flags
flags.DEFINE_string('output_dir', constant.LOCAL_OUTPUT, 'Output Directory.')
flags.DEFINE_string('input_dir', constant.LOCAL_INPUT, 'Input Directory.')
flags.DEFINE_integer('num_steps', constant.DEFAULT_NUM_STEP, 'Number of training steps')
flags.DEFINE_boolean('gs', False, 'Google Cloud Storage.')

def __dict__(name):
	return FLAGS.FlagsByModuleDict(name)

def get_flag_value(name):
	if __dict__(name) is not None:
		return __dict__(name).value
	return None

def check_flag_value(name, value):
	if __dict__(name) is not None and __dict__(name).value == value:
		return True
	return False