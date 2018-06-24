import tensorflow as tf
import constant

flags = tf.app.flags
FLAGS = flags.FLAGS

# Define flags
flags.DEFINE_string('output_dir', constant.LOCAL_OUTPUT, 'Output Directory.')
flags.DEFINE_string('input_dir', constant.LOCAL_INPUT, 'Input Directory.')
flags.DEFINE_integer('num_step', constant.DEFAULT_NUM_STEP, 'Number of training steps')
flags.DEFINE_boolean('gs', False, 'Google Cloud Storage.')

def get_flag_value(name):
	if FLAGS[name] is not None:
		return FLAGS[name].value
	return None

def check_flag_value(name, value)
	if FLAGS[name] is not None and FLAGS[name].value == value:
		return True
	return False