import sys
import time
import numpy as np
import tensorflow as tf
import cv2


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,input_mean=0, input_std=255):
	with tf.Graph().as_default():
  		input_name = "file_reader"
  		output_name = "normalized"
  		#file_reader = tf.read_file(file_name, input_name)
  		#image_reader = tf.image.decode_jpeg(file_name, channels = 3,name='jpeg_reader')
  		#image_reader = np.ndarray(file_name, channels = 3,name='jpeg_reader')
  		float_caster = tf.cast(file_name, tf.float32)
  		dims_expander = tf.expand_dims(float_caster, 0);
  		resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  		normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  		sess = tf.Session()
  		result = sess.run(normalized)
  		return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def test():
	paused = False
	cap = cv2.VideoCapture(0)
	if (cap.isOpened()==False):
		print('error cam is broken')


	file_name = "img0.jpg"
	model_file = "output_graph.pb"
	label_file = "output_labels.txt"
	input_height = 224
	input_width = 224
	input_mean = 128
	input_std = 128
	input_layer = "input"
	output_layer = "final_result"

	graph = load_graph(model_file)
	input_name = "import/" + input_layer
	output_name = "import/" + output_layer
	input_operation = graph.get_operation_by_name(input_name);
	output_operation = graph.get_operation_by_name(output_name);
	labels = load_labels(label_file)
	ret, frame = cap.read()

	t = read_tensor_from_image_file(frame,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)
	

	with tf.Session(graph=graph) as sess:
		while (cap.isOpened()):
			ret, frame = cap.read()
			cv2.imshow('frame',frame)
			if paused==False :
				
				
				t = read_tensor_from_image_file(frame,input_height=input_height,input_width=input_width,input_mean=input_mean,input_std=input_std)
				start = time.time()
				results = sess.run(output_operation.outputs[0],{input_operation.outputs[0]: t})
				end=time.time()
				results = np.squeeze(results)
				top_k = results.argsort()[-5:][::-1]
				print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))

				for i in top_k:
					print(labels[i], results[i])
			#cv2.imshow('frame',frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			if cv2.waitKey(1) & 0xFF == ord('x'):
				paused = not paused
				if (paused==False):
					paused==True
					print('Paused!')
				else:
					paused==False
					print('Unpaused!')
					time.sleep(0.2)

	cap.release()
	cv2.destroyAllWindows()
