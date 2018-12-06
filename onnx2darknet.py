from collections import OrderedDict
import sys
import numpy as np
import onnx
from array import array
from pprint import pprint

def onnx2darknet(onnxfile):

    # Load the ONNX model
	model = onnx.load(onnxfile)

    # Check that the IR is well formed
	onnx.checker.check_model(model)

    # Print a human readable representation of the graph
	print(onnx.helper.printable_graph(model.graph))
	
	#onnx -> darknet convert
	jj=0
	kk=0
	i= 0
	end = 0
	layer_num = 0
	act_layer = []
	k=0
	wdata = []
	blocks = []
	block = OrderedDict()
	block['type'] = 'net'
	block['batch'] = 1
	block['channels'] = 3
	block['height'] = 416
	block['width'] = 416
	blocks.append(block)
	
	while i <  len(model.graph.node):
		layer = model.graph.node[i]
		if layer.op_type == 'Conv':
		
			#[route] layer => attribute 1
			if  int( layer.input[0]) !=1 and act_layer.index( int(layer.input[0]))  - len(act_layer) +1 < 0:
				block = OrderedDict()
				block['type'] = 'route'
				block['layers'] =  act_layer.index( int(layer.input[0]))  - len(act_layer)
				blocks.append(block)
				act_layer.append(int(layer.output[0]))
				
			block = OrderedDict()
			block['type'] = 'convolutional'
			
			#Input informations => filters
			input_num = layer.input[1]
			block['filters'] = model.graph.input[int(input_num)].type.tensor_type.shape.dim[0].dim_value
			
			j=0
			while j < len(layer.attribute):
  		  		#kernel_shape => size
				if layer.attribute[j].name == 'kernel_shape':
					block['size'] = layer.attribute[j].ints[0]
					j = j+1
			
		  		#strides => stride		    
				elif layer.attribute[j].name == 'strides':
					block['stride'] = '1'
					j = j+1
			
		  		#pads => pad
				elif layer.attribute[j].name == 'pads':
					block['pad'] ='1'
					j = j+1
				
				else:
					#blocks.append("<unknown>")
					j = j+1
								
			i = i + 1
		  
		elif layer.op_type == 'BatchNormalization':
			#is_test => batch_normalize
			if layer.attribute[0].name == 'is_test':
				block['batch_normalize'] = '1'

				kk = kk + 5
				while jj < len(model.graph.initializer[kk-3].raw_data):
					wdata += list(array('f',model.graph.initializer[kk-3].raw_data[jj:jj+4]))
					jj = jj + 4
				jj = 0
				while jj < len(model.graph.initializer[kk-4].raw_data):
					wdata += list(array('f',model.graph.initializer[kk-4].raw_data[jj:jj+4]))
					jj = jj + 4
				jj = 0
				while jj < len(model.graph.initializer[kk-2].raw_data):
					wdata += list(array('f',model.graph.initializer[kk-2].raw_data[jj:jj+4]))
					jj = jj + 4
				jj = 0
				while jj < len(model.graph.initializer[kk-1].raw_data):
					wdata += list(array('f',model.graph.initializer[kk-1].raw_data[jj:jj+4]))
					jj = jj + 4
				jj = 0
				while jj < len(model.graph.initializer[kk-5].raw_data):
					wdata += list(array('f',model.graph.initializer[kk-5].raw_data[jj:jj+4]))
					jj = jj + 4
				jj = 0
			i = i + 1
		
		elif layer.op_type == 'LeakyRelu':
			#LeakyRelu => activation=leaky
			block['activation'] = 'leaky'
			blocks.append(block)
			i = i + 1
			act_layer.append(int(layer.output[0]))

		elif layer.op_type == 'Add':
			#LeakyRelu => activation=linear 
			block['activation'] = 'linear '
			blocks.append(block)
			
			kk = kk + 1
			while jj < len(model.graph.initializer[kk].raw_data):
				wdata += list(array('f',model.graph.initializer[kk].raw_data[jj:jj+4]))
				jj = jj + 4
			jj = 0
			while jj < len(model.graph.initializer[kk-1].raw_data):
				wdata += list(array('f',model.graph.initializer[kk-1].raw_data[jj:jj+4]))
				jj = jj + 4
			jj = 0
						
			i = i + 1
			
########################################################		
		
		elif layer.op_type == 'MaxPool':
			block = OrderedDict()
			block['type'] = 'maxpool'
			j = 0
			while j < len(layer.attribute):
		  		#kernel_shape => size
				if layer.attribute[j].name == 'kernel_shape':
					block['size'] = layer.attribute[j].ints[0]
					j = j + 1
		  		#strides => stride
				elif layer.attribute[j].name == 'strides':
					block['stride'] = layer.attribute[j].ints[0]
					blocks.append(block)
					j = j + 1
				else:
					j = j + 1
			i = i + 1
			
			act_layer.append(int(layer.output[0]))
		
########################################################		
		#Reshpae => reorg layer
		elif  layer.op_type == 'Reshape':
			if end == 0:
				block = OrderedDict()
				block['type'] = 'reorg'
				block['stride'] = '2'
				blocks.append(block)
				end = 1
			else:
				if(model.graph.node[i+1].op_type) == 'Transpose':
					end
				else:
					act_layer.append(int(layer.output[0]))
			i = i + 1
########################################################	
		
#		elif  layer.op_type == 'Transpose':
#			if layer['attribute'] == 'perm':	
		
########################################################	
		#Concat => [route] layer => attribute 2
		elif layer.op_type == 'Concat':		
			block = OrderedDict()
			block['type'] = 'route'
			first_num = act_layer.index( int(layer.input[0]))  - len(act_layer)
			last_num = act_layer.index( int(layer.input[1]))  - len(act_layer) 
			block['layers'] = str(first_num) + ',' + str(last_num)
			
			blocks.append(block)
			i = i + 1
			
			act_layer.append(int(layer.output[0]))		  
########################################################		

		else:
			i = i + 1

	
	block = OrderedDict()
	block['type'] = 'region'
	block['anchors'] = '0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828'
	block['bias_match']=1 
	block['classes']=80 
	block['coords']=4 
	block['num']=5 
	block['softmax']=1 
	block['jitter']=.3 
	block['rescore']=1 
	block['object_scale']=5 
	block['noobject_scale']=1 
	block['class_scale']=1 
	block['coord_scale']=1 
	block['absolute']=1 
	block['thresh'] =' .6' 
	block['random']=1 

	blocks.append(block)
	
	return blocks, np.array(wdata)

def save_cfg(blocks, cfgfile):
	print ('Save to ', cfgfile)
	with open(cfgfile, 'w') as fp:
		for block in blocks:
			fp.write('[%s]\n' % (block['type']))
			for key,value in block.items():
				if key != 'type':
					fp.write('%s=%s\n' % (key, value))
			fp.write('\n')


def save_weights(data, weightfile):
	#onnx weights -> darknet weights
	print ('Save to ', weightfile)
	wsize = data.size
	weights = np.zeros((wsize+4,), dtype=np.int32)
    ## write info 
	weights[0] = 0 	   ## major version
	weights[1] = 1 	   ## minor version
	weights[2] = 0      ## revision
	weights[3] = 0      ## net.seen
	weights.tofile(weightfile)
	weights = np.fromfile(weightfile, dtype=np.float32)	
	weights[4:] = data
	weights.tofile(weightfile)

if __name__ == '__main__':
	import sys
    
	if len(sys.argv) != 4:
		print('try:')
		print('python onnx2darknet.py yolov2.onnx yolov2.cfg yolov2.weights')
		exit()
	onnxfile = sys.argv[1]
	cfgfile = sys.argv[2]
	weightfile = sys.argv[3]

	blocks, data = onnx2darknet(onnxfile)
	
	save_cfg(blocks, cfgfile)
	save_weights(data, weightfile)
	
