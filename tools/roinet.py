import caffe
from caffe import layers as L, params as P, to_proto
from caffe.coord_map import crop # crop(net.upscore, net.data) automatically calculate the axis and offset

def bn_relu_conv(bottom, num_output, pad=0, kernel_size=3, stride=1, update_param=True):
	if update_param:
		bn = L.BatchNorm(bottom, in_place=False, use_global_stats=use_global_stats, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
		scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
		relu = L.ReLU(scale, in_place=True, engine=engine)
		conv = L.Convolution(relu, num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
			param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],		
			weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
			engine=engine)
	else:
		bn = L.BatchNorm(bottom, in_place=False, use_global_stats=True, moving_average_fraction=1, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], propagate_down=0)
		scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0), param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], propagate_down=0)
		relu = L.ReLU(scale, in_place=True, engine=engine, propagate_down=0)
		conv = L.Convolution(relu, num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],		
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=engine, propagate_down=0)
	return bn, scale, relu, conv

def conv_bn(bottom, num_output, pad=0, kernel_size=3, stride=1, update_param=True):
	if update_param:
		conv = L.Convolution(bottom, num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
			param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],		
			weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
			engine=engine)
		bn = L.BatchNorm(conv, in_place=True, use_global_stats=use_global_stats, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
		scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	else:
		conv = L.Convolution(bottom, num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
			param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],		
			weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
			engine=engine, propagate_down=0)
		bn = L.BatchNorm(conv, in_place=True, use_global_stats=True, moving_average_fraction=1, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], propagate_down=0)
		scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0), param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], propagate_down=0)
	return conv, bn, scale

def conv_bn_relu(bottom, num_output, pad=0, kernel_size=3, stride=1, update_param=True):
	if update_param:
		conv = L.Convolution(bottom, num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
			param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],		
			weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
			engine=engine)
		bn = L.BatchNorm(conv, in_place=True, use_global_stats=use_global_stats, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
		scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
		relu = L.ReLU(scale, in_place=True, engine=engine)
	else:
		conv = L.Convolution(bottom, num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
			param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],		
			weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
			engine=engine, propagate_down=0)
		bn = L.BatchNorm(conv, in_place=True, use_global_stats=True, moving_average_fraction=1, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], propagate_down=0)
		scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0), param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], propagate_down=0)
		relu = L.ReLU(scale, in_place=True, engine=engine, propagate_down=0)
	return conv, bn, scale, relu

def deconv_bn_relu(bottom, num_output, pad=0, kernel_size=3, stride=1, update_param=True):
	if update_param:
		deconv = L.Deconvolution(bottom, param=[dict(lr_mult=1, decay_mult=1)],
			convolution_param=dict(num_output=num_output,  pad=pad, kernel_size=kernel_size, stride=stride,
				weight_filler=dict(type='msra'), bias_term=0,
				engine=engine))
		bn = L.BatchNorm(deconv, in_place=True, use_global_stats=use_global_stats, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
		scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
		relu = L.ReLU(scale, in_place=True, engine=engine)
	else:
		deconv = L.Deconvolution(bottom, param=[dict(lr_mult=0, decay_mult=0)],
			convolution_param=dict(num_output=num_output,  pad=pad, kernel_size=kernel_size, stride=stride,
				weight_filler=dict(type='msra'), bias_term=0,
				engine=engine), propagate_down=0)
		bn = L.BatchNorm(deconv, in_place=True, use_global_stats=True, moving_average_fraction=1, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], propagate_down=0)
		scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0), param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], propagate_down=0)
		relu = L.ReLU(scale, in_place=True, engine=engine, propagate_down=0)
	return deconv, bn, scale, relu

def add_layer(bottom1, bottom2, num_output, update_param=True):
	conv, bn, scale = conv_bn(bottom1, num_output=num_output, pad=0, kernel_size=1, stride=1, update_param=update_param)
	if update_param:
		eltw = L.Eltwise(conv, bottom2, eltwise_param=dict(operation=1))
		rule = L.ReLU(eltw, in_place=True, engine=engine)
	else:
		eltw = L.Eltwise(conv, bottom2, eltwise_param=dict(operation=1), propagate_down=[0,0])
		rule = L.ReLU(eltw, in_place=True, engine=engine, propagate_down=0)
	return conv, bn, scale, eltw, rule

def add_relu(bottom1, bottom2, update_param=True):
	if update_param:
		eltw = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1))
		rule = L.ReLU(eltw, in_place=True, engine=engine)
	else:
		eltw = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1), propagate_down=[0,0])
		rule = L.ReLU(eltw, in_place=True, engine=engine, propagate_down=0)
	return eltw, rule

def conv1_conv2_add_bn_relu(bottom1, bottom2, num_output, pad=0, kernel_size=3, stride=1, update_param=True):
	if update_param:
		conv1 = L.Convolution(bottom1, num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
			param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],		
			weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
			engine=engine)
		conv2 = L.Convolution(bottom2, num_output=num_output, pad=0, kernel_size=1, stride=1,
			param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],		
			weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
			engine=engine)
		eltw = L.Eltwise(conv1, conv2, eltwise_param=dict(operation=1))
		bn = L.BatchNorm(eltw, in_place=True, use_global_stats=use_global_stats, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
		scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
		relu = L.ReLU(scale, in_place=True, engine=engine)
	else:
		conv1 = L.Convolution(bottom1, num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
			param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],		
			weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
			engine=engine, propagate_down=0)
		conv2 = L.Convolution(bottom2, num_output=num_output, pad=0, kernel_size=1, stride=1,
			param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],		
			weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
			engine=engine, propagate_down=0)
		eltw = L.Eltwise(conv1, conv2, eltwise_param=dict(operation=1), propagate_down=[0,0])
		bn = L.BatchNorm(eltw, in_place=True, use_global_stats=True, moving_average_fraction=1, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], propagate_down=0)
		scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0), param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], propagate_down=0)
		relu = L.ReLU(scale, in_place=True, engine=engine, propagate_down=0)
	
	return conv1, conv2, eltw, bn, scale, relu

def pool_conv_bn_relu_concat(bottom, num_output, update_param=True):
	if update_param:
		pool = L.Pooling(bottom, pool=P.Pooling.MAX, pad=0, kernel_size=2, stride=2, engine=engine)
		conv, bn, scale, relu = conv_bn_relu(bottom, num_output, pad=1, kernel_size=3, stride=2, update_param=update_param)
		concat = L.Concat(pool, conv, axis=1)
	else:
		pool = L.Pooling(bottom, pool=P.Pooling.MAX, pad=0, kernel_size=2, stride=2, engine=engine, propagate_down=0)
		conv, bn, scale, relu = conv_bn_relu(bottom, num_output, pad=1, kernel_size=3, stride=2, update_param=update_param)
		concat = L.Concat(pool, conv, axis=1, propagate_down=[0,0])

	return pool, conv, bn, scale, relu, concat

def max_pool(bottom, pad=0, kernel_size=2, stride=2, update_param=True):
	if update_param:
		return L.Pooling(bottom, pool=P.Pooling.MAX, pad=pad, kernel_size=kernel_size, stride=stride, engine=engine)
	else:
		return L.Pooling(bottom, pool=P.Pooling.MAX, pad=pad, kernel_size=kernel_size, stride=stride, engine=engine, propagate_down=0)
############ ############


def roinet_2d_bn_u2(dim_data, dim_label, num_class, phase='train'):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))
	net.d0b_conv, net.d0b_bn, net.d0b_scale, net.d0b_relu = conv_bn_relu(net.data, 64, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d0c_conv, net.d0c_bn, net.d0c_scale, net.d0c_relu = conv_bn_relu(net.d0b_relu, 64, pad=1, kernel_size=3, stride=1, update_param=False)
	############ d1 ############
	# net.d1a_pool = max_pool(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1a_pool, net.d1a_conv, net.d1a_bn, net.d1a_scale, net.d1a_relu, net.d1a_concat = pool_conv_bn_relu_concat(net.d0c_relu, 64, update_param=False)
	net.d1b_conv, net.d1b_bn, net.d1b_scale, net.d1b_relu = conv_bn_relu(net.d1a_concat, 128, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d1c_conv, net.d1c_bn, net.d1c_scale = conv_bn(net.d1b_relu, 128, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d1c_conv, net.d1c_bn, net.d1c_scale, net.d1c_relu = conv_bn_relu(net.d1b_relu, 128, pad=1, kernel_size=3, stride=1)
	# net.d1d_conv, net.d1d_bn, net.d1d_scale, net.d1d_relu = conv_bn_relu(net.d1c_relu, 256, pad=1, kernel_size=3, stride=1)
	# net.d1e_eltw = L.Eltwise(net.d1a_concat, net.d1c_relu, eltwise_param=dict(operation=1))
	net.d1e_conv, net.d1e_bn, net.d1e_scale, net.d1e_eltw, net.d1e_relu = add_layer(net.d1a_concat, net.d1c_conv, 128, update_param=False)
	
	############ d2 ############
	# net.d2a_pool = max_pool(net.d1d_relu, pad=0, kernel_size=2, stride=2)
	net.d2a_pool, net.d2a_conv, net.d2a_bn, net.d2a_scale, net.d2a_relu, net.d2a_concat = pool_conv_bn_relu_concat(net.d1e_eltw, 128, update_param=False)
	net.d2b_conv, net.d2b_bn, net.d2b_scale, net.d2b_relu = conv_bn_relu(net.d2a_concat, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d2c_conv, net.d2c_bn, net.d2c_scale, net.d2c_relu = conv_bn_relu(net.d2b_relu, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d2d_conv, net.d2d_bn, net.d2d_scale, net.d2d_relu = conv_bn_relu(net.d2c_relu, 256, pad=1, kernel_size=3, stride=1)
	# net.d2e_eltw = L.Eltwise(net.d2a_concat, net.d2d_relu, eltwise_param=dict(operation=1))
	net.d2d_conv, net.d2d_bn, net.d2d_scale = conv_bn(net.d2c_conv, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d2e_conv, net.d2e_bn, net.d2e_scale, net.d2e_eltw, net.d2e_relu = add_layer(net.d2a_concat, net.d2d_conv, 256, update_param=False)
	
	############ d3 ############
	# net.d3a_pool = max_pool(net.d2d_relu, pad=0, kernel_size=2, stride=2)
	net.d3a_pool, net.d3a_conv, net.d3a_bn, net.d3a_scale, net.d3a_relu, net.d3a_concat = pool_conv_bn_relu_concat(net.d2e_eltw, 256, update_param=False)
	net.d3b_conv, net.d3b_bn, net.d3b_scale, net.d3b_relu = conv_bn_relu(net.d3a_concat, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d3c_conv, net.d3c_bn, net.d3c_scale, net.d3c_relu = conv_bn_relu(net.d3b_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d3d_conv, net.d3d_bn, net.d3d_scale, net.d3d_relu = conv_bn_relu(net.d3c_relu, 512, pad=1, kernel_size=3, stride=1)
	# net.d3e_eltw = L.Eltwise(net.d3a_concat, net.d3d_relu, eltwise_param=dict(operation=1))
	net.d3d_conv, net.d3d_bn, net.d3d_scale = conv_bn(net.d3c_conv, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d3e_conv, net.d3e_bn, net.d3e_scale, net.d3e_eltw, net.d3e_relu = add_layer(net.d3a_concat, net.d3d_conv, 512, update_param=False)
	
	############ d4 ############
	# net.d4a_pool = max_pool(net.d3d_relu, pad=0, kernel_size=2, stride=2)
	net.d4a_pool, net.d4a_conv, net.d4a_bn, net.d4a_scale, net.d4a_relu, net.d4a_concat = pool_conv_bn_relu_concat(net.d3e_eltw, 512, update_param=False)
	net.d4b_conv, net.d4b_bn, net.d4b_scale, net.d4b_relu = conv_bn_relu(net.d4a_concat, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d4c_conv, net.d4c_bn, net.d4c_scale, net.d4c_relu = conv_bn_relu(net.d4b_relu, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d4d_conv, net.d4d_bn, net.d4d_scale, net.d4d_relu = conv_bn_relu(net.d4c_relu, 1024, pad=1, kernel_size=3, stride=1)
	# net.d4e_eltw = L.Eltwise(net.d4a_concat, net.d4d_relu, eltwise_param=dict(operation=1))
	net.d4d_conv, net.d4d_bn, net.d4d_scale = conv_bn(net.d4c_conv, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d4e_conv, net.d4e_bn, net.d4e_scale, net.d4e_eltw, net.d4e_relu = add_layer(net.d4a_concat, net.d4d_conv, 1024, update_param=False)
	
	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_bn, net.u3a_scale, net.u3a_relu = deconv_bn_relu(net.d4e_eltw, 512, pad=0, kernel_size=2, stride=2, update_param=False)
	net.u3b_concat = L.Concat(net.u3a_relu, net.d3e_eltw, axis=1, propagate_down=[0,0])
	net.u3c_conv, net.u3c_bn, net.u3c_scale, net.u3c_relu = conv_bn_relu(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3d_conv, net.u3d_bn, net.u3d_scale, net.u3d_relu = conv_bn_relu(net.u3c_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3e_conv, net.u3e_bn, net.u3e_scale = conv_bn(net.u3d_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3f_conv, net.u3f_bn, net.u3f_scale, net.u3f_eltw, net.u3f_relu = add_layer(net.u3b_concat, net.u3e_conv, 512, update_param=False)
	############ score3 ###########
	net.score3 = L.Convolution(net.u3f_relu,
		param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine, propagate_down=0)
	net.upscore3 = L.Deconvolution(net.score3,
		param=[dict(lr_mult=0, decay_mult=0)],
		convolution_param=dict(num_output=num_class,  pad=4, kernel_size=16, stride=8,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine), propagate_down=0)
	# if phase == "train":
	# 	# net.loss3 = L.WeightedSoftmaxWithLoss(net.upscore3, net.label, net.label_weight,
	# 	net.loss3 = L.SoftmaxWithLoss(net.upscore3, net.label,
	# 		phase=0,
	# 		loss_weight=0.0625,
	# 		loss_param=dict(ignore_label=ignore_label))

	############ get roi ############
	net.prob3 = L.Softmax(net.upscore3, axis=1, propagate_down=0)
	net.roi_mask, net.roi_coordinate = L.ROICoordinate(net.prob3, pad=16, threshold=0.80, ntop=2, propagate_down=0)
	# net.slicence = L.Silence(net.roi_mask, ntop=0, propagate_down=0)
	if phase == "train":
		net.slicence = L.Silence(net.roi_mask, ntop=0, propagate_down=0)
	if phase == "train":
		# net.roi_label = L.ROIWarping(net.label, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
		net.label_map = L.ROIWarping(net.label, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
		net.roi_label = L.MultipleThreshold(net.label_map, threshold_point=[0.5,1.5], threshold_value=[0,1,2], propagate_down=0)

	############ u2 ############
	### a ### Second Deconvolution
	net.u2a_dconv, net.u2a_bn, net.u2a_scale, net.u2a_relu = deconv_bn_relu(net.u3f_relu, 256, pad=0, kernel_size=2, stride=2, update_param=False)
	net.u2b_concat = L.Concat(net.u2a_relu, net.d2e_eltw, axis=1, propagate_down=[0,0])
	net.u2c_conv, net.u2c_bn, net.u2c_scale, net.u2c_relu = conv_bn_relu(net.u2b_concat, 256, pad=1, kernel_size=3, stride=1)
	net.u2d_conv, net.u2d_bn, net.u2d_scale, net.u2d_relu = conv_bn_relu(net.u2c_relu, 256, pad=1, kernel_size=3, stride=1)
	net.u2e_conv, net.u2e_bn, net.u2e_scale = conv_bn(net.u2d_relu, 256, pad=1, kernel_size=3, stride=1)
	net.u2f_conv, net.u2f_bn, net.u2f_scale, net.u2f_eltw, net.u2f_relu = add_layer(net.u2b_concat, net.u2e_conv, 256)
	# ### loss 2
	# net.score2 = L.Convolution(net.u2f_relu,
	# 	param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
	# 	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	# net.upscore2 = L.Deconvolution(net.score2,
	# 	param=[dict(lr_mult=10, decay_mult=1)],
	# 	convolution_param=dict(num_output=num_class,  pad=2, kernel_size=8, stride=4,
	# 		weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	# if phase == "train":
	# 	# net.loss2 = L.WeightedSoftmaxWithLoss(net.upscore2, net.label, net.label_weight,
	# 	net.loss2 = L.SoftmaxWithLoss(net.upscore2, net.label,
	# 		phase=0,
	# 		loss_weight=0.125,
	# 		loss_param=dict(ignore_label=ignore_label))

	############ u1 ############
	############ roi warping ############
	net.u1a_roi_warping_u2f_relu = L.ROIWarping(net.u2f_relu, net.roi_coordinate, pooled_h=dim_data[2]/2, pooled_w=dim_data[2]/2, spatial_scale=0.25, propagate_down=[1,0])
	net.u1a_roi_warping_d1e_eltw = L.ROIWarping(net.d1e_eltw, net.roi_coordinate, pooled_h=dim_data[2]/2, pooled_w=dim_data[2]/2, spatial_scale=0.5, propagate_down=[0,0])
	### a ### Third Deconvolution
	# net.u1a_dconv, net.u1a_bn, net.u1a_scale, net.u1a_relu = deconv_bn_relu(net.u2f_relu, 128, pad=0, kernel_size=2, stride=2)
	net.u1b_concat = L.Concat(net.u1a_roi_warping_u2f_relu, net.u1a_roi_warping_d1e_eltw, axis=1)
	net.u1c_conv, net.u1c_bn, net.u1c_scale, net.u1c_relu = conv_bn_relu(net.u1b_concat, 128, pad=1, kernel_size=3, stride=1)
	# net.u1d_conv, net.u1d_bn, net.u1d_scale, net.u1d_relu = conv_bn_relu(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1)
	net.u1e_conv, net.u1e_bn, net.u1e_scale = conv_bn(net.u1c_relu, 128, pad=1, kernel_size=3, stride=1)
	net.u1f_conv, net.u1f_bn, net.u1f_scale, net.u1f_eltw, net.u1f_relu = add_layer(net.u1b_concat, net.u1e_conv, 128)
	# ### loss 1
	# net.score1 = L.Convolution(net.u1f_relu,
	# 	param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
	# 	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	# net.upscore1 = L.Deconvolution(net.score1,
	# 	param=[dict(lr_mult=10, decay_mult=1)],
	# 	convolution_param=dict(num_output=num_class,  pad=1, kernel_size=4, stride=2,
	# 		weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	# if phase == "train":
	# 	# net.loss1 = L.WeightedSoftmaxWithLoss(net.upscore1, net.label, net.label_weight,
	# 	net.loss1 = L.SoftmaxWithLoss(net.upscore1, net.roi_label,
	# 		phase=0,
	# 		loss_weight=0.25,
	# 		loss_param=dict(ignore_label=ignore_label), propagate_down=[1,0])
	############ u0 ############
	############ roi warping ############
	net.u0a_roi_warping_d0c_conv = L.ROIWarping(net.d0c_conv, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
	### a ### Fourth Deconvolution
	net.u0a_dconv, net.u0a_bn, net.u0a_scale, net.u0a_relu = deconv_bn_relu(net.u1f_relu, 64, pad=0, kernel_size=2, stride=2)
	net.u0b_concat = L.Concat(net.u0a_relu, net.u0a_roi_warping_d0c_conv, axis=1)
	net.u0c_conv, net.u0c_bn, net.u0c_scale, net.u0c_relu = conv_bn_relu(net.u0b_concat, 64, pad=1, kernel_size=3, stride=1)
	net.u0d_conv, net.u0d_bn, net.u0d_scale = conv_bn(net.u0c_relu, 64, pad=1, kernel_size=3, stride=1)
	net.u0e_conv, net.u0e_bn, net.u0e_scale, net.u0e_eltw, net.u0e_relu = add_layer(net.u0b_concat, net.u0d_conv, 64)
	### loss 0
	net.score = L.Convolution(net.u0e_relu,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	if phase == "train":
		# net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
		net.loss = L.SoftmaxWithLoss(net.score, net.roi_label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label), propagate_down=[1,0])
	else:
		# net.prob = L.Softmax(net.score, axis=1)
		net.roi_score_reconstruction = L.ROIPatchReconstruction(net.score, net.roi_coordinate, height=dim_data[2], width=dim_data[2])
		net.prob = L.Softmax(net.roi_score_reconstruction, axis=1)
	return net.to_proto()

def roinet_2d_bn_u3(dim_data, dim_label, num_class, phase='train'):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))
	net.d0b_conv, net.d0b_bn, net.d0b_scale, net.d0b_relu = conv_bn_relu(net.data, 64, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d0c_conv, net.d0c_bn, net.d0c_scale, net.d0c_relu = conv_bn_relu(net.d0b_relu, 64, pad=1, kernel_size=3, stride=1, update_param=False)
	############ d1 ############
	# net.d1a_pool = max_pool(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1a_pool, net.d1a_conv, net.d1a_bn, net.d1a_scale, net.d1a_relu, net.d1a_concat = pool_conv_bn_relu_concat(net.d0c_relu, 64, update_param=False)
	net.d1b_conv, net.d1b_bn, net.d1b_scale, net.d1b_relu = conv_bn_relu(net.d1a_concat, 128, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d1c_conv, net.d1c_bn, net.d1c_scale = conv_bn(net.d1b_relu, 128, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d1c_conv, net.d1c_bn, net.d1c_scale, net.d1c_relu = conv_bn_relu(net.d1b_relu, 128, pad=1, kernel_size=3, stride=1)
	# net.d1d_conv, net.d1d_bn, net.d1d_scale, net.d1d_relu = conv_bn_relu(net.d1c_relu, 256, pad=1, kernel_size=3, stride=1)
	# net.d1e_eltw = L.Eltwise(net.d1a_concat, net.d1c_relu, eltwise_param=dict(operation=1))
	net.d1e_conv, net.d1e_bn, net.d1e_scale, net.d1e_eltw, net.d1e_relu = add_layer(net.d1a_concat, net.d1c_conv, 128, update_param=False)
	
	############ d2 ############
	# net.d2a_pool = max_pool(net.d1d_relu, pad=0, kernel_size=2, stride=2)
	net.d2a_pool, net.d2a_conv, net.d2a_bn, net.d2a_scale, net.d2a_relu, net.d2a_concat = pool_conv_bn_relu_concat(net.d1e_eltw, 128, update_param=False)
	net.d2b_conv, net.d2b_bn, net.d2b_scale, net.d2b_relu = conv_bn_relu(net.d2a_concat, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d2c_conv, net.d2c_bn, net.d2c_scale, net.d2c_relu = conv_bn_relu(net.d2b_relu, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d2d_conv, net.d2d_bn, net.d2d_scale, net.d2d_relu = conv_bn_relu(net.d2c_relu, 256, pad=1, kernel_size=3, stride=1)
	# net.d2e_eltw = L.Eltwise(net.d2a_concat, net.d2d_relu, eltwise_param=dict(operation=1))
	net.d2d_conv, net.d2d_bn, net.d2d_scale = conv_bn(net.d2c_conv, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d2e_conv, net.d2e_bn, net.d2e_scale, net.d2e_eltw, net.d2e_relu = add_layer(net.d2a_concat, net.d2d_conv, 256, update_param=False)
	
	############ d3 ############
	# net.d3a_pool = max_pool(net.d2d_relu, pad=0, kernel_size=2, stride=2)
	net.d3a_pool, net.d3a_conv, net.d3a_bn, net.d3a_scale, net.d3a_relu, net.d3a_concat = pool_conv_bn_relu_concat(net.d2e_eltw, 256, update_param=False)
	net.d3b_conv, net.d3b_bn, net.d3b_scale, net.d3b_relu = conv_bn_relu(net.d3a_concat, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d3c_conv, net.d3c_bn, net.d3c_scale, net.d3c_relu = conv_bn_relu(net.d3b_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d3d_conv, net.d3d_bn, net.d3d_scale, net.d3d_relu = conv_bn_relu(net.d3c_relu, 512, pad=1, kernel_size=3, stride=1)
	# net.d3e_eltw = L.Eltwise(net.d3a_concat, net.d3d_relu, eltwise_param=dict(operation=1))
	net.d3d_conv, net.d3d_bn, net.d3d_scale = conv_bn(net.d3c_conv, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d3e_conv, net.d3e_bn, net.d3e_scale, net.d3e_eltw, net.d3e_relu = add_layer(net.d3a_concat, net.d3d_conv, 512, update_param=False)
	
	############ d4 ############
	# net.d4a_pool = max_pool(net.d3d_relu, pad=0, kernel_size=2, stride=2)
	net.d4a_pool, net.d4a_conv, net.d4a_bn, net.d4a_scale, net.d4a_relu, net.d4a_concat = pool_conv_bn_relu_concat(net.d3e_eltw, 512, update_param=False)
	net.d4b_conv, net.d4b_bn, net.d4b_scale, net.d4b_relu = conv_bn_relu(net.d4a_concat, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d4c_conv, net.d4c_bn, net.d4c_scale, net.d4c_relu = conv_bn_relu(net.d4b_relu, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d4d_conv, net.d4d_bn, net.d4d_scale, net.d4d_relu = conv_bn_relu(net.d4c_relu, 1024, pad=1, kernel_size=3, stride=1)
	# net.d4e_eltw = L.Eltwise(net.d4a_concat, net.d4d_relu, eltwise_param=dict(operation=1))
	net.d4d_conv, net.d4d_bn, net.d4d_scale = conv_bn(net.d4c_conv, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d4e_conv, net.d4e_bn, net.d4e_scale, net.d4e_eltw, net.d4e_relu = add_layer(net.d4a_concat, net.d4d_conv, 1024, update_param=False)
	
	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_bn, net.u3a_scale, net.u3a_relu = deconv_bn_relu(net.d4e_eltw, 512, pad=0, kernel_size=2, stride=2, update_param=False)
	net.u3b_concat = L.Concat(net.u3a_relu, net.d3e_eltw, axis=1, propagate_down=[0,0])
	net.u3c_conv, net.u3c_bn, net.u3c_scale, net.u3c_relu = conv_bn_relu(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3d_conv, net.u3d_bn, net.u3d_scale, net.u3d_relu = conv_bn_relu(net.u3c_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3e_conv, net.u3e_bn, net.u3e_scale = conv_bn(net.u3d_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3f_conv, net.u3f_bn, net.u3f_scale, net.u3f_eltw, net.u3f_relu = add_layer(net.u3b_concat, net.u3e_conv, 512, update_param=False)
	############ score3 ###########
	net.score3 = L.Convolution(net.u3f_relu,
		param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine, propagate_down=0)
	net.upscore3 = L.Deconvolution(net.score3,
		param=[dict(lr_mult=0, decay_mult=0)],
		convolution_param=dict(num_output=num_class,  pad=4, kernel_size=16, stride=8,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine), propagate_down=0)
	# if phase == "train":
	# 	# net.loss3 = L.WeightedSoftmaxWithLoss(net.upscore3, net.label, net.label_weight,
	# 	net.loss3 = L.SoftmaxWithLoss(net.upscore3, net.label,
	# 		phase=0,
	# 		loss_weight=0.0625,
	# 		loss_param=dict(ignore_label=ignore_label))

	############ get roi ############
	net.prob3 = L.Softmax(net.upscore3, axis=1, propagate_down=0)
	net.roi_mask, net.roi_coordinate = L.ROICoordinate(net.prob3, pad=16, threshold=0.80, ntop=2, propagate_down=0)
	# net.slicence = L.Silence(net.roi_mask, ntop=0, propagate_down=0)
	if phase == "train":
		net.slicence = L.Silence(net.roi_mask, ntop=0, propagate_down=0)
	if phase == "train":
		# net.roi_label = L.ROIWarping(net.label, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
		net.label_map = L.ROIWarping(net.label, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
		net.roi_label = L.MultipleThreshold(net.label_map, threshold_point=[0.5,1.5], threshold_value=[0,1,2], propagate_down=0)
	
	############ u2 ############
	############ roi warping ############
	net.u2a_roi_warping_u3f_relu = L.ROIWarping(net.u3f_relu, net.roi_coordinate, pooled_h=dim_data[2]/4, pooled_w=dim_data[2]/4, spatial_scale=0.125, propagate_down=[0,0])
	net.u2a_roi_warping_d2e_eltw = L.ROIWarping(net.d2e_eltw, net.roi_coordinate, pooled_h=dim_data[2]/4, pooled_w=dim_data[2]/4, spatial_scale=0.25, propagate_down=[0,0])
	### a ### Second Deconvolution
	# net.u2a_dconv, net.u2a_bn, net.u2a_scale, net.u2a_relu = deconv_bn_relu(net.u3f_relu, 256, pad=0, kernel_size=2, stride=2)
	net.u2b_concat = L.Concat(net.u2a_roi_warping_u3f_relu, net.u2a_roi_warping_d2e_eltw, axis=1)
	net.u2c_conv, net.u2c_bn, net.u2c_scale, net.u2c_relu = conv_bn_relu(net.u2b_concat, 256, pad=1, kernel_size=3, stride=1)
	net.u2d_conv, net.u2d_bn, net.u2d_scale, net.u2d_relu = conv_bn_relu(net.u2c_relu, 256, pad=1, kernel_size=3, stride=1)
	net.u2e_conv, net.u2e_bn, net.u2e_scale = conv_bn(net.u2d_relu, 256, pad=1, kernel_size=3, stride=1)
	net.u2f_conv, net.u2f_bn, net.u2f_scale, net.u2f_eltw, net.u2f_relu = add_layer(net.u2b_concat, net.u2e_conv, 256)
	# ### loss 2
	# net.score2 = L.Convolution(net.u2f_relu,
	# 	param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
	# 	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	# net.upscore2 = L.Deconvolution(net.score2,
	# 	param=[dict(lr_mult=10, decay_mult=1)],
	# 	convolution_param=dict(num_output=num_class,  pad=2, kernel_size=8, stride=4,
	# 		weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	# if phase == "train":
	# 	# net.loss2 = L.WeightedSoftmaxWithLoss(net.upscore2, net.label, net.label_weight,
	# 	net.loss2 = L.SoftmaxWithLoss(net.upscore2, net.roi_label,
	# 		phase=0,
	# 		loss_weight=0.125,
	# 		loss_param=dict(ignore_label=ignore_label))

	############ u1 ############
	net.u1a_roi_warping_d1e_eltw = L.ROIWarping(net.d1e_eltw, net.roi_coordinate, pooled_h=dim_data[2]/2, pooled_w=dim_data[2]/2, spatial_scale=0.5, propagate_down=[0,0])
	### a ### Third Deconvolution
	net.u1a_dconv, net.u1a_bn, net.u1a_scale, net.u1a_relu = deconv_bn_relu(net.u2f_relu, 128, pad=0, kernel_size=2, stride=2)
	net.u1b_concat = L.Concat(net.u1a_relu, net.u1a_roi_warping_d1e_eltw, axis=1)
	net.u1c_conv, net.u1c_bn, net.u1c_scale, net.u1c_relu = conv_bn_relu(net.u1b_concat, 128, pad=1, kernel_size=3, stride=1)
	# net.u1d_conv, net.u1d_bn, net.u1d_scale, net.u1d_relu = conv_bn_relu(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1)
	net.u1e_conv, net.u1e_bn, net.u1e_scale = conv_bn(net.u1c_relu, 128, pad=1, kernel_size=3, stride=1)
	net.u1f_conv, net.u1f_bn, net.u1f_scale, net.u1f_eltw, net.u1f_relu = add_layer(net.u1b_concat, net.u1e_conv, 128)
	# ### loss 1
	# net.score1 = L.Convolution(net.u1f_relu,
	# 	param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
	# 	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	# net.upscore1 = L.Deconvolution(net.score1,
	# 	param=[dict(lr_mult=10, decay_mult=1)],
	# 	convolution_param=dict(num_output=num_class,  pad=1, kernel_size=4, stride=2,
	# 		weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	# if phase == "train":
	# 	# net.loss1 = L.WeightedSoftmaxWithLoss(net.upscore1, net.label, net.label_weight,
	# 	net.loss1 = L.SoftmaxWithLoss(net.upscore1, net.roi_label,
	# 		phase=0,
	# 		loss_weight=0.25,
	# 		loss_param=dict(ignore_label=ignore_label), propagate_down=[1,0])
	############ u0 ############
	net.u0a_roi_warping_d0c_conv = L.ROIWarping(net.d0c_conv, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
	### a ### Fourth Deconvolution
	net.u0a_dconv, net.u0a_bn, net.u0a_scale, net.u0a_relu = deconv_bn_relu(net.u1f_relu, 64, pad=0, kernel_size=2, stride=2)
	############ roi warping ############
	net.u0b_concat = L.Concat(net.u0a_relu, net.u0a_roi_warping_d0c_conv, axis=1)
	net.u0c_conv, net.u0c_bn, net.u0c_scale, net.u0c_relu = conv_bn_relu(net.u0b_concat, 64, pad=1, kernel_size=3, stride=1)
	net.u0d_conv, net.u0d_bn, net.u0d_scale = conv_bn(net.u0c_relu, 64, pad=1, kernel_size=3, stride=1)
	net.u0e_conv, net.u0e_bn, net.u0e_scale, net.u0e_eltw, net.u0e_relu = add_layer(net.u0b_concat, net.u0d_conv, 64)
	### loss 0
	net.score = L.Convolution(net.u0e_relu,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	if phase == "train":
		# net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
		net.loss = L.SoftmaxWithLoss(net.score, net.roi_label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label), propagate_down=[1,0])
	else:
		# net.prob = L.Softmax(net.score, axis=1)
		net.roi_score_reconstruction = L.ROIPatchReconstruction(net.score, net.roi_coordinate, height=dim_data[2], width=dim_data[2])
		net.prob = L.Softmax(net.roi_score_reconstruction, axis=1)
	return net.to_proto()

def roinet_2d_bn_u3_weighted(dim_data, dim_label, num_class, phase='train'):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))
	net.d0b_conv, net.d0b_bn, net.d0b_scale, net.d0b_relu = conv_bn_relu(net.data, 64, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d0c_conv, net.d0c_bn, net.d0c_scale, net.d0c_relu = conv_bn_relu(net.d0b_relu, 64, pad=1, kernel_size=3, stride=1, update_param=False)
	############ d1 ############
	# net.d1a_pool = max_pool(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1a_pool, net.d1a_conv, net.d1a_bn, net.d1a_scale, net.d1a_relu, net.d1a_concat = pool_conv_bn_relu_concat(net.d0c_relu, 64, update_param=False)
	net.d1b_conv, net.d1b_bn, net.d1b_scale, net.d1b_relu = conv_bn_relu(net.d1a_concat, 128, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d1c_conv, net.d1c_bn, net.d1c_scale = conv_bn(net.d1b_relu, 128, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d1c_conv, net.d1c_bn, net.d1c_scale, net.d1c_relu = conv_bn_relu(net.d1b_relu, 128, pad=1, kernel_size=3, stride=1)
	# net.d1d_conv, net.d1d_bn, net.d1d_scale, net.d1d_relu = conv_bn_relu(net.d1c_relu, 256, pad=1, kernel_size=3, stride=1)
	# net.d1e_eltw = L.Eltwise(net.d1a_concat, net.d1c_relu, eltwise_param=dict(operation=1))
	net.d1e_conv, net.d1e_bn, net.d1e_scale, net.d1e_eltw, net.d1e_relu = add_layer(net.d1a_concat, net.d1c_conv, 128, update_param=False)
	
	############ d2 ############
	# net.d2a_pool = max_pool(net.d1d_relu, pad=0, kernel_size=2, stride=2)
	net.d2a_pool, net.d2a_conv, net.d2a_bn, net.d2a_scale, net.d2a_relu, net.d2a_concat = pool_conv_bn_relu_concat(net.d1e_eltw, 128, update_param=False)
	net.d2b_conv, net.d2b_bn, net.d2b_scale, net.d2b_relu = conv_bn_relu(net.d2a_concat, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d2c_conv, net.d2c_bn, net.d2c_scale, net.d2c_relu = conv_bn_relu(net.d2b_relu, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d2d_conv, net.d2d_bn, net.d2d_scale, net.d2d_relu = conv_bn_relu(net.d2c_relu, 256, pad=1, kernel_size=3, stride=1)
	# net.d2e_eltw = L.Eltwise(net.d2a_concat, net.d2d_relu, eltwise_param=dict(operation=1))
	net.d2d_conv, net.d2d_bn, net.d2d_scale = conv_bn(net.d2c_conv, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d2e_conv, net.d2e_bn, net.d2e_scale, net.d2e_eltw, net.d2e_relu = add_layer(net.d2a_concat, net.d2d_conv, 256, update_param=False)
	
	############ d3 ############
	# net.d3a_pool = max_pool(net.d2d_relu, pad=0, kernel_size=2, stride=2)
	net.d3a_pool, net.d3a_conv, net.d3a_bn, net.d3a_scale, net.d3a_relu, net.d3a_concat = pool_conv_bn_relu_concat(net.d2e_eltw, 256, update_param=False)
	net.d3b_conv, net.d3b_bn, net.d3b_scale, net.d3b_relu = conv_bn_relu(net.d3a_concat, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d3c_conv, net.d3c_bn, net.d3c_scale, net.d3c_relu = conv_bn_relu(net.d3b_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d3d_conv, net.d3d_bn, net.d3d_scale, net.d3d_relu = conv_bn_relu(net.d3c_relu, 512, pad=1, kernel_size=3, stride=1)
	# net.d3e_eltw = L.Eltwise(net.d3a_concat, net.d3d_relu, eltwise_param=dict(operation=1))
	net.d3d_conv, net.d3d_bn, net.d3d_scale = conv_bn(net.d3c_conv, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d3e_conv, net.d3e_bn, net.d3e_scale, net.d3e_eltw, net.d3e_relu = add_layer(net.d3a_concat, net.d3d_conv, 512, update_param=False)
	
	############ d4 ############
	# net.d4a_pool = max_pool(net.d3d_relu, pad=0, kernel_size=2, stride=2)
	net.d4a_pool, net.d4a_conv, net.d4a_bn, net.d4a_scale, net.d4a_relu, net.d4a_concat = pool_conv_bn_relu_concat(net.d3e_eltw, 512, update_param=False)
	net.d4b_conv, net.d4b_bn, net.d4b_scale, net.d4b_relu = conv_bn_relu(net.d4a_concat, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d4c_conv, net.d4c_bn, net.d4c_scale, net.d4c_relu = conv_bn_relu(net.d4b_relu, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d4d_conv, net.d4d_bn, net.d4d_scale, net.d4d_relu = conv_bn_relu(net.d4c_relu, 1024, pad=1, kernel_size=3, stride=1)
	# net.d4e_eltw = L.Eltwise(net.d4a_concat, net.d4d_relu, eltwise_param=dict(operation=1))
	net.d4d_conv, net.d4d_bn, net.d4d_scale = conv_bn(net.d4c_conv, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d4e_conv, net.d4e_bn, net.d4e_scale, net.d4e_eltw, net.d4e_relu = add_layer(net.d4a_concat, net.d4d_conv, 1024, update_param=False)
	
	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_bn, net.u3a_scale, net.u3a_relu = deconv_bn_relu(net.d4e_eltw, 512, pad=0, kernel_size=2, stride=2, update_param=False)
	net.u3b_concat = L.Concat(net.u3a_relu, net.d3e_eltw, axis=1, propagate_down=[0,0])
	net.u3c_conv, net.u3c_bn, net.u3c_scale, net.u3c_relu = conv_bn_relu(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3d_conv, net.u3d_bn, net.u3d_scale, net.u3d_relu = conv_bn_relu(net.u3c_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3e_conv, net.u3e_bn, net.u3e_scale = conv_bn(net.u3d_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3f_conv, net.u3f_bn, net.u3f_scale, net.u3f_eltw, net.u3f_relu = add_layer(net.u3b_concat, net.u3e_conv, 512, update_param=False)
	############ score3 ###########
	net.score3 = L.Convolution(net.u3f_relu,
		param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine, propagate_down=0)
	net.upscore3 = L.Deconvolution(net.score3,
		param=[dict(lr_mult=0, decay_mult=0)],
		convolution_param=dict(num_output=num_class,  pad=4, kernel_size=16, stride=8,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine), propagate_down=0)
	# if phase == "train":
	# 	# net.loss3 = L.WeightedSoftmaxWithLoss(net.upscore3, net.label, net.label_weight,
	# 	net.loss3 = L.SoftmaxWithLoss(net.upscore3, net.label,
	# 		phase=0,
	# 		loss_weight=0.0625,
	# 		loss_param=dict(ignore_label=ignore_label))

	############ get roi ############
	net.prob3 = L.Softmax(net.upscore3, axis=1, propagate_down=0)
	net.roi_mask, net.roi_coordinate = L.ROICoordinate(net.prob3, pad=16, threshold=0.80, ntop=2, propagate_down=0)
	# net.slicence = L.Silence(net.roi_mask, ntop=0, propagate_down=0)
	if phase == "train":
		net.slicence = L.Silence(net.roi_mask, ntop=0, propagate_down=0)
	if phase == "train":
		# net.roi_label = L.ROIWarping(net.label, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
		net.label_map = L.ROIWarping(net.label, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
		net.roi_label = L.MultipleThreshold(net.label_map, threshold_point=[0.5,1.5], threshold_value=[0,1,2], propagate_down=0)
		net.roi_label_weight = L.ROIWarping(net.label_weight, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
	
	############ u2 ############
	############ roi warping ############
	net.u2a_roi_warping_u3f_relu = L.ROIWarping(net.u3f_relu, net.roi_coordinate, pooled_h=dim_data[2]/4, pooled_w=dim_data[2]/4, spatial_scale=0.125, propagate_down=[0,0])
	net.u2a_roi_warping_d2e_eltw = L.ROIWarping(net.d2e_eltw, net.roi_coordinate, pooled_h=dim_data[2]/4, pooled_w=dim_data[2]/4, spatial_scale=0.25, propagate_down=[0,0])
	### a ### Second Deconvolution
	# net.u2a_dconv, net.u2a_bn, net.u2a_scale, net.u2a_relu = deconv_bn_relu(net.u3f_relu, 256, pad=0, kernel_size=2, stride=2)
	net.u2b_concat = L.Concat(net.u2a_roi_warping_u3f_relu, net.u2a_roi_warping_d2e_eltw, axis=1)
	net.u2c_conv, net.u2c_bn, net.u2c_scale, net.u2c_relu = conv_bn_relu(net.u2b_concat, 256, pad=1, kernel_size=3, stride=1)
	net.u2d_conv, net.u2d_bn, net.u2d_scale, net.u2d_relu = conv_bn_relu(net.u2c_relu, 256, pad=1, kernel_size=3, stride=1)
	net.u2e_conv, net.u2e_bn, net.u2e_scale = conv_bn(net.u2d_relu, 256, pad=1, kernel_size=3, stride=1)
	net.u2f_conv, net.u2f_bn, net.u2f_scale, net.u2f_eltw, net.u2f_relu = add_layer(net.u2b_concat, net.u2e_conv, 256)
	# ### loss 2
	net.score2 = L.Convolution(net.u2f_relu,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore2 = L.Deconvolution(net.score2,
		param=[dict(lr_mult=10, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=2, kernel_size=8, stride=4,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss2 = L.WeightedSoftmaxWithLoss(net.upscore2, net.roi_label, net.roi_label_weight,
		# net.loss2 = L.SoftmaxWithLoss(net.upscore2, net.roi_label,
			phase=0,
			loss_weight=0.125,
			loss_param=dict(ignore_label=ignore_label), propagate_down=[1,0,0])

	############ u1 ############
	net.u1a_roi_warping_d1e_eltw = L.ROIWarping(net.d1e_eltw, net.roi_coordinate, pooled_h=dim_data[2]/2, pooled_w=dim_data[2]/2, spatial_scale=0.5, propagate_down=[0,0])
	### a ### Third Deconvolution
	net.u1a_dconv, net.u1a_bn, net.u1a_scale, net.u1a_relu = deconv_bn_relu(net.u2f_relu, 128, pad=0, kernel_size=2, stride=2)
	net.u1b_concat = L.Concat(net.u1a_relu, net.u1a_roi_warping_d1e_eltw, axis=1)
	net.u1c_conv, net.u1c_bn, net.u1c_scale, net.u1c_relu = conv_bn_relu(net.u1b_concat, 128, pad=1, kernel_size=3, stride=1)
	# net.u1d_conv, net.u1d_bn, net.u1d_scale, net.u1d_relu = conv_bn_relu(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1)
	net.u1e_conv, net.u1e_bn, net.u1e_scale = conv_bn(net.u1c_relu, 128, pad=1, kernel_size=3, stride=1)
	net.u1f_conv, net.u1f_bn, net.u1f_scale, net.u1f_eltw, net.u1f_relu = add_layer(net.u1b_concat, net.u1e_conv, 128)
	# ### loss 1
	net.score1 = L.Convolution(net.u1f_relu,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore1 = L.Deconvolution(net.score1,
		param=[dict(lr_mult=10, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=1, kernel_size=4, stride=2,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss1 = L.WeightedSoftmaxWithLoss(net.upscore1, net.roi_label, net.roi_label_weight,
		# net.loss1 = L.SoftmaxWithLoss(net.upscore1, net.roi_label,
			phase=0,
			loss_weight=0.25,
			loss_param=dict(ignore_label=ignore_label), propagate_down=[1,0,0])
	############ u0 ############
	net.u0a_roi_warping_d0c_conv = L.ROIWarping(net.d0c_conv, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
	### a ### Fourth Deconvolution
	net.u0a_dconv, net.u0a_bn, net.u0a_scale, net.u0a_relu = deconv_bn_relu(net.u1f_relu, 64, pad=0, kernel_size=2, stride=2)
	############ roi warping ############
	net.u0b_concat = L.Concat(net.u0a_relu, net.u0a_roi_warping_d0c_conv, axis=1)
	net.u0c_conv, net.u0c_bn, net.u0c_scale, net.u0c_relu = conv_bn_relu(net.u0b_concat, 64, pad=1, kernel_size=3, stride=1)
	net.u0d_conv, net.u0d_bn, net.u0d_scale = conv_bn(net.u0c_relu, 64, pad=1, kernel_size=3, stride=1)
	net.u0e_conv, net.u0e_bn, net.u0e_scale, net.u0e_eltw, net.u0e_relu = add_layer(net.u0b_concat, net.u0d_conv, 64)
	### loss 0
	net.score = L.Convolution(net.u0e_relu,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	if phase == "train":
		net.loss = L.WeightedSoftmaxWithLoss(net.score, net.roi_label, net.roi_label_weight,
		# net.loss = L.SoftmaxWithLoss(net.score, net.roi_label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label), propagate_down=[1,0,0])
	else:
		# net.prob = L.Softmax(net.score, axis=1)
		net.roi_score_reconstruction = L.ROIPatchReconstruction(net.score, net.roi_coordinate, height=dim_data[2], width=dim_data[2])
		net.prob = L.Softmax(net.roi_score_reconstruction, axis=1)
	return net.to_proto()

def roinet_2d_bn_d3(dim_data, dim_label, num_class, phase='train'):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))
	net.d0b_conv, net.d0b_bn, net.d0b_scale, net.d0b_relu = conv_bn_relu(net.data, 64, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d0c_conv, net.d0c_bn, net.d0c_scale, net.d0c_relu = conv_bn_relu(net.d0b_relu, 64, pad=1, kernel_size=3, stride=1, update_param=False)
	############ d1 ############
	# net.d1a_pool = max_pool(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1a_pool, net.d1a_conv, net.d1a_bn, net.d1a_scale, net.d1a_relu, net.d1a_concat = pool_conv_bn_relu_concat(net.d0c_relu, 64, update_param=False)
	net.d1b_conv, net.d1b_bn, net.d1b_scale, net.d1b_relu = conv_bn_relu(net.d1a_concat, 128, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d1c_conv, net.d1c_bn, net.d1c_scale = conv_bn(net.d1b_relu, 128, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d1c_conv, net.d1c_bn, net.d1c_scale, net.d1c_relu = conv_bn_relu(net.d1b_relu, 128, pad=1, kernel_size=3, stride=1)
	# net.d1d_conv, net.d1d_bn, net.d1d_scale, net.d1d_relu = conv_bn_relu(net.d1c_relu, 256, pad=1, kernel_size=3, stride=1)
	# net.d1e_eltw = L.Eltwise(net.d1a_concat, net.d1c_relu, eltwise_param=dict(operation=1))
	net.d1e_conv, net.d1e_bn, net.d1e_scale, net.d1e_eltw, net.d1e_relu = add_layer(net.d1a_concat, net.d1c_conv, 128, update_param=False)
	
	############ d2 ############
	# net.d2a_pool = max_pool(net.d1d_relu, pad=0, kernel_size=2, stride=2)
	net.d2a_pool, net.d2a_conv, net.d2a_bn, net.d2a_scale, net.d2a_relu, net.d2a_concat = pool_conv_bn_relu_concat(net.d1e_relu, 128, update_param=False)
	net.d2b_conv, net.d2b_bn, net.d2b_scale, net.d2b_relu = conv_bn_relu(net.d2a_concat, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d2c_conv, net.d2c_bn, net.d2c_scale, net.d2c_relu = conv_bn_relu(net.d2b_relu, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d2d_conv, net.d2d_bn, net.d2d_scale, net.d2d_relu = conv_bn_relu(net.d2c_relu, 256, pad=1, kernel_size=3, stride=1)
	# net.d2e_eltw = L.Eltwise(net.d2a_concat, net.d2d_relu, eltwise_param=dict(operation=1))
	net.d2d_conv, net.d2d_bn, net.d2d_scale = conv_bn(net.d2c_relu, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d2e_conv, net.d2e_bn, net.d2e_scale, net.d2e_eltw, net.d2e_relu = add_layer(net.d2a_concat, net.d2d_conv, 256, update_param=False)
	
	############ d3 ############
	# net.d3a_pool = max_pool(net.d2d_relu, pad=0, kernel_size=2, stride=2)
	net.d3a_pool, net.d3a_conv, net.d3a_bn, net.d3a_scale, net.d3a_relu, net.d3a_concat = pool_conv_bn_relu_concat(net.d2e_relu, 256, update_param=False)
	net.d3b_conv, net.d3b_bn, net.d3b_scale, net.d3b_relu = conv_bn_relu(net.d3a_concat, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d3c_conv, net.d3c_bn, net.d3c_scale, net.d3c_relu = conv_bn_relu(net.d3b_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d3d_conv, net.d3d_bn, net.d3d_scale, net.d3d_relu = conv_bn_relu(net.d3c_relu, 512, pad=1, kernel_size=3, stride=1)
	# net.d3e_eltw = L.Eltwise(net.d3a_concat, net.d3d_relu, eltwise_param=dict(operation=1))
	net.d3d_conv, net.d3d_bn, net.d3d_scale = conv_bn(net.d3c_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d3e_conv, net.d3e_bn, net.d3e_scale, net.d3e_eltw, net.d3e_relu = add_layer(net.d3a_concat, net.d3d_conv, 512, update_param=False)
	
	############ d4 ############
	# net.d4a_pool = max_pool(net.d3d_relu, pad=0, kernel_size=2, stride=2)
	net.d4a_pool, net.d4a_conv, net.d4a_bn, net.d4a_scale, net.d4a_relu, net.d4a_concat = pool_conv_bn_relu_concat(net.d3e_relu, 512, update_param=False)
	net.d4b_conv, net.d4b_bn, net.d4b_scale, net.d4b_relu = conv_bn_relu(net.d4a_concat, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d4c_conv, net.d4c_bn, net.d4c_scale, net.d4c_relu = conv_bn_relu(net.d4b_relu, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d4d_conv, net.d4d_bn, net.d4d_scale, net.d4d_relu = conv_bn_relu(net.d4c_relu, 1024, pad=1, kernel_size=3, stride=1)
	# net.d4e_eltw = L.Eltwise(net.d4a_concat, net.d4d_relu, eltwise_param=dict(operation=1))
	net.d4d_conv, net.d4d_bn, net.d4d_scale = conv_bn(net.d4c_relu, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d4e_conv, net.d4e_bn, net.d4e_scale, net.d4e_eltw, net.d4e_relu = add_layer(net.d4a_concat, net.d4d_conv, 1024, update_param=False)
	
	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_bn, net.u3a_scale, net.u3a_relu = deconv_bn_relu(net.d4e_relu, 512, pad=0, kernel_size=2, stride=2, update_param=False)
	net.u3b_concat = L.Concat(net.u3a_relu, net.d3e_relu, axis=1, propagate_down=[0,0])
	net.u3c_conv, net.u3c_bn, net.u3c_scale, net.u3c_relu = conv_bn_relu(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3d_conv, net.u3d_bn, net.u3d_scale, net.u3d_relu = conv_bn_relu(net.u3c_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3e_conv, net.u3e_bn, net.u3e_scale = conv_bn(net.u3d_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3f_conv, net.u3f_bn, net.u3f_scale, net.u3f_eltw, net.u3f_relu = add_layer(net.u3b_concat, net.u3e_conv, 512, update_param=False)
	############ score3 ###########
	net.score3 = L.Convolution(net.u3f_relu,
		param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine, propagate_down=0)
	net.upscore3 = L.Deconvolution(net.score3,
		param=[dict(lr_mult=0, decay_mult=0)],
		convolution_param=dict(num_output=num_class,  pad=4, kernel_size=16, stride=8,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine), propagate_down=0)
	# if phase == "train":
	# 	# net.loss3 = L.WeightedSoftmaxWithLoss(net.upscore3, net.label, net.label_weight,
	# 	net.loss3 = L.SoftmaxWithLoss(net.upscore3, net.label,
	# 		phase=0,
	# 		loss_weight=0.0625,
	# 		loss_param=dict(ignore_label=ignore_label))

	############ get roi ############
	net.prob3 = L.Softmax(net.upscore3, axis=1, propagate_down=0)
	net.roi_mask, net.roi_coordinate = L.ROICoordinate(net.prob3, pad=16, threshold=0.80, ntop=2, propagate_down=0)
	# net.slicence = L.Silence(net.roi_mask, ntop=0, propagate_down=0)
	if phase == "train":
		net.slicence = L.Silence(net.roi_mask, ntop=0, propagate_down=0)
	if phase == "train":
		# net.roi_label = L.ROIWarping(net.label, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
		net.label_map = L.ROIWarping(net.label, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
		net.roi_label = L.MultipleThreshold(net.label_map, threshold_point=[0.5,1.5], threshold_value=[0,1,2], propagate_down=0)
		# net.roi_label_weight = L.ROIWarping(net.label_weight, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
	
	############ roi warping ############
	net.roi_warping_d3e_relu = L.ROIWarping(net.d3e_relu, net.roi_coordinate, pooled_h=dim_data[2]/4, pooled_w=dim_data[2]/4, spatial_scale=0.125, propagate_down=[0,0])
	net.roi_warping_d2e_relu = L.ROIWarping(net.d2e_relu, net.roi_coordinate, pooled_h=dim_data[2]/4, pooled_w=dim_data[2]/4, spatial_scale=0.25, propagate_down=[0,0])
	net.roi_warping_concat = L.Concat(net.roi_warping_d3e_relu, net.roi_warping_d2e_relu, axis=1)
	net.u2c_conv_roi, net.u2c_bn_roi, net.u2c_scale_roi, net.u2c_relu_roi = conv_bn_relu(net.roi_warping_concat, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u2d_conv_roi, net.u2d_bn_roi, net.u2d_scale_roi, net.u2d_relu_roi = conv_bn_relu(net.u2c_relu_roi, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u2e_conv_roi, net.u2e_bn_roi, net.u2e_scale_roi = conv_bn(net.u2d_relu_roi, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u2f_conv_roi, net.u2f_bn_roi, net.u2f_scale_roi, net.u2f_eltw_roi, net.u2f_relu_roi = add_layer(net.roi_warping_concat, net.u2e_conv_roi, 256, update_param=False)

	### loss 2
	net.score2 = L.Convolution(net.u2f_relu_roi,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore2 = L.Deconvolution(net.score2,
		param=[dict(lr_mult=10, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=2, kernel_size=8, stride=4,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		# net.loss2 = L.WeightedSoftmaxWithLoss(net.upscore2, net.roi_label, net.roi_label_weight,
		net.loss2 = L.SoftmaxWithLoss(net.upscore2, net.roi_label,
			phase=0,
			loss_weight=0.25,
			loss_param=dict(ignore_label=ignore_label), propagate_down=[1,0])

	############ u1 ############
	net.u1a_roi_warping_d1e_relu = L.ROIWarping(net.d1e_relu, net.roi_coordinate, pooled_h=dim_data[2]/2, pooled_w=dim_data[2]/2, spatial_scale=0.5, propagate_down=[0,0])
	### a ### Third Deconvolution
	net.u1a_dconv_roi, net.u1a_bn_roi, net.u1a_scale_roi, net.u1a_relu_roi = deconv_bn_relu(net.u2f_relu_roi, 128, pad=0, kernel_size=2, stride=2)
	net.u1b_concat_roi = L.Concat(net.u1a_relu_roi, net.u1a_roi_warping_d1e_relu, axis=1)
	net.u1c_conv_roi, net.u1c_bn_roi, net.u1c_scale_roi, net.u1c_relu_roi = conv_bn_relu(net.u1b_concat_roi, 128, pad=1, kernel_size=3, stride=1)
	# net.u1d_conv, net.u1d_bn, net.u1d_scale, net.u1d_relu = conv_bn_relu(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1)
	net.u1e_conv_roi, net.u1e_bn_roi, net.u1e_scale_roi = conv_bn(net.u1c_relu_roi, 128, pad=1, kernel_size=3, stride=1)
	net.u1f_conv_roi, net.u1f_bn_roi, net.u1f_scale_roi, net.u1f_eltw_roi, net.u1f_relu_roi = add_layer(net.u1b_concat_roi, net.u1e_conv_roi, 128)
	# ### loss 1
	net.score1 = L.Convolution(net.u1f_relu_roi,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore1 = L.Deconvolution(net.score1,
		param=[dict(lr_mult=10, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=1, kernel_size=4, stride=2,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		# net.loss1 = L.WeightedSoftmaxWithLoss(net.upscore1, net.roi_label, net.roi_label_weight,
		net.loss1 = L.SoftmaxWithLoss(net.upscore1, net.roi_label,
			phase=0,
			loss_weight=0.5,
			loss_param=dict(ignore_label=ignore_label), propagate_down=[1,0])
	############ u0 ############
	net.u0a_roi_warping_d0c_conv = L.ROIWarping(net.d0c_conv, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
	### a ### Fourth Deconvolution
	net.u0a_dconv_roi, net.u0a_bn_roi, net.u0a_scale_roi, net.u0a_relu_roi = deconv_bn_relu(net.u1f_relu_roi, 64, pad=0, kernel_size=2, stride=2)
	############ roi warping ############
	net.u0b_concat_roi = L.Concat(net.u0a_relu_roi, net.u0a_roi_warping_d0c_conv, axis=1)
	net.u0c_conv_roi, net.u0c_bn_roi, net.u0c_scale_roi, net.u0c_relu_roi = conv_bn_relu(net.u0b_concat_roi, 64, pad=1, kernel_size=3, stride=1)
	net.u0d_conv_roi, net.u0d_bn_roi, net.u0d_scale_roi = conv_bn(net.u0c_relu_roi, 64, pad=1, kernel_size=3, stride=1)
	net.u0e_conv_roi, net.u0e_bn_roi, net.u0e_scale_roi, net.u0e_eltw_roi, net.u0e_relu_roi = add_layer(net.u0b_concat_roi, net.u0d_conv_roi, 64)
	### loss 0
	net.score = L.Convolution(net.u0e_relu_roi,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	if phase == "train":
		# net.loss = L.WeightedSoftmaxWithLoss(net.score, net.roi_label, net.roi_label_weight,
		net.loss = L.SoftmaxWithLoss(net.score, net.roi_label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label), propagate_down=[1,0])
	else:
		net.roi_score_reconstruction = L.ROIPatchReconstruction(net.score, net.roi_coordinate, height=dim_data[2], width=dim_data[2])
		net.prob = L.Softmax(net.roi_score_reconstruction, axis=1)
		# net.probtmp = L.Softmax(net.score, axis=1)
		# net.prob = L.ROIPatchReconstruction(net.probtmp, net.roi_coordinate, height=dim_data[2], width=dim_data[2])
	return net.to_proto()

def roinet_2d_bn_d3_weighted(dim_data, dim_label, num_class, phase='train'):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))
	net.d0b_conv, net.d0b_bn, net.d0b_scale, net.d0b_relu = conv_bn_relu(net.data, 64, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d0c_conv, net.d0c_bn, net.d0c_scale, net.d0c_relu = conv_bn_relu(net.d0b_relu, 64, pad=1, kernel_size=3, stride=1, update_param=False)
	############ d1 ############
	# net.d1a_pool = max_pool(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1a_pool, net.d1a_conv, net.d1a_bn, net.d1a_scale, net.d1a_relu, net.d1a_concat = pool_conv_bn_relu_concat(net.d0c_relu, 64, update_param=False)
	net.d1b_conv, net.d1b_bn, net.d1b_scale, net.d1b_relu = conv_bn_relu(net.d1a_concat, 128, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d1c_conv, net.d1c_bn, net.d1c_scale = conv_bn(net.d1b_relu, 128, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d1c_conv, net.d1c_bn, net.d1c_scale, net.d1c_relu = conv_bn_relu(net.d1b_relu, 128, pad=1, kernel_size=3, stride=1)
	# net.d1d_conv, net.d1d_bn, net.d1d_scale, net.d1d_relu = conv_bn_relu(net.d1c_relu, 256, pad=1, kernel_size=3, stride=1)
	# net.d1e_eltw = L.Eltwise(net.d1a_concat, net.d1c_relu, eltwise_param=dict(operation=1))
	net.d1e_conv, net.d1e_bn, net.d1e_scale, net.d1e_eltw, net.d1e_relu = add_layer(net.d1a_concat, net.d1c_conv, 128, update_param=False)
	
	############ d2 ############
	# net.d2a_pool = max_pool(net.d1d_relu, pad=0, kernel_size=2, stride=2)
	net.d2a_pool, net.d2a_conv, net.d2a_bn, net.d2a_scale, net.d2a_relu, net.d2a_concat = pool_conv_bn_relu_concat(net.d1e_relu, 128, update_param=False)
	net.d2b_conv, net.d2b_bn, net.d2b_scale, net.d2b_relu = conv_bn_relu(net.d2a_concat, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d2c_conv, net.d2c_bn, net.d2c_scale, net.d2c_relu = conv_bn_relu(net.d2b_relu, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d2d_conv, net.d2d_bn, net.d2d_scale, net.d2d_relu = conv_bn_relu(net.d2c_relu, 256, pad=1, kernel_size=3, stride=1)
	# net.d2e_eltw = L.Eltwise(net.d2a_concat, net.d2d_relu, eltwise_param=dict(operation=1))
	net.d2d_conv, net.d2d_bn, net.d2d_scale = conv_bn(net.d2c_relu, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d2e_conv, net.d2e_bn, net.d2e_scale, net.d2e_eltw, net.d2e_relu = add_layer(net.d2a_concat, net.d2d_conv, 256, update_param=False)
	
	############ d3 ############
	# net.d3a_pool = max_pool(net.d2d_relu, pad=0, kernel_size=2, stride=2)
	net.d3a_pool, net.d3a_conv, net.d3a_bn, net.d3a_scale, net.d3a_relu, net.d3a_concat = pool_conv_bn_relu_concat(net.d2e_relu, 256, update_param=False)
	net.d3b_conv, net.d3b_bn, net.d3b_scale, net.d3b_relu = conv_bn_relu(net.d3a_concat, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d3c_conv, net.d3c_bn, net.d3c_scale, net.d3c_relu = conv_bn_relu(net.d3b_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d3d_conv, net.d3d_bn, net.d3d_scale, net.d3d_relu = conv_bn_relu(net.d3c_relu, 512, pad=1, kernel_size=3, stride=1)
	# net.d3e_eltw = L.Eltwise(net.d3a_concat, net.d3d_relu, eltwise_param=dict(operation=1))
	net.d3d_conv, net.d3d_bn, net.d3d_scale = conv_bn(net.d3c_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d3e_conv, net.d3e_bn, net.d3e_scale, net.d3e_eltw, net.d3e_relu = add_layer(net.d3a_concat, net.d3d_conv, 512, update_param=False)
	
	############ d4 ############
	# net.d4a_pool = max_pool(net.d3d_relu, pad=0, kernel_size=2, stride=2)
	net.d4a_pool, net.d4a_conv, net.d4a_bn, net.d4a_scale, net.d4a_relu, net.d4a_concat = pool_conv_bn_relu_concat(net.d3e_relu, 512, update_param=False)
	net.d4b_conv, net.d4b_bn, net.d4b_scale, net.d4b_relu = conv_bn_relu(net.d4a_concat, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d4c_conv, net.d4c_bn, net.d4c_scale, net.d4c_relu = conv_bn_relu(net.d4b_relu, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d4d_conv, net.d4d_bn, net.d4d_scale, net.d4d_relu = conv_bn_relu(net.d4c_relu, 1024, pad=1, kernel_size=3, stride=1)
	# net.d4e_eltw = L.Eltwise(net.d4a_concat, net.d4d_relu, eltwise_param=dict(operation=1))
	net.d4d_conv, net.d4d_bn, net.d4d_scale = conv_bn(net.d4c_relu, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d4e_conv, net.d4e_bn, net.d4e_scale, net.d4e_eltw, net.d4e_relu = add_layer(net.d4a_concat, net.d4d_conv, 1024, update_param=False)
	
	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_bn, net.u3a_scale, net.u3a_relu = deconv_bn_relu(net.d4e_relu, 512, pad=0, kernel_size=2, stride=2, update_param=False)
	net.u3b_concat = L.Concat(net.u3a_relu, net.d3e_relu, axis=1, propagate_down=[0,0])
	net.u3c_conv, net.u3c_bn, net.u3c_scale, net.u3c_relu = conv_bn_relu(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3d_conv, net.u3d_bn, net.u3d_scale, net.u3d_relu = conv_bn_relu(net.u3c_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3e_conv, net.u3e_bn, net.u3e_scale = conv_bn(net.u3d_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3f_conv, net.u3f_bn, net.u3f_scale, net.u3f_eltw, net.u3f_relu = add_layer(net.u3b_concat, net.u3e_conv, 512, update_param=False)
	############ score3 ###########
	net.score3 = L.Convolution(net.u3f_relu,
		param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine, propagate_down=0)
	net.upscore3 = L.Deconvolution(net.score3,
		param=[dict(lr_mult=0, decay_mult=0)],
		convolution_param=dict(num_output=num_class,  pad=4, kernel_size=16, stride=8,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine), propagate_down=0)
	# if phase == "train":
	# 	# net.loss3 = L.WeightedSoftmaxWithLoss(net.upscore3, net.label, net.label_weight,
	# 	net.loss3 = L.SoftmaxWithLoss(net.upscore3, net.label,
	# 		phase=0,
	# 		loss_weight=0.0625,
	# 		loss_param=dict(ignore_label=ignore_label))

	############ get roi ############
	net.prob3 = L.Softmax(net.upscore3, axis=1, propagate_down=0)
	net.roi_mask, net.roi_coordinate = L.ROICoordinate(net.prob3, pad=16, threshold=0.80, ntop=2, propagate_down=0)
	# net.slicence = L.Silence(net.roi_mask, ntop=0, propagate_down=0)
	if phase == "train":
		net.slicence = L.Silence(net.roi_mask, ntop=0, propagate_down=0)
	if phase == "train":
		# net.roi_label = L.ROIWarping(net.label, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
		net.label_map = L.ROIWarping(net.label, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
		net.roi_label = L.MultipleThreshold(net.label_map, threshold_point=[0.5,1.5], threshold_value=[0,1,2], propagate_down=0)
		net.roi_label_weight = L.ROIWarping(net.label_weight, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
	
	############ roi warping ############
	net.roi_warping_d3e_relu = L.ROIWarping(net.d3e_relu, net.roi_coordinate, pooled_h=dim_data[2]/4, pooled_w=dim_data[2]/4, spatial_scale=0.125, propagate_down=[0,0])
	net.roi_warping_d2e_relu = L.ROIWarping(net.d2e_relu, net.roi_coordinate, pooled_h=dim_data[2]/4, pooled_w=dim_data[2]/4, spatial_scale=0.25, propagate_down=[0,0])
	net.roi_warping_concat = L.Concat(net.roi_warping_d3e_relu, net.roi_warping_d2e_relu, axis=1)
	net.u2c_conv_roi, net.u2c_bn_roi, net.u2c_scale_roi, net.u2c_relu_roi = conv_bn_relu(net.roi_warping_concat, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u2d_conv_roi, net.u2d_bn_roi, net.u2d_scale_roi, net.u2d_relu_roi = conv_bn_relu(net.u2c_relu_roi, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u2e_conv_roi, net.u2e_bn_roi, net.u2e_scale_roi = conv_bn(net.u2d_relu_roi, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u2f_conv_roi, net.u2f_bn_roi, net.u2f_scale_roi, net.u2f_eltw_roi, net.u2f_relu_roi = add_layer(net.roi_warping_concat, net.u2e_conv_roi, 256, update_param=False)
	
	### loss 2
	net.score2 = L.Convolution(net.u2f_relu_roi,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore2 = L.Deconvolution(net.score2,
		param=[dict(lr_mult=10, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=2, kernel_size=8, stride=4,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss2 = L.WeightedSoftmaxWithLoss(net.upscore2, net.roi_label, net.roi_label_weight,
		# net.loss2 = L.SoftmaxWithLoss(net.upscore2, net.roi_label,
			phase=0,
			loss_weight=0.25,
			loss_param=dict(ignore_label=ignore_label), propagate_down=[1,0,0])

	############ u1 ############
	net.u1a_roi_warping_d1e_relu = L.ROIWarping(net.d1e_relu, net.roi_coordinate, pooled_h=dim_data[2]/2, pooled_w=dim_data[2]/2, spatial_scale=0.5, propagate_down=[0,0])
	### a ### Third Deconvolution
	net.u1a_dconv_roi, net.u1a_bn_roi, net.u1a_scale_roi, net.u1a_relu_roi = deconv_bn_relu(net.u2f_relu_roi, 128, pad=0, kernel_size=2, stride=2)
	net.u1b_concat_roi = L.Concat(net.u1a_relu_roi, net.u1a_roi_warping_d1e_relu, axis=1)
	net.u1c_conv_roi, net.u1c_bn_roi, net.u1c_scale_roi, net.u1c_relu_roi = conv_bn_relu(net.u1b_concat_roi, 128, pad=1, kernel_size=3, stride=1)
	# net.u1d_conv, net.u1d_bn, net.u1d_scale, net.u1d_relu = conv_bn_relu(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1)
	net.u1e_conv_roi, net.u1e_bn_roi, net.u1e_scale_roi = conv_bn(net.u1c_relu_roi, 128, pad=1, kernel_size=3, stride=1)
	net.u1f_conv_roi, net.u1f_bn_roi, net.u1f_scale_roi, net.u1f_eltw_roi, net.u1f_relu_roi = add_layer(net.u1b_concat_roi, net.u1e_conv_roi, 128)
	# ### loss 1
	net.score1 = L.Convolution(net.u1f_relu_roi,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore1 = L.Deconvolution(net.score1,
		param=[dict(lr_mult=10, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=1, kernel_size=4, stride=2,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss1 = L.WeightedSoftmaxWithLoss(net.upscore1, net.roi_label, net.roi_label_weight,
		# net.loss1 = L.SoftmaxWithLoss(net.upscore1, net.roi_label,
			phase=0,
			loss_weight=0.5,
			loss_param=dict(ignore_label=ignore_label), propagate_down=[1,0,0])
	############ u0 ############
	net.u0a_roi_warping_d0c_conv = L.ROIWarping(net.d0c_conv, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
	### a ### Fourth Deconvolution
	net.u0a_dconv_roi, net.u0a_bn_roi, net.u0a_scale_roi, net.u0a_relu_roi = deconv_bn_relu(net.u1f_relu_roi, 64, pad=0, kernel_size=2, stride=2)
	############ roi warping ############
	net.u0b_concat_roi = L.Concat(net.u0a_relu_roi, net.u0a_roi_warping_d0c_conv, axis=1)
	net.u0c_conv_roi, net.u0c_bn_roi, net.u0c_scale_roi, net.u0c_relu_roi = conv_bn_relu(net.u0b_concat_roi, 64, pad=1, kernel_size=3, stride=1)
	net.u0d_conv_roi, net.u0d_bn_roi, net.u0d_scale_roi = conv_bn(net.u0c_relu_roi, 64, pad=1, kernel_size=3, stride=1)
	net.u0e_conv_roi, net.u0e_bn_roi, net.u0e_scale_roi, net.u0e_eltw_roi, net.u0e_relu_roi = add_layer(net.u0b_concat_roi, net.u0d_conv_roi, 64)
	### loss 0
	net.score = L.Convolution(net.u0e_relu_roi,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	if phase == "train":
		net.loss = L.WeightedSoftmaxWithLoss(net.score, net.roi_label, net.roi_label_weight,
		# net.loss = L.SoftmaxWithLoss(net.score, net.roi_label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label), propagate_down=[1,0,0])
	else:
		net.roi_score_reconstruction = L.ROIPatchReconstruction(net.score, net.roi_coordinate, height=dim_data[2], width=dim_data[2])
		net.prob = L.Softmax(net.roi_score_reconstruction, axis=1)
		# net.probtmp = L.Softmax(net.score, axis=1)
		# net.prob = L.ROIPatchReconstruction(net.probtmp, net.roi_coordinate, height=dim_data[2], width=dim_data[2])
	return net.to_proto()

def roinet_2d_bn_d4(dim_data, dim_label, num_class, phase='train'):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))
	net.d0b_conv, net.d0b_bn, net.d0b_scale, net.d0b_relu = conv_bn_relu(net.data, 64, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d0c_conv, net.d0c_bn, net.d0c_scale, net.d0c_relu = conv_bn_relu(net.d0b_relu, 64, pad=1, kernel_size=3, stride=1, update_param=False)
	############ d1 ############
	# net.d1a_pool = max_pool(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1a_pool, net.d1a_conv, net.d1a_bn, net.d1a_scale, net.d1a_relu, net.d1a_concat = pool_conv_bn_relu_concat(net.d0c_relu, 64, update_param=False)
	net.d1b_conv, net.d1b_bn, net.d1b_scale, net.d1b_relu = conv_bn_relu(net.d1a_concat, 128, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d1c_conv, net.d1c_bn, net.d1c_scale = conv_bn(net.d1b_relu, 128, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d1c_conv, net.d1c_bn, net.d1c_scale, net.d1c_relu = conv_bn_relu(net.d1b_relu, 128, pad=1, kernel_size=3, stride=1)
	# net.d1d_conv, net.d1d_bn, net.d1d_scale, net.d1d_relu = conv_bn_relu(net.d1c_relu, 256, pad=1, kernel_size=3, stride=1)
	# net.d1e_eltw = L.Eltwise(net.d1a_concat, net.d1c_relu, eltwise_param=dict(operation=1))
	net.d1e_conv, net.d1e_bn, net.d1e_scale, net.d1e_eltw, net.d1e_relu = add_layer(net.d1a_concat, net.d1c_conv, 128, update_param=False)
	
	############ d2 ############
	# net.d2a_pool = max_pool(net.d1d_relu, pad=0, kernel_size=2, stride=2)
	net.d2a_pool, net.d2a_conv, net.d2a_bn, net.d2a_scale, net.d2a_relu, net.d2a_concat = pool_conv_bn_relu_concat(net.d1e_relu, 128, update_param=False)
	net.d2b_conv, net.d2b_bn, net.d2b_scale, net.d2b_relu = conv_bn_relu(net.d2a_concat, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d2c_conv, net.d2c_bn, net.d2c_scale, net.d2c_relu = conv_bn_relu(net.d2b_relu, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d2d_conv, net.d2d_bn, net.d2d_scale, net.d2d_relu = conv_bn_relu(net.d2c_relu, 256, pad=1, kernel_size=3, stride=1)
	# net.d2e_eltw = L.Eltwise(net.d2a_concat, net.d2d_relu, eltwise_param=dict(operation=1))
	net.d2d_conv, net.d2d_bn, net.d2d_scale = conv_bn(net.d2c_relu, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d2e_conv, net.d2e_bn, net.d2e_scale, net.d2e_eltw, net.d2e_relu = add_layer(net.d2a_concat, net.d2d_conv, 256, update_param=False)
	
	############ d3 ############
	# net.d3a_pool = max_pool(net.d2d_relu, pad=0, kernel_size=2, stride=2)
	net.d3a_pool, net.d3a_conv, net.d3a_bn, net.d3a_scale, net.d3a_relu, net.d3a_concat = pool_conv_bn_relu_concat(net.d2e_relu, 256, update_param=False)
	net.d3b_conv, net.d3b_bn, net.d3b_scale, net.d3b_relu = conv_bn_relu(net.d3a_concat, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d3c_conv, net.d3c_bn, net.d3c_scale, net.d3c_relu = conv_bn_relu(net.d3b_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d3d_conv, net.d3d_bn, net.d3d_scale, net.d3d_relu = conv_bn_relu(net.d3c_relu, 512, pad=1, kernel_size=3, stride=1)
	# net.d3e_eltw = L.Eltwise(net.d3a_concat, net.d3d_relu, eltwise_param=dict(operation=1))
	net.d3d_conv, net.d3d_bn, net.d3d_scale = conv_bn(net.d3c_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d3e_conv, net.d3e_bn, net.d3e_scale, net.d3e_eltw, net.d3e_relu = add_layer(net.d3a_concat, net.d3d_conv, 512, update_param=False)
	
	############ d4 ############
	# net.d4a_pool = max_pool(net.d3d_relu, pad=0, kernel_size=2, stride=2)
	net.d4a_pool, net.d4a_conv, net.d4a_bn, net.d4a_scale, net.d4a_relu, net.d4a_concat = pool_conv_bn_relu_concat(net.d3e_relu, 512, update_param=False)
	net.d4b_conv, net.d4b_bn, net.d4b_scale, net.d4b_relu = conv_bn_relu(net.d4a_concat, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d4c_conv, net.d4c_bn, net.d4c_scale, net.d4c_relu = conv_bn_relu(net.d4b_relu, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d4d_conv, net.d4d_bn, net.d4d_scale, net.d4d_relu = conv_bn_relu(net.d4c_relu, 1024, pad=1, kernel_size=3, stride=1)
	# net.d4e_eltw = L.Eltwise(net.d4a_concat, net.d4d_relu, eltwise_param=dict(operation=1))
	net.d4d_conv, net.d4d_bn, net.d4d_scale = conv_bn(net.d4c_relu, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d4e_conv, net.d4e_bn, net.d4e_scale, net.d4e_eltw, net.d4e_relu = add_layer(net.d4a_concat, net.d4d_conv, 1024, update_param=False)
	
	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_bn, net.u3a_scale, net.u3a_relu = deconv_bn_relu(net.d4e_relu, 512, pad=0, kernel_size=2, stride=2, update_param=False)
	net.u3b_concat = L.Concat(net.u3a_relu, net.d3e_relu, axis=1, propagate_down=[0,0])
	net.u3c_conv, net.u3c_bn, net.u3c_scale, net.u3c_relu = conv_bn_relu(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3d_conv, net.u3d_bn, net.u3d_scale, net.u3d_relu = conv_bn_relu(net.u3c_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3e_conv, net.u3e_bn, net.u3e_scale = conv_bn(net.u3d_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3f_conv, net.u3f_bn, net.u3f_scale, net.u3f_eltw, net.u3f_relu = add_layer(net.u3b_concat, net.u3e_conv, 512, update_param=False)
	############ score3 ###########
	net.score3 = L.Convolution(net.u3f_relu,
		param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine, propagate_down=0)
	net.upscore3 = L.Deconvolution(net.score3,
		param=[dict(lr_mult=0, decay_mult=0)],
		convolution_param=dict(num_output=num_class,  pad=4, kernel_size=16, stride=8,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine), propagate_down=0)
	# if phase == "train":
	# 	# net.loss3 = L.WeightedSoftmaxWithLoss(net.upscore3, net.label, net.label_weight,
	# 	net.loss3 = L.SoftmaxWithLoss(net.upscore3, net.label,
	# 		phase=0,
	# 		loss_weight=0.0625,
	# 		loss_param=dict(ignore_label=ignore_label))

	############ get roi ############
	net.prob3 = L.Softmax(net.upscore3, axis=1, propagate_down=0)
	net.roi_mask, net.roi_coordinate = L.ROICoordinate(net.prob3, pad=16, threshold=0.80, ntop=2, propagate_down=0)
	# net.slicence = L.Silence(net.roi_mask, ntop=0, propagate_down=0)
	if phase == "train":
		net.slicence = L.Silence(net.roi_mask, ntop=0, propagate_down=0)
	if phase == "train":
		# net.roi_label = L.ROIWarping(net.label, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
		net.label_map = L.ROIWarping(net.label, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
		net.roi_label = L.MultipleThreshold(net.label_map, threshold_point=[0.5,1.5], threshold_value=[0,1,2], propagate_down=0)
	
	############ roi warping ############
	net.roi_warping_d4e_relu = L.ROIWarping(net.d4e_relu, net.roi_coordinate, pooled_h=dim_data[2]/8, pooled_w=dim_data[2]/8, spatial_scale=0.0625, propagate_down=[0,0])
	net.roi_warping_d3e_relu = L.ROIWarping(net.d3e_relu, net.roi_coordinate, pooled_h=dim_data[2]/8, pooled_w=dim_data[2]/8, spatial_scale=0.125, propagate_down=[0,0])
	net.roi_warping_concat = L.Concat(net.roi_warping_d4e_relu, net.roi_warping_d3e_relu, axis=1)
	net.u3c_conv_roi, net.u3c_bn_roi, net.u3c_scale_roi, net.u3c_relu_roi = conv_bn_relu(net.roi_warping_concat, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3d_conv_roi, net.u3d_bn_roi, net.u3d_scale_roi, net.u3d_relu_roi = conv_bn_relu(net.u3c_relu_roi, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3e_conv_roi, net.u3e_bn_roi, net.u3e_scale_roi = conv_bn(net.u3d_relu_roi, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3f_conv_roi, net.u3f_bn_roi, net.u3f_scale_roi, net.u3f_eltw_roi, net.u3f_relu_roi = add_layer(net.roi_warping_concat, net.u3e_conv_roi, 512, update_param=False)

	############ u2 ############
	net.u2a_roi_warping_d2e_relu = L.ROIWarping(net.d2e_relu, net.roi_coordinate, pooled_h=dim_data[2]/4, pooled_w=dim_data[2]/4, spatial_scale=0.25, propagate_down=[0,0])
	### a ### Second Deconvolution
	net.u2a_dconv_roi, net.u2a_bn_roi, net.u2a_scale_roi, net.u2a_relu_roi = deconv_bn_relu(net.u3f_relu_roi, 256, pad=0, kernel_size=2, stride=2)
	net.u2b_concat_roi = L.Concat(net.u2a_relu_roi, net.u2a_roi_warping_d2e_relu, axis=1)
	net.u2c_conv_roi, net.u2c_bn_roi, net.u2c_scale_roi, net.u2c_relu_roi = conv_bn_relu(net.u2b_concat_roi, 256, pad=1, kernel_size=3, stride=1)
	net.u2d_conv_roi, net.u2d_bn_roi, net.u2d_scale_roi, net.u2d_relu_roi = conv_bn_relu(net.u2c_relu_roi, 256, pad=1, kernel_size=3, stride=1)
	net.u2e_conv_roi, net.u2e_bn_roi, net.u2e_scale_roi = conv_bn(net.u2d_relu_roi, 256, pad=1, kernel_size=3, stride=1)
	net.u2f_conv_roi, net.u2f_bn_roi, net.u2f_scale_roi, net.u2f_eltw_roi, net.u2f_relu_roi = add_layer(net.u2b_concat_roi, net.u2e_conv_roi, 256)
	# ### loss 2
	# net.score2 = L.Convolution(net.u2f_relu,
	# 	param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
	# 	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	# net.upscore2 = L.Deconvolution(net.score2,
	# 	param=[dict(lr_mult=10, decay_mult=1)],
	# 	convolution_param=dict(num_output=num_class,  pad=2, kernel_size=8, stride=4,
	# 		weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	# if phase == "train":
	# 	# net.loss2 = L.WeightedSoftmaxWithLoss(net.upscore2, net.label, net.label_weight,
	# 	net.loss2 = L.SoftmaxWithLoss(net.upscore2, net.roi_label,
	# 		phase=0,
	# 		loss_weight=0.125,
	# 		loss_param=dict(ignore_label=ignore_label))

	############ u1 ############
	net.u1a_roi_warping_d1e_relu = L.ROIWarping(net.d1e_relu, net.roi_coordinate, pooled_h=dim_data[2]/2, pooled_w=dim_data[2]/2, spatial_scale=0.5, propagate_down=[0,0])
	### a ### Third Deconvolution
	net.u1a_dconv_roi, net.u1a_bn_roi, net.u1a_scale_roi, net.u1a_relu_roi = deconv_bn_relu(net.u2f_relu_roi, 128, pad=0, kernel_size=2, stride=2)
	net.u1b_concat_roi = L.Concat(net.u1a_relu_roi, net.u1a_roi_warping_d1e_relu, axis=1)
	net.u1c_conv_roi, net.u1c_bn_roi, net.u1c_scale_roi, net.u1c_relu_roi = conv_bn_relu(net.u1b_concat_roi, 128, pad=1, kernel_size=3, stride=1)
	# net.u1d_conv, net.u1d_bn, net.u1d_scale, net.u1d_relu = conv_bn_relu(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1)
	net.u1e_conv_roi, net.u1e_bn_roi, net.u1e_scale_roi = conv_bn(net.u1c_relu_roi, 128, pad=1, kernel_size=3, stride=1)
	net.u1f_conv_roi, net.u1f_bn_roi, net.u1f_scale_roi, net.u1f_eltw_roi, net.u1f_relu_roi = add_layer(net.u1b_concat_roi, net.u1e_conv_roi, 128)
	# ### loss 1
	# net.score1 = L.Convolution(net.u1f_relu,
	# 	param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
	# 	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	# net.upscore1 = L.Deconvolution(net.score1,
	# 	param=[dict(lr_mult=10, decay_mult=1)],
	# 	convolution_param=dict(num_output=num_class,  pad=1, kernel_size=4, stride=2,
	# 		weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	# if phase == "train":
	# 	# net.loss1 = L.WeightedSoftmaxWithLoss(net.upscore1, net.label, net.label_weight,
	# 	net.loss1 = L.SoftmaxWithLoss(net.upscore1, net.roi_label,
	# 		phase=0,
	# 		loss_weight=0.25,
	# 		loss_param=dict(ignore_label=ignore_label), propagate_down=[1,0])
	############ u0 ############
	net.u0a_roi_warping_d0c_conv = L.ROIWarping(net.d0c_conv, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
	### a ### Fourth Deconvolution
	net.u0a_dconv_roi, net.u0a_bn_roi, net.u0a_scale_roi, net.u0a_relu_roi = deconv_bn_relu(net.u1f_relu_roi, 64, pad=0, kernel_size=2, stride=2)
	############ roi warping ############
	net.u0b_concat_roi = L.Concat(net.u0a_relu_roi, net.u0a_roi_warping_d0c_conv, axis=1)
	net.u0c_conv_roi, net.u0c_bn_roi, net.u0c_scale_roi, net.u0c_relu_roi = conv_bn_relu(net.u0b_concat_roi, 64, pad=1, kernel_size=3, stride=1)
	net.u0d_conv_roi, net.u0d_bn_roi, net.u0d_scale_roi = conv_bn(net.u0c_relu_roi, 64, pad=1, kernel_size=3, stride=1)
	net.u0e_conv_roi, net.u0e_bn_roi, net.u0e_scale_roi, net.u0e_eltw_roi, net.u0e_relu_roi = add_layer(net.u0b_concat_roi, net.u0d_conv_roi, 64)
	### loss 0
	net.score = L.Convolution(net.u0e_relu_roi,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	if phase == "train":
		# net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
		net.loss = L.SoftmaxWithLoss(net.score, net.roi_label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label), propagate_down=[1,0])
	else:
		# net.prob = L.Softmax(net.score, axis=1)
		net.roi_score_reconstruction = L.ROIPatchReconstruction(net.score, net.roi_coordinate, height=dim_data[2], width=dim_data[2])
		net.prob = L.Softmax(net.roi_score_reconstruction, axis=1)
	return net.to_proto()

def roinet_2d_bn_d4_weighted(dim_data, dim_label, num_class, phase='train'):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))
	net.d0b_conv, net.d0b_bn, net.d0b_scale, net.d0b_relu = conv_bn_relu(net.data, 64, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d0c_conv, net.d0c_bn, net.d0c_scale, net.d0c_relu = conv_bn_relu(net.d0b_relu, 64, pad=1, kernel_size=3, stride=1, update_param=False)
	############ d1 ############
	# net.d1a_pool = max_pool(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1a_pool, net.d1a_conv, net.d1a_bn, net.d1a_scale, net.d1a_relu, net.d1a_concat = pool_conv_bn_relu_concat(net.d0c_relu, 64, update_param=False)
	net.d1b_conv, net.d1b_bn, net.d1b_scale, net.d1b_relu = conv_bn_relu(net.d1a_concat, 128, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d1c_conv, net.d1c_bn, net.d1c_scale = conv_bn(net.d1b_relu, 128, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d1c_conv, net.d1c_bn, net.d1c_scale, net.d1c_relu = conv_bn_relu(net.d1b_relu, 128, pad=1, kernel_size=3, stride=1)
	# net.d1d_conv, net.d1d_bn, net.d1d_scale, net.d1d_relu = conv_bn_relu(net.d1c_relu, 256, pad=1, kernel_size=3, stride=1)
	# net.d1e_eltw = L.Eltwise(net.d1a_concat, net.d1c_relu, eltwise_param=dict(operation=1))
	net.d1e_conv, net.d1e_bn, net.d1e_scale, net.d1e_eltw, net.d1e_relu = add_layer(net.d1a_concat, net.d1c_conv, 128, update_param=False)
	
	############ d2 ############
	# net.d2a_pool = max_pool(net.d1d_relu, pad=0, kernel_size=2, stride=2)
	net.d2a_pool, net.d2a_conv, net.d2a_bn, net.d2a_scale, net.d2a_relu, net.d2a_concat = pool_conv_bn_relu_concat(net.d1e_relu, 128, update_param=False)
	net.d2b_conv, net.d2b_bn, net.d2b_scale, net.d2b_relu = conv_bn_relu(net.d2a_concat, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d2c_conv, net.d2c_bn, net.d2c_scale, net.d2c_relu = conv_bn_relu(net.d2b_relu, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d2d_conv, net.d2d_bn, net.d2d_scale, net.d2d_relu = conv_bn_relu(net.d2c_relu, 256, pad=1, kernel_size=3, stride=1)
	# net.d2e_eltw = L.Eltwise(net.d2a_concat, net.d2d_relu, eltwise_param=dict(operation=1))
	net.d2d_conv, net.d2d_bn, net.d2d_scale = conv_bn(net.d2c_relu, 256, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d2e_conv, net.d2e_bn, net.d2e_scale, net.d2e_eltw, net.d2e_relu = add_layer(net.d2a_concat, net.d2d_conv, 256, update_param=False)
	
	############ d3 ############
	# net.d3a_pool = max_pool(net.d2d_relu, pad=0, kernel_size=2, stride=2)
	net.d3a_pool, net.d3a_conv, net.d3a_bn, net.d3a_scale, net.d3a_relu, net.d3a_concat = pool_conv_bn_relu_concat(net.d2e_relu, 256, update_param=False)
	net.d3b_conv, net.d3b_bn, net.d3b_scale, net.d3b_relu = conv_bn_relu(net.d3a_concat, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d3c_conv, net.d3c_bn, net.d3c_scale, net.d3c_relu = conv_bn_relu(net.d3b_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d3d_conv, net.d3d_bn, net.d3d_scale, net.d3d_relu = conv_bn_relu(net.d3c_relu, 512, pad=1, kernel_size=3, stride=1)
	# net.d3e_eltw = L.Eltwise(net.d3a_concat, net.d3d_relu, eltwise_param=dict(operation=1))
	net.d3d_conv, net.d3d_bn, net.d3d_scale = conv_bn(net.d3c_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d3e_conv, net.d3e_bn, net.d3e_scale, net.d3e_eltw, net.d3e_relu = add_layer(net.d3a_concat, net.d3d_conv, 512, update_param=False)
	
	############ d4 ############
	# net.d4a_pool = max_pool(net.d3d_relu, pad=0, kernel_size=2, stride=2)
	net.d4a_pool, net.d4a_conv, net.d4a_bn, net.d4a_scale, net.d4a_relu, net.d4a_concat = pool_conv_bn_relu_concat(net.d3e_relu, 512, update_param=False)
	net.d4b_conv, net.d4b_bn, net.d4b_scale, net.d4b_relu = conv_bn_relu(net.d4a_concat, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d4c_conv, net.d4c_bn, net.d4c_scale, net.d4c_relu = conv_bn_relu(net.d4b_relu, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	# net.d4d_conv, net.d4d_bn, net.d4d_scale, net.d4d_relu = conv_bn_relu(net.d4c_relu, 1024, pad=1, kernel_size=3, stride=1)
	# net.d4e_eltw = L.Eltwise(net.d4a_concat, net.d4d_relu, eltwise_param=dict(operation=1))
	net.d4d_conv, net.d4d_bn, net.d4d_scale = conv_bn(net.d4c_relu, 1024, pad=1, kernel_size=3, stride=1, update_param=False)
	net.d4e_conv, net.d4e_bn, net.d4e_scale, net.d4e_eltw, net.d4e_relu = add_layer(net.d4a_concat, net.d4d_conv, 1024, update_param=False)
	
	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_bn, net.u3a_scale, net.u3a_relu = deconv_bn_relu(net.d4e_relu, 512, pad=0, kernel_size=2, stride=2, update_param=False)
	net.u3b_concat = L.Concat(net.u3a_relu, net.d3e_relu, axis=1, propagate_down=[0,0])
	net.u3c_conv, net.u3c_bn, net.u3c_scale, net.u3c_relu = conv_bn_relu(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3d_conv, net.u3d_bn, net.u3d_scale, net.u3d_relu = conv_bn_relu(net.u3c_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3e_conv, net.u3e_bn, net.u3e_scale = conv_bn(net.u3d_relu, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3f_conv, net.u3f_bn, net.u3f_scale, net.u3f_eltw, net.u3f_relu = add_layer(net.u3b_concat, net.u3e_conv, 512, update_param=False)
	############ score3 ###########
	net.score3 = L.Convolution(net.u3f_relu,
		param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine, propagate_down=0)
	net.upscore3 = L.Deconvolution(net.score3,
		param=[dict(lr_mult=0, decay_mult=0)],
		convolution_param=dict(num_output=num_class,  pad=4, kernel_size=16, stride=8,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine), propagate_down=0)
	# if phase == "train":
	# 	# net.loss3 = L.WeightedSoftmaxWithLoss(net.upscore3, net.label, net.label_weight,
	# 	net.loss3 = L.SoftmaxWithLoss(net.upscore3, net.label,
	# 		phase=0,
	# 		loss_weight=0.0625,
	# 		loss_param=dict(ignore_label=ignore_label))

	############ get roi ############
	net.prob3 = L.Softmax(net.upscore3, axis=1, propagate_down=0)
	net.roi_mask, net.roi_coordinate = L.ROICoordinate(net.prob3, pad=16, threshold=0.80, ntop=2, propagate_down=0)
	# net.slicence = L.Silence(net.roi_mask, ntop=0, propagate_down=0)
	if phase == "train":
		net.slicence = L.Silence(net.roi_mask, ntop=0, propagate_down=0)
	if phase == "train":
		# net.roi_label = L.ROIWarping(net.label, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
		net.label_map = L.ROIWarping(net.label, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
		net.roi_label = L.MultipleThreshold(net.label_map, threshold_point=[0.5,1.5], threshold_value=[0,1,2], propagate_down=0)
		net.roi_label_weight = L.ROIWarping(net.label_weight, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
	
	############ roi warping ############
	net.roi_warping_d4e_relu = L.ROIWarping(net.d4e_relu, net.roi_coordinate, pooled_h=dim_data[2]/8, pooled_w=dim_data[2]/8, spatial_scale=0.0625, propagate_down=[0,0])
	net.roi_warping_d3e_relu = L.ROIWarping(net.d3e_relu, net.roi_coordinate, pooled_h=dim_data[2]/8, pooled_w=dim_data[2]/8, spatial_scale=0.125, propagate_down=[0,0])
	net.roi_warping_concat = L.Concat(net.roi_warping_d4e_relu, net.roi_warping_d3e_relu, axis=1)
	net.u3c_conv_roi, net.u3c_bn_roi, net.u3c_scale_roi, net.u3c_relu_roi = conv_bn_relu(net.roi_warping_concat, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3d_conv_roi, net.u3d_bn_roi, net.u3d_scale_roi, net.u3d_relu_roi = conv_bn_relu(net.u3c_relu_roi, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3e_conv_roi, net.u3e_bn_roi, net.u3e_scale_roi = conv_bn(net.u3d_relu_roi, 512, pad=1, kernel_size=3, stride=1, update_param=False)
	net.u3f_conv_roi, net.u3f_bn_roi, net.u3f_scale_roi, net.u3f_eltw_roi, net.u3f_relu_roi = add_layer(net.roi_warping_concat, net.u3e_conv_roi, 512, update_param=False)

	############ u2 ############
	net.u2a_roi_warping_d2e_relu = L.ROIWarping(net.d2e_relu, net.roi_coordinate, pooled_h=dim_data[2]/4, pooled_w=dim_data[2]/4, spatial_scale=0.25, propagate_down=[0,0])
	### a ### Second Deconvolution
	net.u2a_dconv_roi, net.u2a_bn_roi, net.u2a_scale_roi, net.u2a_relu_roi = deconv_bn_relu(net.u3f_relu_roi, 256, pad=0, kernel_size=2, stride=2)
	net.u2b_concat_roi = L.Concat(net.u2a_relu_roi, net.u2a_roi_warping_d2e_relu, axis=1)
	net.u2c_conv_roi, net.u2c_bn_roi, net.u2c_scale_roi, net.u2c_relu_roi = conv_bn_relu(net.u2b_concat_roi, 256, pad=1, kernel_size=3, stride=1)
	net.u2d_conv_roi, net.u2d_bn_roi, net.u2d_scale_roi, net.u2d_relu_roi = conv_bn_relu(net.u2c_relu_roi, 256, pad=1, kernel_size=3, stride=1)
	net.u2e_conv_roi, net.u2e_bn_roi, net.u2e_scale_roi = conv_bn(net.u2d_relu_roi, 256, pad=1, kernel_size=3, stride=1)
	net.u2f_conv_roi, net.u2f_bn_roi, net.u2f_scale_roi, net.u2f_eltw_roi, net.u2f_relu_roi = add_layer(net.u2b_concat_roi, net.u2e_conv_roi, 256)
	# ### loss 2
	net.score2 = L.Convolution(net.u2f_relu_roi,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore2 = L.Deconvolution(net.score2,
		param=[dict(lr_mult=10, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=2, kernel_size=8, stride=4,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss2 = L.WeightedSoftmaxWithLoss(net.upscore2, net.roi_label, net.roi_label_weight,
		# net.loss2 = L.SoftmaxWithLoss(net.upscore2, net.roi_label,
			phase=0,
			loss_weight=0.25,
			loss_param=dict(ignore_label=ignore_label), propagate_down=[1,0,0])

	############ u1 ############
	net.u1a_roi_warping_d1e_relu = L.ROIWarping(net.d1e_relu, net.roi_coordinate, pooled_h=dim_data[2]/2, pooled_w=dim_data[2]/2, spatial_scale=0.5, propagate_down=[0,0])
	### a ### Third Deconvolution
	net.u1a_dconv_roi, net.u1a_bn_roi, net.u1a_scale_roi, net.u1a_relu_roi = deconv_bn_relu(net.u2f_relu_roi, 128, pad=0, kernel_size=2, stride=2)
	net.u1b_concat_roi = L.Concat(net.u1a_relu_roi, net.u1a_roi_warping_d1e_relu, axis=1)
	net.u1c_conv_roi, net.u1c_bn_roi, net.u1c_scale_roi, net.u1c_relu_roi = conv_bn_relu(net.u1b_concat_roi, 128, pad=1, kernel_size=3, stride=1)
	# net.u1d_conv, net.u1d_bn, net.u1d_scale, net.u1d_relu = conv_bn_relu(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1)
	net.u1e_conv_roi, net.u1e_bn_roi, net.u1e_scale_roi = conv_bn(net.u1c_relu_roi, 128, pad=1, kernel_size=3, stride=1)
	net.u1f_conv_roi, net.u1f_bn_roi, net.u1f_scale_roi, net.u1f_eltw_roi, net.u1f_relu_roi = add_layer(net.u1b_concat_roi, net.u1e_conv_roi, 128)
	# ### loss 1
	net.score1 = L.Convolution(net.u1f_relu_roi,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore1 = L.Deconvolution(net.score1,
		param=[dict(lr_mult=10, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=1, kernel_size=4, stride=2,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss1 = L.WeightedSoftmaxWithLoss(net.upscore1, net.roi_label, net.roi_label_weight,
		# net.loss1 = L.SoftmaxWithLoss(net.upscore1, net.roi_label,
			phase=0,
			loss_weight=0.5,
			loss_param=dict(ignore_label=ignore_label), propagate_down=[1,0,0])
	############ u0 ############
	net.u0a_roi_warping_d0c_conv = L.ROIWarping(net.d0c_conv, net.roi_coordinate, pooled_h=dim_data[2], pooled_w=dim_data[2], spatial_scale=1, propagate_down=[0,0])
	### a ### Fourth Deconvolution
	net.u0a_dconv_roi, net.u0a_bn_roi, net.u0a_scale_roi, net.u0a_relu_roi = deconv_bn_relu(net.u1f_relu_roi, 64, pad=0, kernel_size=2, stride=2)
	############ roi warping ############
	net.u0b_concat_roi = L.Concat(net.u0a_relu_roi, net.u0a_roi_warping_d0c_conv, axis=1)
	net.u0c_conv_roi, net.u0c_bn_roi, net.u0c_scale_roi, net.u0c_relu_roi = conv_bn_relu(net.u0b_concat_roi, 64, pad=1, kernel_size=3, stride=1)
	net.u0d_conv_roi, net.u0d_bn_roi, net.u0d_scale_roi = conv_bn(net.u0c_relu_roi, 64, pad=1, kernel_size=3, stride=1)
	net.u0e_conv_roi, net.u0e_bn_roi, net.u0e_scale_roi, net.u0e_eltw_roi, net.u0e_relu_roi = add_layer(net.u0b_concat_roi, net.u0d_conv_roi, 64)
	### loss 0
	net.score = L.Convolution(net.u0e_relu_roi,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	if phase == "train":
		net.loss = L.WeightedSoftmaxWithLoss(net.score, net.roi_label, net.roi_label_weight,
		# net.loss = L.SoftmaxWithLoss(net.score, net.roi_label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label), propagate_down=[1,0,0])
	else:
		net.roi_score_reconstruction = L.ROIPatchReconstruction(net.score, net.roi_coordinate, height=dim_data[2], width=dim_data[2])
		net.prob = L.Softmax(net.roi_score_reconstruction, axis=1)
		# net.probtmp = L.Softmax(net.score, axis=1)
		# net.prob = L.ROIPatchReconstruction(net.probtmp, net.roi_coordinate, height=dim_data[2], width=dim_data[2])
	return net.to_proto()

def make_roinet(net, dim_data, dim_label, num_class, prototxt_train, prototxt_test):
	# register net
	__nets = ['roinet_2d_bn_u2', 'roinet_2d_bn_u3', 'roinet_2d_bn_u3_weighted', 'roinet_2d_bn_d3','roinet_2d_bn_d3_weighted', 'roinet_2d_bn_d4', 'roinet_2d_bn_d4_weighted']
	assert net in __nets, 'Unknown net: {}'.format(net)
	global use_global_stats, engine, ignore_label
	engine = 2
	ignore_label = 255
	if net == 'roinet_2d_bn_u2':
		use_global_stats = 0
		train_net = roinet_2d_bn_u2(dim_data, dim_label, num_class, phase='train')
		use_global_stats = 1
		dim_data[0] = 1
		test_net = roinet_2d_bn_u2(dim_data, dim_label, num_class, phase='test')

	if net == 'roinet_2d_bn_u3':
		use_global_stats = 0
		train_net = roinet_2d_bn_u3(dim_data, dim_label, num_class, phase='train')
		use_global_stats = 1
		dim_data[0] = 1
		test_net = roinet_2d_bn_u3(dim_data, dim_label, num_class, phase='test')

	if net == 'roinet_2d_bn_u3_weighted':
		use_global_stats = 0
		train_net = roinet_2d_bn_u3_weighted(dim_data, dim_label, num_class, phase='train')
		use_global_stats = 1
		test_net = roinet_2d_bn_u3_weighted(dim_data, dim_label, num_class, phase='test')

	if net == 'roinet_2d_bn_d3':
		use_global_stats = 0
		train_net = roinet_2d_bn_d3(dim_data, dim_label, num_class, phase='train')
		use_global_stats = 1
		dim_data[0] = 1
		test_net = roinet_2d_bn_d3(dim_data, dim_label, num_class, phase='test')

	if net == 'roinet_2d_bn_d3_weighted':
		use_global_stats = 0
		train_net = roinet_2d_bn_d3_weighted(dim_data, dim_label, num_class, phase='train')
		use_global_stats = 1
		dim_data[0] = 1
		test_net = roinet_2d_bn_d3_weighted(dim_data, dim_label, num_class, phase='test')

	if net == 'roinet_2d_bn_d4':
		use_global_stats = 0
		train_net = roinet_2d_bn_d4(dim_data, dim_label, num_class, phase='train')
		use_global_stats = 1
		dim_data[0] = 1
		test_net = roinet_2d_bn_d4(dim_data, dim_label, num_class, phase='test')

	if net == 'roinet_2d_bn_d4_weighted':
		use_global_stats = 0
		train_net = roinet_2d_bn_d4_weighted(dim_data, dim_label, num_class, phase='train')
		use_global_stats = 1
		dim_data[0] = 1
		test_net = roinet_2d_bn_d4_weighted(dim_data, dim_label, num_class, phase='test')

	with open(prototxt_train, 'w') as f:
		f.write(str(train_net))
	with open(prototxt_test, 'w') as f:
		f.write(str(test_net))