import caffe
from caffe import layers as L, params as P, to_proto
from caffe.coord_map import crop # crop(net.upscore, net.data) automatically calculate the axis and offset

# def bn_relu_conv(bottom, num_output, pad=0, kernel_size=3, stride=1):
# 	bn = L.BatchNorm(bottom, in_place=False, use_global_stats=use_global_stats, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
# 	scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
# 	relu = L.ReLU(scale, in_place=True, engine=engine)
# 	conv = L.Convolution(relu, num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
# 		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],		
# 		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
# 		engine=engine)
# 	return bn, scale, relu, conv

def conv_bn(bottom, num_output, pad=0, kernel_size=3, stride=1):
	conv = L.Convolution(bottom, num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],		
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=engine)
	bn = L.BatchNorm(conv, in_place=True, use_global_stats=use_global_stats, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	return conv, bn, scale

def conv_bn_relu(bottom, num_output, pad=0, kernel_size=3, stride=1):
	conv = L.Convolution(bottom, num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],		
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=engine)
	bn = L.BatchNorm(conv, in_place=True, use_global_stats=use_global_stats, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	relu = L.ReLU(scale, in_place=True, engine=engine)
	return conv, bn, scale, relu

def deconv_bn_relu(bottom, num_output, pad=0, kernel_size=3, stride=1):
	deconv = L.Deconvolution(bottom, param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_output,  pad=pad, kernel_size=kernel_size, stride=stride,
			weight_filler=dict(type='msra'), bias_term=0,
			engine=engine))
	bn = L.BatchNorm(deconv, in_place=True, use_global_stats=use_global_stats, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	relu = L.ReLU(scale, in_place=True, engine=engine)
	return deconv, bn, scale, relu

def add_layer(bottom1, bottom2, num_output):
	conv, bn, scale = conv_bn(bottom1, num_output=num_output, pad=0, kernel_size=1, stride=1)
	eltw = L.Eltwise(conv, bottom2, eltwise_param=dict(operation=1))
	rule = L.ReLU(eltw, in_place=True, engine=engine)
	return conv, bn, scale, eltw, rule

# def add_relu(bottom1, bottom2):
# 	eltw = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1))
# 	rule = L.ReLU(eltw, in_place=True, engine=engine)
# 	return eltw, rule

# def conv1_conv2_add_bn_relu(bottom1, bottom2, num_output, pad=0, kernel_size=3, stride=1):
# 	conv1 = L.Convolution(bottom1, num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
# 		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],		
# 		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
# 		engine=engine)
# 	conv2 = L.Convolution(bottom2, num_output=num_output, pad=0, kernel_size=1, stride=1,
# 		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],		
# 		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
# 		engine=engine)
# 	eltw = L.Eltwise(conv1, conv2, eltwise_param=dict(operation=1))
# 	bn = L.BatchNorm(eltw, in_place=True, use_global_stats=use_global_stats, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
# 	scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
# 	relu = L.ReLU(scale, in_place=True, engine=engine)
	
# 	return conv1, conv2, eltw, bn, scale, relu

def pool_conv_bn_relu_concat(bottom, num_output):
	pool = L.Pooling(bottom, pool=P.Pooling.MAX, pad=0, kernel_size=2, stride=2, engine=engine)
	conv, bn, scale, relu = conv_bn_relu(bottom, num_output, pad=1, kernel_size=3, stride=2)
	concat = L.Concat(pool, conv, axis=1)

	return pool, conv, bn, scale, relu, concat

# def max_pool(bottom, pad=0, kernel_size=2, stride=2):
# 	return L.Pooling(bottom, pool=P.Pooling.MAX, pad=pad, kernel_size=kernel_size, stride=stride, engine=engine)

def incept_2(bottom, num_output):
	# branch 1
	conv_b1_a, bn_b1_a, scale_b1_a = conv_bn(bottom, num_output, pad=1, kernel_size=3, stride=1)
	# add
	eltw = L.Eltwise(bottom, conv_b1_a, eltwise_param=dict(operation=1))
	rule = L.ReLU(eltw, in_place=True, engine=engine)

	return conv_b1_a, bn_b1_a, scale_b1_a, \
	eltw, rule

def incept_3(bottom, num_output):
	# branch 1
	conv_b1_a, bn_b1_a, scale_b1_a = conv_bn(bottom, num_output, pad=1, kernel_size=3, stride=1)
	# branch 2
	conv_b2_a, bn_b2_a, scale_b2_a, relu_b2_a = conv_bn_relu(bottom, num_output, pad=1, kernel_size=3, stride=1)
	conv_b2_b, bn_b2_b, scale_b2_b = conv_bn(relu_b2_a, num_output, pad=1, kernel_size=3, stride=1)
	# add
	eltw = L.Eltwise(bottom, conv_b1_a, conv_b2_b, eltwise_param=dict(operation=1))
	rule = L.ReLU(eltw, in_place=True, engine=engine)

	return conv_b1_a, bn_b1_a, scale_b1_a, \
	conv_b2_a, bn_b2_a, scale_b2_a, relu_b2_a, conv_b2_b, bn_b2_b, scale_b2_b, \
	eltw, rule

def incept_4(bottom, num_output):
	# branch 1
	conv_b1_a, bn_b1_a, scale_b1_a = conv_bn(bottom, num_output, pad=1, kernel_size=3, stride=1)
	# branch 2
	conv_b2_a, bn_b2_a, scale_b2_a, relu_b2_a = conv_bn_relu(bottom, num_output, pad=1, kernel_size=3, stride=1)
	conv_b2_b, bn_b2_b, scale_b2_b = conv_bn(relu_b2_a, num_output, pad=1, kernel_size=3, stride=1)
	# branch 3
	conv_b3_a, bn_b3_a, scale_b3_a, relu_b3_a = conv_bn_relu(bottom, num_output, pad=1, kernel_size=3, stride=1)
	conv_b3_b, bn_b3_b, scale_b3_b, relu_b3_b = conv_bn_relu(relu_b3_a, num_output, pad=1, kernel_size=3, stride=1)
	conv_b3_c, bn_b3_c, scale_b3_c = conv_bn(relu_b3_b, num_output, pad=1, kernel_size=3, stride=1)
	
	# add
	eltw = L.Eltwise(bottom, conv_b1_a, conv_b2_b, conv_b3_c, eltwise_param=dict(operation=1))
	relu = L.ReLU(eltw, in_place=True, engine=engine)

	return conv_b1_a, bn_b1_a, scale_b1_a, \
	conv_b2_a, bn_b2_a, scale_b2_a, relu_b2_a, conv_b2_b, bn_b2_b, scale_b2_b, \
	conv_b3_a, bn_b3_a, scale_b3_a, relu_b3_a, conv_b3_b, bn_b3_b, scale_b3_b, relu_b3_b, conv_b3_c, bn_b3_c, scale_b3_c, \
	eltw, relu

def incept_3_decode(bottom, num_output):
	# branch 1
	conv_b1_a, bn_b1_a, scale_b1_a = conv_bn(bottom, num_output, pad=0, kernel_size=1, stride=1)
	# branch 2
	conv_b2_a, bn_b2_a, scale_b2_a, relu_b2_a = conv_bn_relu(bottom, num_output, pad=1, kernel_size=3, stride=1)
	conv_b2_b, bn_b2_b, scale_b2_b = conv_bn(relu_b2_a, num_output, pad=1, kernel_size=3, stride=1)
	# add
	eltw = L.Eltwise(conv_b1_a, conv_b2_b, eltwise_param=dict(operation=1))
	rule = L.ReLU(eltw, in_place=True, engine=engine)

	return conv_b1_a, bn_b1_a, scale_b1_a, \
	conv_b2_a, bn_b2_a, scale_b2_a, relu_b2_a, conv_b2_b, bn_b2_b, scale_b2_b, \
	eltw, rule

def incept_4_decode(bottom, num_output):
	# branch 1
	conv_b1_a, bn_b1_a, scale_b1_a = conv_bn(bottom, num_output, pad=0, kernel_size=1, stride=1)
	# branch 2
	conv_b2_a, bn_b2_a, scale_b2_a, relu_b2_a = conv_bn_relu(bottom, num_output, pad=1, kernel_size=3, stride=1)
	conv_b2_b, bn_b2_b, scale_b2_b = conv_bn(relu_b2_a, num_output, pad=1, kernel_size=3, stride=1)
	# branch 3
	conv_b3_a, bn_b3_a, scale_b3_a, relu_b3_a = conv_bn_relu(bottom, num_output, pad=1, kernel_size=3, stride=1)
	conv_b3_b, bn_b3_b, scale_b3_b, relu_b3_b = conv_bn_relu(relu_b3_a, num_output, pad=1, kernel_size=3, stride=1)
	conv_b3_c, bn_b3_c, scale_b3_c = conv_bn(relu_b3_b, num_output, pad=1, kernel_size=3, stride=1)
	
	# add
	eltw = L.Eltwise(conv_b1_a, conv_b2_b, conv_b3_c, eltwise_param=dict(operation=1))
	relu = L.ReLU(eltw, in_place=True, engine=engine)

	return conv_b1_a, bn_b1_a, scale_b1_a, \
	conv_b2_a, bn_b2_a, scale_b2_a, relu_b2_a, conv_b2_b, bn_b2_b, scale_b2_b, \
	conv_b3_a, bn_b3_a, scale_b3_a, relu_b3_a, conv_b3_b, bn_b3_b, scale_b3_b, relu_b3_b, conv_b3_c, bn_b3_c, scale_b3_c, \
	eltw, relu

################################
def uvinet_2d_bn_weighted(dim_data, dim_label, num_class, phase='train'):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))	
	net.d0b_conv, net.d0b_bn, net.d0b_scale, net.d0b_relu = conv_bn_relu(net.data, 64, pad=1, kernel_size=3, stride=1)
	net.d0c_conv, net.d0c_bn, net.d0c_scale, net.d0c_relu = conv_bn_relu(net.d0b_relu, 64, pad=1, kernel_size=3, stride=1)
	############ d1 ############
	# net.d1a_pool = max_pool(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1a_pool, net.d1a_conv, net.d1a_bn, net.d1a_scale, net.d1a_relu, net.d1a_concat = pool_conv_bn_relu_concat(net.d0c_relu, 64)
	net.d1_conv_b1_a, net.d1_bn_b1_a, net.d1_scale_b1_a, \
	net.d1_conv_b2_a, net.d1_bn_b2_a, net.d1_scale_b2_a, net.d1_relu_b2_a, net.d1_conv_b2_b, net.d1_bn_b2_b, net.d1_scale_b2_b, \
	net.d1_eltw, net.d1_relu = incept_3(net.d1a_concat, 128)
	
	############ d2 ############
	# net.d2a_pool = max_pool(net.d1d_relu, pad=0, kernel_size=2, stride=2)
	net.d2a_pool, net.d2a_conv, net.d2a_bn, net.d2a_scale, net.d2a_relu, net.d2a_concat = pool_conv_bn_relu_concat(net.d1_relu, 128)
	net.d2_conv_b1_a, net.d2_bn_b1_a, net.d2_scale_b1_a, \
	net.d2_conv_b2_a, net.d2_bn_b2_a, net.d2_scale_b2_a, net.d2_relu_b2_a, net.d2_conv_b2_b, net.d2_bn_b2_b, net.d2_scale_b2_b, \
	net.d2_conv_b3_a, net.d2_bn_b3_a, net.d2_scale_b3_a, net.d2_relu_b3_a, net.d2_conv_b3_b, net.d2_bn_b3_b, net.d2_scale_b3_b, net.d2_relu_b3_b, net.d2_conv_b3_c, net.d2_bn_b3_c, net.d2_scale_b3_c, \
	net.d2_eltw, net.d2_relu = incept_4(net.d2a_concat, 256)
	
	############ d3 ############
	# net.d3a_pool = max_pool(net.d2d_relu, pad=0, kernel_size=2, stride=2)
	net.d3a_pool, net.d3a_conv, net.d3a_bn, net.d3a_scale, net.d3a_relu, net.d3a_concat = pool_conv_bn_relu_concat(net.d2_relu, 256)
	net.d3_conv_b1_a, net.d3_bn_b1_a, net.d3_scale_b1_a, \
	net.d3_conv_b2_a, net.d3_bn_b2_a, net.d3_scale_b2_a, net.d3_relu_b2_a, net.d3_conv_b2_b, net.d3_bn_b2_b, net.d3_scale_b2_b, \
	net.d3_conv_b3_a, net.d3_bn_b3_a, net.d3_scale_b3_a, net.d3_relu_b3_a, net.d3_conv_b3_b, net.d3_bn_b3_b, net.d3_scale_b3_b, net.d3_relu_b3_b, net.d3_conv_b3_c, net.d3_bn_b3_c, net.d3_scale_b3_c, \
	net.d3_eltw, net.d3_relu = incept_4(net.d3a_concat, 512)
	
	############ d4 ############
	# net.d4a_pool = max_pool(net.d3d_relu, pad=0, kernel_size=2, stride=2)
	net.d4a_pool, net.d4a_conv, net.d4a_bn, net.d4a_scale, net.d4a_relu, net.d4a_concat = pool_conv_bn_relu_concat(net.d3_relu, 512)
	net.d4_conv_b1_a, net.d4_bn_b1_a, net.d4_scale_b1_a, \
	net.d4_conv_b2_a, net.d4_bn_b2_a, net.d4_scale_b2_a, net.d4_relu_b2_a, net.d4_conv_b2_b, net.d4_bn_b2_b, net.d4_scale_b2_b, \
	net.d4_conv_b3_a, net.d4_bn_b3_a, net.d4_scale_b3_a, net.d4_relu_b3_a, net.d4_conv_b3_b, net.d4_bn_b3_b, net.d4_scale_b3_b, net.d4_relu_b3_b, net.d4_conv_b3_c, net.d4_bn_b3_c, net.d4_scale_b3_c, \
	net.d4_eltw, net.d4_relu = incept_4(net.d4a_concat, 1024)
	
	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_bn, net.u3a_scale, net.u3a_relu = deconv_bn_relu(net.d4_relu, 512, pad=0, kernel_size=2, stride=2)
	net.u3b_concat = L.Concat(net.u3a_relu, net.d3_relu, axis=1)
	net.u3_conv_b1_a, net.u3_bn_b1_a, net.u3_scale_b1_a, \
	net.u3_conv_b2_a, net.u3_bn_b2_a, net.u3_scale_b2_a, net.u3_relu_b2_a, net.u3_conv_b2_b, net.u3_bn_b2_b, net.u3_scale_b2_b, \
	net.u3_conv_b3_a, net.u3_bn_b3_a, net.u3_scale_b3_a, net.u3_relu_b3_a, net.u3_conv_b3_b, net.u3_bn_b3_b, net.u3_scale_b3_b, net.u3_relu_b3_b, net.u3_conv_b3_c, net.u3_bn_b3_c, net.u3_scale_b3_c, \
	net.u3_eltw, net.u3_relu = incept_4_decode(net.u3b_concat, 512)

	############ u2 ############
	### a ### Second Deconvolution
	net.u2a_dconv, net.u2a_bn, net.u2a_scale, net.u2a_relu = deconv_bn_relu(net.u3_relu, 256, pad=0, kernel_size=2, stride=2)
	net.u2b_concat = L.Concat(net.u2a_relu, net.d2_relu, axis=1)
	net.u2_conv_b1_a, net.u2_bn_b1_a, net.u2_scale_b1_a, \
	net.u2_conv_b2_a, net.u2_bn_b2_a, net.u2_scale_b2_a, net.u2_relu_b2_a, net.u2_conv_b2_b, net.u2_bn_b2_b, net.u2_scale_b2_b, \
	net.u2_conv_b3_a, net.u2_bn_b3_a, net.u2_scale_b3_a, net.u2_relu_b3_a, net.u2_conv_b3_b, net.u2_bn_b3_b, net.u2_scale_b3_b, net.u2_relu_b3_b, net.u2_conv_b3_c, net.u2_bn_b3_c, net.u2_scale_b3_c, \
	net.u2_eltw, net.u2_relu = incept_4_decode(net.u2b_concat, 256)

	############ u1 ############
	### a ### Third Deconvolution
	net.u1a_dconv, net.u1a_bn, net.u1a_scale, net.u1a_relu = deconv_bn_relu(net.u2_relu, 128, pad=0, kernel_size=2, stride=2)
	net.u1b_concat = L.Concat(net.u1a_relu, net.d1_relu, axis=1)
	net.u1_conv_b1_a, net.u1_bn_b1_a, net.u1_scale_b1_a, \
	net.u1_conv_b2_a, net.u1_bn_b2_a, net.u1_scale_b2_a, net.u1_relu_b2_a, net.u1_conv_b2_b, net.u1_bn_b2_b, net.u1_scale_b2_b, \
	net.u1_eltw, net.u1_relu = incept_3_decode(net.u1b_concat, 128)

	############ u0 ############
	### a ### Fourth Deconvolution
	net.u0a_dconv, net.u0a_bn, net.u0a_scale, net.u0a_relu = deconv_bn_relu(net.u1_relu, 64, pad=0, kernel_size=2, stride=2)
	net.u0b_concat = L.Concat(net.u0a_relu, net.d0c_relu, axis=1)
	net.u0_conv_b1_a, net.u0_bn_b1_a, net.u0_scale_b1_a, \
	net.u0_conv_b2_a, net.u0_bn_b2_a, net.u0_scale_b2_a, net.u0_relu_b2_a, net.u0_conv_b2_b, net.u0_bn_b2_b, net.u0_scale_b2_b, \
	net.u0_eltw, net.u0_relu = incept_3_decode(net.u0b_concat, 64)

	############ score ###########
	net.score4 = L.Convolution(net.d4_relu,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore4 = L.Deconvolution(net.score4,
		param=[dict(lr_mult=10, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=8, kernel_size=32, stride=16,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss4 = L.WeightedSoftmaxWithLoss(net.upscore4, net.label, net.label_weight,
		# net.loss4 = L.SoftmaxWithLoss(net.upscore4, net.label,
			phase=0,
			loss_weight=0.0625,
			loss_param=dict(ignore_label=ignore_label))

	net.score3 = L.Convolution(net.u3_relu,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore3 = L.Deconvolution(net.score3,
		param=[dict(lr_mult=10, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=4, kernel_size=16, stride=8,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss3 = L.WeightedSoftmaxWithLoss(net.upscore3, net.label, net.label_weight,
		# net.loss3 = L.SoftmaxWithLoss(net.upscore3, net.label,
			phase=0,
			loss_weight=0.0625,
			loss_param=dict(ignore_label=ignore_label))

	### loss 2
	net.score2 = L.Convolution(net.u2_relu,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore2 = L.Deconvolution(net.score2,
		param=[dict(lr_mult=10, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=2, kernel_size=8, stride=4,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss2 = L.WeightedSoftmaxWithLoss(net.upscore2, net.label, net.label_weight,
		# net.loss2 = L.SoftmaxWithLoss(net.upscore2, net.label,
			phase=0,
			loss_weight=0.125,
			loss_param=dict(ignore_label=ignore_label))

	### loss 1
	net.score1 = L.Convolution(net.u1_relu,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore1 = L.Deconvolution(net.score1,
		param=[dict(lr_mult=10, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=1, kernel_size=4, stride=2,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss1 = L.WeightedSoftmaxWithLoss(net.upscore1, net.label, net.label_weight,
		# net.loss1 = L.SoftmaxWithLoss(net.upscore1, net.label,
			phase=0,
			loss_weight=0.25,
			loss_param=dict(ignore_label=ignore_label))

	### loss 0
	net.score = L.Convolution(net.u0_relu,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	if phase == "train":
		net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
		# net.loss = L.SoftmaxWithLoss(net.score, net.label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label))
	else:
		net.prob = L.Softmax(net.score, axis=1)
	return net.to_proto()


def make_uvinet(net, dim_data, dim_label, num_class, prototxt_train, prototxt_test):
	# register net
	__nets = ['uvinet_2d_bn_weighted']
	assert net in __nets, 'Unknown net: {}'.format(net)
	global use_global_stats, engine, ignore_label
	engine = 2
	ignore_label = 255

	if net == 'uvinet_2d_bn_weighted':
		use_global_stats = 0
		train_net = uvinet_2d_bn_weighted(dim_data, dim_label, num_class, phase='train')
		use_global_stats = 1
		dim_data[0] = 1
		test_net = uvinet_2d_bn_weighted(dim_data, dim_label, num_class, phase='test')
	with open(prototxt_train, 'w') as f:
		f.write(str(train_net))
	with open(prototxt_test, 'w') as f:
		f.write(str(test_net))
