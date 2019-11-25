from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from keras.applications.mobilenet import preprocess_input

from tensorflow import ConfigProto
from tensorflow import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata related to the training (generated when training).",
				default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
parser.add_option("--write", dest="write", help="to write out the image with detections or not.", action='store_true')
parser.add_option("--load", dest="load", help="specify model path.", default=None)
(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
	parser.error('Error: path to test data must be specified. Pass --path to command line')


config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

# we will use resnet. may change to vgg
if options.network == 'vgg':
	C.network = 'vgg16'
	from keras_frcnn import vgg as nn
elif options.network == 'resnet50':
	from keras_frcnn import resnet as nn
	C.network = 'resnet50'
elif options.network == 'vgg19':
	from keras_frcnn import vgg19 as nn
	C.network = 'vgg19'
elif options.network == 'mobilenetv1':
	from keras_frcnn import mobilenetv1 as nn
	C.network = 'mobilenetv1'
#	from keras.applications.mobilenet import preprocess_input
elif options.network == 'mobilenetv1_05':
	from keras_frcnn import mobilenetv1_05 as nn
	C.network = 'mobilenetv1_05'
#	from keras.applications.mobilenet import preprocess_input
elif options.network == 'mobilenetv1_25':
	from keras_frcnn import mobilenetv1_25 as nn
	C.network = 'mobilenetv1_25'
#	from keras.applications.mobilenet import preprocess_input
elif options.network == 'mobilenetv2':
	from keras_frcnn import mobilenetv2 as nn
	C.network = 'mobilenetv2'
else:
	print('Not a valid model')
	raise ValueError

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

# Converts numerical classification to defect label
def convert_to_defect(number):
	switcher = {
		1: "open",
		2: "short",
		3: "mousebite",
		4: "spur",
		5: "copper",
		6: "pin-hole",
	}
	return switcher.get(number,"nothing")

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle

	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

# matches predicted vs actual bounding boxes
def calculate_IOUS(actual,predicted,TP_FN_FP):
	#TP_FN_FP[1] += len(actual) - len(predicted)
	IOUS = []
	actual_bb = []

	for i, bb_predicted in enumerate(predicted):
		highest_IOU = 0
		highest_IOU_actual_idx = 0

		for j, bb_actual in enumerate(actual):
			IOU = bb_intersection_over_union(bb_predicted,bb_actual)
			if  IOU > highest_IOU:
				highest_IOU = IOU
				highest_IOU_actual_idx = j

		actual_bb.append(actual[highest_IOU_actual_idx])
		actual.pop(highest_IOU_actual_idx)
		IOUS.append([highest_IOU,bb_predicted])
	
	for i,IOU in enumerate(IOUS):
		defect_num = IOU[-1][-1]
		if IOU[1] >= 0.5:
			if defect_num == actual_bb[i][-1]:
				TP_FN_FP[defect_num-1][0] += 1
			else: 
				TP_FN_FP[defect_num-1][1] += 1
		else:
			if defect_num == actual_bb[i][-1]:
				TP_FN_FP[defect_num-1][2] += 1
	
	for bb in actual:
		TP_FN_FP[bb[-1]-1][1] += 1

def return_results(TP_FN_FP):
	precision = [0,0,0,0,0,0]
	recall = [0,0,0,0,0,0]
	f1 = [0,0,0,0,0,0]
	for i,defect_class in enumerate(TP_FN_FP):
		TP = defect_class[0]
		FN = defect_class[1]
		FP = defect_class[2]
		if TP == 0:
			precision[i] = 0
			recall[i] = 0
			f1[i] = 0
		else:
			precision[i] = TP / (TP + FP)
			recall[i] = TP / (TP + FN)
			f1[i] = precision[i] * recall[i] / (precision[i] + recall[i])
	return precision, recall, f1
	
	
	

# returns coordinates of bounding boxes ordered by classification number
def reorder_by_classification(coords):
	# reorder boxes by classification number
	classes = []
	ordered = []
	for coord in coords:
		classes.append(coord[-1])
	# get indices of sorted classes
	idxs = np.argsort(classes)
	for i in idxs:
		ordered.append(coords[i])	
	return ordered


class_mapping = C.class_mapping

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

if C.network == 'resnet50':
	num_features = 1024
elif C.network =="mobilenetv2":
	num_features = 320
else:
	# may need to fix this up with your backbone..!
	print("backbone is not resnet50. number of features chosen is 512")
	num_features = 512

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (num_features, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, num_features)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping))

model_rpn = Model(img_input, rpn_layers)
model_classifier = Model([feature_map_input, roi_input], classifier)
# model loading
if options.load == None:
  print('Loading weights from {}'.format(C.model_path))
  model_rpn.load_weights(C.model_path, by_name=True)
  model_classifier.load_weights(C.model_path, by_name=True)
else:
  print('Loading weights from {}'.format(options.load))
  model_rpn.load_weights(options.load, by_name=True)
  model_classifier.load_weights(options.load, by_name=True)

#model_rpn.compile(optimizer='adam', loss='mse')
#model_classifier.compile(optimizer='adam', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.5

visualise = True

num_rois = C.num_rois

TP_FN_FP = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

for idx, img_name in enumerate(sorted(os.listdir(img_path))):
	actual = []
	predicted = []
	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
		continue
	
	print(img_name)
	st = time.time()
	filepath = os.path.join(img_path,img_name)

	img = cv2.imread(filepath)
	
    # preprocess image
	X, ratio = format_img(img, C)
	img_scaled = (np.transpose(X[0,:,:,:],(1,2,0)) + 127.5).astype('uint8')
	if K.image_dim_ordering() == 'tf':
		X = np.transpose(X, (0, 2, 3, 1))
	# get the feature maps and output from the RPN
	[Y1, Y2, F] = model_rpn.predict(X)
	

	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.3)
	print(R.shape)
    
	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}
	for jk in range(R.shape[0]//num_rois + 1):
		ROIs = np.expand_dims(R[num_rois*jk:num_rois*(jk+1),:],axis=0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0]//num_rois:
			#pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:,:curr_shape[1],:] = ROIs
			ROIs_padded[0,curr_shape[1]:,:] = ROIs[0,0,:]
			ROIs = ROIs_padded

		[P_cls,P_regr] = model_classifier.predict([F, ROIs])
		print(P_cls)

		for ii in range(P_cls.shape[1]):

			if np.max(P_cls[0,ii,:]) < 0.8 or np.argmax(P_cls[0,ii,:]) == (P_cls.shape[2] - 1):
				continue

			cls_name = class_mapping[np.argmax(P_cls[0,ii,:])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []
			(x,y,w,h) = ROIs[0,ii,:]

			bboxes[cls_name].append([16*x,16*y,16*(x+w),16*(y+h)])
			probs[cls_name].append(np.max(P_cls[0,ii,:]))

	all_dets = []
	for key in bboxes:
		print(key)
		print(len(bboxes[key]))
		bbox = np.array(bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh = 0.3)
		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]
			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

			cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)
			
			textLabel = '{}: Probability {}%'.format(convert_to_defect(int(key)),int(100*new_probs[jk]))
			all_dets.append((key,100*new_probs[jk]))

			predicted.append([real_x1,real_y1,real_x2,real_y2,int(key)])

			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
			textOrg = (real_x2, real_y1-0)

			#cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
			#cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1)
	
	# plot actual defect locations
	# this is for path to PCBData
	# actual_defects_path = img_path.replace('/'+img_name,'_not/'+img_name.replace('_test.jpg','.txt'))

	actual_defects_path = img_path + img_name.replace('_test.jpg','.txt')
	with open(actual_defects_path,'r') as f:
		for line in f:
			coords = map(int,line.split())
			actual.append(coords)
			cv2.rectangle(img,(coords[0],coords[1]),(coords[2],coords[3]), (0,0,255),1)
			cv2.putText(img,convert_to_defect(coords[-1]),(coords[2],coords[3]),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,0,255),1)

	print('Elapsed time = {}'.format(time.time() - st))
	print(all_dets)
	print(bboxes)
    # enable if you want to show pics
	if options.write:
           import os
           if not os.path.isdir("results"):
              os.mkdir("results")
           cv2.imwrite('./results/{}.png'.format(idx),img)

	calculate_IOUS(actual,predicted,TP_FN_FP)



print("\nTP_FN_FP: \n")
print(TP_FN_FP)

prec,rec,f1 = return_results(TP_FN_FP)
for i in range(0,6):
	print("\n" + convert_to_defect(i+1))
	print("\nPrecision: " + str(prec[i]))
	print("\nRecall: " + str(rec[i]))
	print("\nF1: " + str(f1[i]))

print("\n\nmAP: " + str(np.mean(prec)))
print("\nMean Recall: " + str(np.mean(rec)))
print("\nMean F1: " + str(np.mean(f1)))
