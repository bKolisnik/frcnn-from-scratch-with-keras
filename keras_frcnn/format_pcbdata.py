import os

# Iterates through image and bounding box classification data from DeepPCB dataset
# and generates text file in the format for simple_parser (filepath,x1,y1,x2,y2,class_name)
# e.g.
# /data/imgs/img_001.jpg,837,346,981,456,cow /data/imgs/img_002.jpg,215,312,279,391,cat


# folder organization
# DeepPCB ->
#		groupfolders ->
#			image folders ->
#				'temp' images (have no defects)
#				'test' images (have defects)
#			image feature folders ->
#				text files containing defect bounding boxes and their classification
#
# Notes on provided 'test.txt' an 'trainval.txt'
# 	test.txt format:
# 		group20085/20085/20085291.jpg group20085/20085_not/20085291.txt
# 	trainval.txt format:
# 		group20085/20085/20085000.jpg group20085/20085_not/20085000.txt
#
#	these jpgs do not correspond to any real images as the image files have 
# 	an underscore followed by either 'temp' or 'test'


# initialize output text file
# open trainval.txt file
# 	for each line in text file
#		split by space, get last path (path to defect text file)
#			read_defect_files()

def write_file(input_path,output_file):
	trainval_txt_path = os.path.join(input_path,'trainval.txt')
	with open(trainval_txt_path,'r') as f:
		for line in f:
			defect_txt_path = line.split()[-1]
			new_path = f'{input_path}/{defect_txt_path}'
			write_defects(new_path,output_file)

# 	open defect text file
# 		for each line in the text file 
#			write line in output text file in the following format:q
#				(text_file_name.replace('.txt','').(append('_test.jpg'), x1,x2,y1,y2, defect)

def write_defects(input_path,output_file):
	with open(input_path,'r') as f:
		file_name = os.path.basename(f.name)
		for line in f:
			jpg_name = file_name.replace('.txt','_test.jpg')
			formatted_line = line.rstrip().replace(' ',',')
			new_line = f'{jpg_name},{formatted_line}\n'
			output_file.write(new_line)

if __name__ == '__main__':
	output_file = open("sp_input_file.txt","w")
	path = os.path.join(os.getcwd(),'..','PCBData')
	write_file(path,output_file)