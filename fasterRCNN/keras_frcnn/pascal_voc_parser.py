import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
def get_data(data_path):
	all_imgs = []

	classes_count = {}

	class_mapping = {}

	visualise = False

	print('Parsing annotation files')

	fileList = os.listdir(data_path)
	annots = []
	for i in range(len(fileList)):
		if fileList[i].endswith(".xml"):
			annots.append(os.path.join(data_path, fileList[i]))

	idx = 0
	for annot in annots:
		try:
			idx += 1

			et = ET.parse(annot)
			element = et.getroot()

			element_objs = element.findall('object')
			element_filename = element.find('filename').text
			element_width = int(element.find('size').find('width').text)
			element_height = int(element.find('size').find('height').text)

			if len(element_objs) > 0:
				annotation_data = {'filepath': os.path.join(data_path, element_filename), 'width': element_width,
								   'height': element_height, 'bboxes': [], 'imageset': 'train'}

			for element_obj in element_objs:
				class_name = element_obj.find('name').text
				if class_name not in classes_count:
					classes_count[class_name] = 1
				else:
					classes_count[class_name] += 1

				if class_name not in class_mapping:
					class_mapping[class_name] = len(class_mapping)

				obj_bbox = element_obj.find('bndbox')
				x1 = int(round(float(obj_bbox.find('xmin').text)))
				y1 = int(round(float(obj_bbox.find('ymin').text)))
				x2 = int(round(float(obj_bbox.find('xmax').text)))
				y2 = int(round(float(obj_bbox.find('ymax').text)))
				difficulty = int(element_obj.find('difficult').text) == 1
				annotation_data['bboxes'].append(
					{'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
			all_imgs.append(annotation_data)

			if visualise:
				img = cv2.imread(annotation_data['filepath'])
				for bbox in annotation_data['bboxes']:
					cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
								  'x2'], bbox['y2']), (0, 0, 255))
				cv2.imshow('img', img)
				cv2.waitKey(0)

		except Exception as e:
			print(e)
			continue
	return all_imgs, classes_count, class_mapping
