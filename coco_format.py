import json 
import shutil, os
import cv2

def getImgsInfo(images_folder):
	imgsInfo =[]
	for img in os.listdir(images_folder):
	    path = os.path.join(images_folder, img)
	    image = cv2.imread(path)
	    h,w,_ = image.shape
	    #imgName = img[:-4] #img.split('_')[1]
	    imgName = img.split('_')[-1][:-4]
	    imgInfo =   {
            "id": int(imgName),
            "license": 1,
            "width": w,
            "height": h,
            "file_name": img
        }
	    imgsInfo.append(imgInfo)

	return imgsInfo

def getGroundTruthInfo(txts_folder):
	annotationsInfo =[]
	annotation_id = 500
	for txt_file in os.listdir(txts_folder):
	    txt_path = os.path.join(txts_folder, txt_file)
	    #imgName = txt_file[3:-4]
	    imgName = txt_file.split('_')[-1][:-4]
	    with open(txt_path) as f:
	            lines = f.readlines() 
	            for i,line in enumerate(lines):
	                line = line.rstrip()
	                temp = line.split(',')[0:8]
	                if temp ==['']:
	                	break
	                points_coordinates = [(temp[0],temp[1]),(temp[2],temp[3]),(temp[4],temp[5]),(temp[6],temp[7])]
	                x_min = min(int(point[0]) for point in points_coordinates)
	                x_max = max(int(point[0]) for point in points_coordinates)
	                y_min = min(int(point[1]) for point in points_coordinates)
	                y_max = max(int(point[1]) for point in points_coordinates)
	                w =  x_max -x_min
	                h =  y_max -y_min
	                annotation ={
				        "image_id": int(imgName),
				        "bbox": [x_min,y_min,w,h],
				        'area': w*h,
				        "iscrowd":0,
				        "category_id": 1,
				        "id": annotation_id}
	                annotation_id +=1
	                annotationsInfo.append(annotation)

	return annotationsInfo



dataset_coco_format = {}

dataset_coco_format['info'] = {
        "description": "Re-ID Dataset",
        "version": "1.0",
        "year": 217 }

dataset_coco_format['licenses'] = [
        {
            "url": "http://creativecommons.org/licenses/by/2.0/",
            "id": 1,
            "name": "Attribution License"
        }
    ]

source_folder ='/content/Deformable-DETR/ReID/ReID_train/train_image/'
dataset_coco_format['images'] =getImgsInfo(source_folder)

txts_folder ='/content/Deformable-DETR/ReID/ReID_train/train_gt/'
dataset_coco_format['annotations'] = getGroundTruthInfo(txts_folder)

dataset_coco_format['categories'] = [
        {
            "supercategory": "text",
            "id": 1,
            "name": "text"
        }]

with open("instances_train.json", "w") as write_file:
    json.dump(dataset_coco_format, write_file,indent=2)
