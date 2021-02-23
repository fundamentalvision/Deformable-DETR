
import random
import shutil,os

random.seed(42)
#Generate 706 random numbers between -- and --
randomlist = random.sample(range(1, 8706), 706)


count =1

img_dir = '/content/Deformable-DETR/ReID/ReID_train/train_image/'
txt_dir = '/content/Deformable-DETR/ReID/ReID_train/train_gt/'
dest_folder ='/content/Deformable-DETR/ReID/ReID_train/val/'
dest_txt_folder ='/content/Deformable-DETR/ReID/ReID_train/val_gt/'
for imgfile in os.listdir(img_dir):
	if imgfile.endswith('.jpg'):
		if count in randomlist:
			img_path =os.path.join(img_dir, imgfile)
			shutil.move(img_path, dest_folder)
		
			file_path =  os.path.join(txt_dir, imgfile[:-3]+'txt')
			shutil.move(file_path, dest_txt_folder)

		count +=1
		
