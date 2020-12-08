import cv2
import os
import csv

#n02098105_2063.jpg
#n02106030_3948.jpg
#해당 위 두 이미지파일들이 size가 안맞기 때문에 해당 두 이미지 삭제


images_path = r'C:\Users\young\FileDetection\dog_breed_test\dog_detection\archive\dog_images'

for CSV_FILE in ['train_labels.csv', 'test_labels.csv']:
  with open(CSV_FILE, 'r') as fid:
      print('[*] Checking file:', CSV_FILE)
      file = csv.reader(fid, delimiter=',')
      first = True
      cnt = 0
      error_cnt = 0
      error = False
      for row in file:
          if error == True:
              error_cnt += 1
              error = False
          if first == True:
              first = False
              continue
          cnt += 1
          name, width, height, xmin, ymin, xmax, ymax = row[0], int(row[1]), int(row[2]), int(row[4]), int(row[5]), int(row[6]), int(row[7])
          path = os.path.join(images_path, name)
          #print(path)
          img = cv2.imread(path)
          if type(img) == type(None):
              error = True
              print('Could not read image', img)
              continue
          org_height, org_width = img.shape[:2]
          if org_width != width:
              error = True
              print('Width mismatch for image: ', name, width, '!=', org_width)
          if org_height != height:
              error = True
              print('Height mismatch for image: ', name, height, '!=', org_height)
          if xmin > org_width:
              error = True
              print('XMIN > org_width for file', name)
          if xmax > org_width:
              error = True
              print('XMAX > org_width for file', name)
          if ymin > org_height:
              error = True
              print('YMIN > org_height for file', name)
          if ymax > org_height:
              error = True
              print('YMAX > org_height for file', name)
          if error == True:
              print('Error for file: %s' % name)
              print()
      print()
      print('Checked %d files and realized %d errors' % (cnt, error_cnt))
      print("-----")