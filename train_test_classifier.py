import os
import shutil
from sklearn.model_selection import train_test_split



def data_split(base_path, data_store_path):
    if not os.path.isdir(data_store_path + '\\train_data'):
        os.makedirs(data_store_path + '\\train_data', 0o777)

    if not os.path.isdir(data_store_path + '\\test_data'):
        os.makedirs(data_store_path + '\\test_data', 0o777)


    base_path_list = os.listdir(base_path)

    for i in base_path_list:

        data_list = os.listdir(base_path + '\\' + i)




        x_train, x_test = train_test_split(data_list, test_size=0.2, train_size=0.8, random_state=0)

        #print(len(x_train), len(x_test))



        for j in x_train:
            shutil.copy(base_path + '\\' + i + '\\' + j, data_store_path + '\\train_data')

        for k in x_test:
            shutil.copy(base_path + '\\' + i + '\\' + k, data_store_path + '\\test_data')


data_split(r"C:\Users\young\FileDetection\dog_breed_test\dog_detection\archive\fine_tuning_practice\Annotation", r'C:\Users\young\FileDetection\dog_breed_test\dog_detection\archive')