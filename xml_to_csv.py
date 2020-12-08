import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    classes_names = []

    xml_list = []

    xml_file = os.listdir(path)

    for i in xml_file:
        tree = ET.parse(path + '\\' + i)
        root = tree.getroot()

        for j in root.findall('object'):
            classes_names.append(j[0].text)
            bbox = j.find('bndbox')


            value = (root.find('filename').text + '.jpg',
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     j[0].text,
                     int(bbox.find('xmin').text),
                     int(bbox.find('ymin').text),
                     int(bbox.find('xmax').text),
                     int(bbox.find('ymax').text),
                     )

            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return xml_df, classes_names

def main(train_xml_path, test_xml_path, csv_path):

    for label_path in [train_xml_path, test_xml_path]:

        xml_df, classes = xml_to_csv(label_path)

        if label_path == train_xml_path:

            xml_df.to_csv(csv_path +'\\train_labels.csv', index=None)
        else:
            xml_df.to_csv(csv_path + '\\test_labels.csv',index=None)

    pbtxt_content = ""

    label_map_path = os.path.join("label_map.pbtxt")

    # creats a pbtxt file the has the class names.
    for i, class_name in enumerate(classes):
        # display_name is optional.
        pbtxt_content = (
                pbtxt_content
                + "item {{\n    id: {0}\n    name: '{1}'\n    display_name: '{2}'\n }}\n\n".format(i + 1, class_name,
                                                                                                   class_name)
        )
    pbtxt_content = pbtxt_content.strip()
    with open(label_map_path, "w") as f:
        f.write(pbtxt_content)

if __name__ == '__main__':

    main(r'C:\Users\young\FileDetection\dog_breed_test\dog_detection\archive\train_data', r'C:\Users\young\FileDetection\dog_breed_test\dog_detection\archive\test_data', r'C:\Users\young\FileDetection\dog_breed_test\dog_detection\archive\csv_file')




