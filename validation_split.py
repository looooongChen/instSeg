import csv
import random
import os

csv_list = 'D:/instSeg/ds_cyst/export-mask-2020-Dec-07-18-47-41/files.csv'
split = 5
save_list = 'D:/instSeg/ds_cyst/cyst.csv'
rel_path = True
rel_path_base = 'D:/instSeg/'

with open(csv_list) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    examples = [row for row in csv_reader]
random.shuffle(examples)
partition_num = len(examples)/split
with open(save_list, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i, e in enumerate(examples):
        if rel_path:
            row = [round(i/partition_num), os.path.relpath(e[0], rel_path_base), os.path.relpath(e[1], rel_path_base)]
        else:
            row = [round(i/partition_num), e[0], e[1]]
        csv_writer.writerow(row)



    