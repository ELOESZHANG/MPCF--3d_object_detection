import torch
import os
import re

filepath = "/home/gaopan2023/Documents/MPCF1/" + "output/kitti_models/mpcf/default/ckpt/消融实验-ColorEh+NSC+0Trans+0ROI--90.25--59/log_train_20231208-170543.txt"


in_files = open(filepath,'r')
# out_files = open('D://*//*.txt','w')

row = []
row0 = []
row1 = []
row2 = []
row3 = []

flag = 1

something = "Performance of EPOCH"
something0 = "Car AP@0.70, 0.70"
something1 = "Car AP_R40@0.70, 0.70"
something2 = "Pedestrian AP_R40@0.50, 0.50"
something3 = "Cyclist AP_R40@0.50, 0.50"
iii=39
ii=iii
for i, line in enumerate(in_files.readlines(), start =1):
    # if something in line:
    #     row.append(i)

    if something0 in line:
        a = i + 3

        if flag:
            row0.append(i)

        row0.append(a)


    if something1 in line:
        a = i + 3

        if flag:
            row1.append(i)
        row1.append(a)
        flag = 0


    if something2 in line:
        a = i+3

        if flag:
            row2.append(i)

        row2.append(a)


    if something3 in line:
        a = i + 3

        if flag:
            row3.append(i)

        row3.append(a)

    if i in row0:
        ii=ii+1
        numbers = re.findall(r"\d+\.\d+", line)
        average = sum(map(float, numbers)) / len(numbers)
        print(line[:-1], '  ', f"{average:.2f}",'       ',ii)


        # print(' ', f"{average:.2f}")
print()

in_files.close()
in_files = open(filepath,'r')
ii=iii
for i1, line1 in enumerate(in_files.readlines(), start=1):

    if i1 in row1:
        ii = ii + 1
        numbers = re.findall(r"\d+\.\d+", line1)
        average = sum(map(float, numbers)) / len(numbers)
        print(line1[:-1], '  ', f"{average:.2f}", '       ', ii)
print()

in_files.close()
in_files = open(filepath, 'r')
ii=iii
for i1, line1 in enumerate(in_files.readlines(), start=1):

    if i1 in row2:
        ii = ii + 1
        numbers = re.findall(r"\d+\.\d+", line1)
        average = sum(map(float, numbers)) / len(numbers)
        print(line1[:-1], '  ', f"{average:.2f}", '       ', ii)

print()

in_files.close()
in_files = open(filepath, 'r')
ii=iii
for i1, line1 in enumerate(in_files.readlines(), start=1):

    if i1 in row3:
        ii = ii + 1
        numbers = re.findall(r"\d+\.\d+", line1)
        average = sum(map(float, numbers)) / len(numbers)
        print(line1[:-1], '  ', f"{average:.2f}", '       ', ii)


            # out_files.write(line)
# out_files.close()

