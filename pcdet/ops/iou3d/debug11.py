import torch

#'Car' [[3.9, 1.6, 1.56]],
#'Pedestrian' [ [ 0.8, 0.6, 1.73 ] ],
#'Cyclist', [ [ 1.76, 0.6, 1.73 ] ],

w_box1=torch.tensor(3.9)
h_box1=torch.tensor(1.56)
w_box2=torch.tensor(3.5)
h_box2=torch.tensor(1.56)
iou3d =0.94
v = (4 / 3.14 ** 2) * abs(torch.atan(h_box1/w_box1 ) - torch.atan(h_box2 / w_box2))
alpha = v / (v - iou3d + (1 + 1e-7))
theta11=torch.tensor([0,0.01,0.1,0.2,3])
ff=(1+torch.exp(theta11*10))
ff1=1/torch.exp(1.6+1/torch.atan(theta11/2+0.1))

# print(torch.atan(h_box1/w_box1))
# print(torch.atan(h_box2 / w_box2))
# print(alpha*v)
# print(v)
print('ff:',ff)
print('ff1:',ff1)
a = torch.rand(300, 90)
b =a[1:]
print('b:',b.shape)


