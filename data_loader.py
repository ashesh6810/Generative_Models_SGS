import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
import netCDF4 as nc
from saveNCfile import savenc


def load_test_data(FF,GG,lead):

  u=FF['U']
  u=u[2500:,:,:,:]
  lat=np.asarray(FF['lat'])
  lon=np.asarray(FF['lon'])

  Nlat=np.size(u,2);
  Nlon=np.size(u,3);

  u_test_input = u[0:np.size(u,0)-lead,:,74:216,:]
  u_test_label = u[0+lead:np.size(u,0),:,74:216,:]



  u_test_input_Tr=np.zeros([np.size(u,0),2,142,Nlon])
  u_test_label_Tr=np.zeros([np.size(u,0),2,142,Nlon])



  for k in range(0,np.size(u_test_input,0)):
    u_test_input_Tr[k,0,:,:] = u_test_input[k,0,:,:]
    u_test_input_Tr[k,1,:,:] = u_test_input[k,1,:,:]
    u_test_label_Tr[k,0,:,:] = u_test_label[k,0,:,:]
    u_test_label_Tr[k,1,:,:] = u_test_label[k,1,:,:]

## convert to torch tensor
  u_test_input_Tr_torch = torch.from_numpy(u_test_input_Tr).float()
  u_test_label_Tr_torch = torch.from_numpy(u_test_label_Tr).float()


  v=GG['V']
  v=v[2500:,:,:,:]
  lat=np.asarray(FF['lat'])
  lon=np.asarray(FF['lon'])

  Nlat=np.size(v,2);
  Nlon=np.size(v,3);

  v_test_input = v[0:np.size(v,0)-lead,:,74:216,:]
  v_test_label = v[0+lead:np.size(v,0),:,74:216,:]



  v_test_input_Tr=np.zeros([np.size(v,0),2,142,Nlon])
  v_test_label_Tr=np.zeros([np.size(v,0),2,142,Nlon])



  for k in range(0,np.size(u_test_input,0)):
    v_test_input_Tr[k,0,:,:] = v_test_input[k,0,:,:]
    v_test_input_Tr[k,1,:,:] = v_test_input[k,1,:,:]
    v_test_label_Tr[k,0,:,:] = v_test_label[k,0,:,:]
    v_test_label_Tr[k,1,:,:] = v_test_label[k,1,:,:]

## convert to torch tensor
  v_test_input_Tr_torch = torch.from_numpy(v_test_input_Tr).float()
  v_test_label_Tr_torch = torch.from_numpy(v_test_label_Tr).float()



  uv_test_input_Tr_torch = torch.cat((u_test_input_Tr_torch,v_test_input_Tr_torch),1)

  uv_test_label_Tr_torch = torch.cat((u_test_label_Tr_torch,v_test_label_Tr_torch),1)

  return uv_test_input_Tr_torch, uv_test_label_Tr_torch


def load_train_data(loop1,loop2, lead,trainN):
  
     File1=nc.Dataset(loop1)
     File2=nc.Dataset(loop2)
     


     u=File1['U']
     u=u[2500:,:,:,:]
     Nlat=np.size(u,2);
     Nlon=np.size(u,3);
     


     u_input = u[0:trainN,:,74:216,:]
     u_label = u[0+lead:trainN+lead,:,74:216,:]

     u_input_Tr=np.zeros([trainN,2,142,Nlon])
     u_label_Tr=np.zeros([trainN,2,142,Nlon])


     for k in range(0,trainN):
      u_input_Tr[k,0,:,:] = u_input[k,0,:,:]
      u_input_Tr[k,1,:,:] = u_input[k,1,:,:]
      u_label_Tr[k,0,:,:] = u_label[k,0,:,:]
      u_label_Tr[k,1,:,:] = u_label[k,1,:,:]

     print('Train input', np.shape(u_input_Tr))
     print('Train label', np.shape(u_label_Tr))
     u_input_Tr_torch = torch.from_numpy(u_input_Tr).float()
     u_label_Tr_torch = torch.from_numpy(u_label_Tr).float()  


     

     v=File2['V']
     v=v[2500:,:,:,:]
     Nlat=np.size(v,2);
     Nlon=np.size(v,3);



     v_input = v[0:trainN,:,74:216,:]
     v_label = v[0+lead:trainN+lead,:,74:216,:]

     v_input_Tr=np.zeros([trainN,2,142,Nlon])
     v_label_Tr=np.zeros([trainN,2,142,Nlon])


     for k in range(0,trainN):
      v_input_Tr[k,0,:,:] = v_input[k,0,:,:]
      v_input_Tr[k,1,:,:] = v_input[k,1,:,:]
      v_label_Tr[k,0,:,:] = v_label[k,0,:,:]
      v_label_Tr[k,1,:,:] = v_label[k,1,:,:]

     print('Train input', np.shape(v_input_Tr))
     print('Train label', np.shape(v_label_Tr))
     v_input_Tr_torch = torch.from_numpy(v_input_Tr).float()
     v_label_Tr_torch = torch.from_numpy(v_label_Tr).float() 

     uv_input_Tr_torch = torch.cat((u_input_Tr_torch,v_input_Tr_torch),1)
     uv_label_Tr_torch = torch.cat((u_label_Tr_torch,v_label_Tr_torch),1)
    
     return uv_input_Tr_torch, uv_label_Tr_torch
