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
from saveNCfile_for_activations import savenc_for_activations
from data_loader import load_test_data
from data_loader import load_train_data
from prettytable import PrettyTable
from count_trainable_params import count_parameters
import hdf5storage


### PATHS and FLAGS ###

path_outputs = '/glade/scratch/asheshc/theory-interp/QG/larger_domain_QG/RK4_UNET_FFT_loss_outputs_velocity_trained/'








##### prepare test data ###################################################

FF=nc.Dataset('/glade/scratch/asheshc/theory-interp/QG/larger_domain_QG/Dry'+str(9)+'/U_output.nc')
GG=nc.Dataset('/glade/scratch/asheshc/theory-interp/QG/larger_domain_QG/Dry'+str(9)+'/V_output.nc')
lat=np.asarray(FF['lat'])
lon=np.asarray(FF['lon'])

lead = 1

psi_test_input_Tr_torch, psi_test_label_Tr_torch  = load_test_data(FF,GG,lead)

psi_test_label_Tr_unnormalized = psi_test_label_Tr_torch.detach().cpu().numpy()


dim_lon = int(142)
new_lat = lat[74:216]

M_test_level1 = torch.mean(torch.flatten(psi_test_input_Tr_torch[:,0,:,:]))
STD_test_level1 = torch.std(torch.flatten(psi_test_input_Tr_torch[:,0,:,:]))

M_test_level2 = torch.mean(torch.flatten(psi_test_input_Tr_torch[:,1,:,:]))
STD_test_level2 = torch.std(torch.flatten(psi_test_input_Tr_torch[:,1,:,:]))


M_test_level3 = torch.mean(torch.flatten(psi_test_input_Tr_torch[:,2,:,:]))
STD_test_level3 = torch.std(torch.flatten(psi_test_input_Tr_torch[:,2,:,:]))

M_test_level4 = torch.mean(torch.flatten(psi_test_input_Tr_torch[:,3,:,:]))
STD_test_level4 = torch.std(torch.flatten(psi_test_input_Tr_torch[:,3,:,:]))



psi_test_input_Tr_torch_norm_level1 = ((psi_test_input_Tr_torch[:,0,None,:,:]-M_test_level1)/STD_test_level1)
psi_test_input_Tr_torch_norm_level2 = ((psi_test_input_Tr_torch[:,1,None,:,:]-M_test_level2)/STD_test_level2)
psi_test_input_Tr_torch_norm_level3 = ((psi_test_input_Tr_torch[:,2,None,:,:]-M_test_level3)/STD_test_level3)
psi_test_input_Tr_torch_norm_level4 = ((psi_test_input_Tr_torch[:,3,None,:,:]-M_test_level4)/STD_test_level4)



psi_test_input_Tr_torch  = torch.cat((psi_test_input_Tr_torch_norm_level1,psi_test_input_Tr_torch_norm_level2,psi_test_input_Tr_torch_norm_level3,psi_test_input_Tr_torch_norm_level4),1)



psi_test_label_Tr_torch_norm_level1 = (psi_test_label_Tr_torch[:,0,None,:,:]-M_test_level1)/STD_test_level1
psi_test_label_Tr_torch_norm_level2 = (psi_test_label_Tr_torch[:,1,None,:,:]-M_test_level2)/STD_test_level2
psi_test_label_Tr_torch_norm_level3 = (psi_test_label_Tr_torch[:,2,None,:,:]-M_test_level3)/STD_test_level3
psi_test_label_Tr_torch_norm_level4 = (psi_test_label_Tr_torch[:,3,None,:,:]-M_test_level4)/STD_test_level4



psi_test_label_Tr_torch = torch.cat((psi_test_label_Tr_torch_norm_level1,psi_test_label_Tr_torch_norm_level2,psi_test_label_Tr_torch_norm_level3,psi_test_label_Tr_torch_norm_level4),1)

print('shape of normalized input test',psi_test_input_Tr_torch.shape)
print('shape of normalized label test',psi_test_label_Tr_torch.shape)













###############################################################################



################### Load training data files ########################################
fileList_train_U=[]
fileList_train_V=[]
mylist = [1,2,3,4,5,6,7,8,10]
for k in mylist:
  fileList_train_U.append ('/glade/scratch/asheshc/theory-interp/QG/larger_domain_QG/Dry'+str(k)+'/U_output.nc')
  fileList_train_V.append ('/glade/scratch/asheshc/theory-interp/QG/larger_domain_QG/Dry'+str(k)+'/V_output.nc')
##########################################################################################


def regular_loss (output, target):

 loss = torch.mean((output-target)**2)
 return loss    



def spectral_loss(output, target,wavenum_init,wavenum_init_ydir,lamda_reg):

 loss1 = torch.mean((output-target)**2)

 out_fft = torch.mean(torch.abs(torch.fft.rfft(output,dim=3)),dim=2)
 target_fft = torch.mean(torch.abs(torch.fft.rfft(target,dim=3)),dim=2)

 out_fft_ydir = torch.mean(torch.abs(torch.fft.rfft(output,dim=2)),dim=3)
 target_fft_ydir = torch.mean(torch.abs(torch.fft.rfft(target,dim=2)),dim=3)


 loss2 = torch.mean(torch.abs(out_fft[:,0,wavenum_init:]-target_fft[:,0,wavenum_init:]))
 loss3 = torch.mean(torch.abs(out_fft[:,1,wavenum_init:]-target_fft[:,1,wavenum_init:]))

 loss2_ydir = torch.mean(torch.abs(out_fft_ydir[:,0,wavenum_init_ydir:]-target_fft_ydir[:,0,wavenum_init_ydir:]))
 loss3_ydir = torch.mean(torch.abs(out_fft_ydir[:,1,wavenum_init_ydir:]-target_fft_ydir[:,1,wavenum_init_ydir:]))


 loss = (1-lamda_reg)*loss1 + 0.25*lamda_reg*loss2 +0.25*lamda_reg*loss3 + 0.25*lamda_reg*loss2_ydir + 0.25*lamda_reg*loss3_ydir

 return loss


def RK4step(net,input_batch):
 output_1,_,_,_,_,_,_ = net(input_batch.cuda())
 output_2,_,_,_,_,_,_ = net(input_batch.cuda()+0.5*output_1)
 output_3,_,_,_,_,_,_ = net(input_batch.cuda()+0.5*output_2)
 output_4,_,_,_,_,_,_ = net(input_batch.cuda()+output_3)

 return input_batch.cuda() + (output_1+2*output_2+2*output_3+output_4)/6


def Eulerstep(net,input_batch):
 output_1,_,_,_,_,_,_ = net(input_batch.cuda())
 return input_batch.cuda() + (output_1) 
  

def directstep(net,input_batch):
  output_1,_,_,_,_,_,_ = net(input_batch.cuda())
  return output_1



class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = (nn.Conv2d(4, 64, kernel_size=5, stride=1, padding='same'))
        self.hidden1 = (nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same' ))
        self.hidden2 = (nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same' ))
        self.hidden3 = (nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same' ))
        self.hidden4 = (nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same' ))


        self.hidden5 = (nn.Conv2d(128, 128, kernel_size=5, stride=1, padding='same' ))
        self.hidden6 = (nn.Conv2d(192, 4, kernel_size=5, stride=1, padding='same' ))
    
    def forward (self,x):

        x1 = F.relu (self.input_layer(x))
        x2 = F.relu (self.hidden1(x1))
        x3 = F.relu (self.hidden2(x2))
        x4 = F.relu (self.hidden3(x3))

        x5 = torch.cat ((F.relu(self.hidden4(x4)),x3), dim =1)
        x6 = torch.cat ((F.relu(self.hidden5(x5)),x2), dim =1)
        

        out = (self.hidden6(x6))


        return out, x1, x2, x3, x4, x5, x6


net = CNN()

net.cuda()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print('**** Number of Trainable Parameters in BNN')
count_parameters(net)


batch_size = 10
num_epochs = 15
num_samples = 2
trainN = 7000
lambda_reg =0.2
wavenum_init=30
wavenum_init_ydir=40



for epoch in range(0, num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for (loop1,loop2) in zip(fileList_train_U,fileList_train_V):
     print('Training loop index',loop1)
     print('Training loop index',loop2)


     psi_train_input_Tr_torch, psi_train_label_Tr_torch = load_train_data(loop1, loop2, lead, trainN)


###### normalize each batch ##########

     M_train_level1 = torch.mean(torch.flatten(psi_train_input_Tr_torch[:,0,:,:]))
     STD_train_level1 = torch.std(torch.flatten(psi_train_input_Tr_torch[:,0,:,:]))

     M_train_level2 = torch.mean(torch.flatten(psi_train_input_Tr_torch[:,1,:,:]))
     STD_train_level2 = torch.std(torch.flatten(psi_train_input_Tr_torch[:,1,:,:]))


     M_train_level3 = torch.mean(torch.flatten(psi_train_input_Tr_torch[:,2,:,:]))
     STD_train_level3 = torch.std(torch.flatten(psi_train_input_Tr_torch[:,2,:,:]))

     M_train_level4 = torch.mean(torch.flatten(psi_train_input_Tr_torch[:,3,:,:]))
     STD_train_level4 = torch.std(torch.flatten(psi_train_input_Tr_torch[:,3,:,:]))



     psi_train_input_Tr_torch_norm_level1 = ((psi_train_input_Tr_torch[:,0,None,:,:]-M_train_level1)/STD_train_level1)
     psi_train_input_Tr_torch_norm_level2 = ((psi_train_input_Tr_torch[:,1,None,:,:]-M_train_level2)/STD_train_level2)
     psi_train_input_Tr_torch_norm_level3 = ((psi_train_input_Tr_torch[:,2,None,:,:]-M_train_level3)/STD_train_level3)
     psi_train_input_Tr_torch_norm_level4 = ((psi_train_input_Tr_torch[:,3,None,:,:]-M_train_level4)/STD_train_level4)



     psi_train_input_Tr_torch  = torch.cat((psi_train_input_Tr_torch_norm_level1,psi_train_input_Tr_torch_norm_level2,psi_train_input_Tr_torch_norm_level3,psi_train_input_Tr_torch_norm_level4),1)



     psi_train_label_Tr_torch_norm_level1 = (psi_train_label_Tr_torch[:,0,None,:,:]-M_train_level1)/STD_train_level1
     psi_train_label_Tr_torch_norm_level2 = (psi_train_label_Tr_torch[:,1,None,:,:]-M_train_level2)/STD_train_level2
     psi_train_label_Tr_torch_norm_level3 = (psi_train_label_Tr_torch[:,2,None,:,:]-M_train_level3)/STD_train_level3
     psi_train_label_Tr_torch_norm_level4 = (psi_train_label_Tr_torch[:,3,None,:,:]-M_train_level4)/STD_train_level4



     psi_train_label_Tr_torch = torch.cat((psi_train_label_Tr_torch_norm_level1,psi_train_label_Tr_torch_norm_level2,psi_train_label_Tr_torch_norm_level3,psi_train_label_Tr_torch_norm_level4),1)







     for step in range(0,trainN,batch_size):
        # get the inputs; data is a list of [inputs, labels]
        indices = np.random.permutation(np.arange(start=step, stop=step+batch_size))
        input_batch, label_batch = psi_train_input_Tr_torch[indices,:,:,:], psi_train_label_Tr_torch[indices,:,:,:]
        print('shape of input', input_batch.shape)
        print('shape of output', label_batch.shape)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
#        output,_,_,_,_,_,_ = net(input_batch.cuda())
        output = directstep(net,input_batch)
        loss = regular_loss(output, label_batch.cuda())
        loss.backward()
        optimizer.step()
        output_val = directstep (net,psi_test_input_Tr_torch[0:num_samples].reshape([num_samples,4,dim_lon,96]))
        val_loss = regular_loss(output_val, psi_test_label_Tr_torch[0:num_samples].reshape([num_samples,4,dim_lon,96]).cuda())
        # print statistics

        if step % 100 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, step + 1, loss))
            print('[%d, %5d] val_loss: %.3f' %
                  (epoch + 1, step + 1, val_loss))
            running_loss = 0.0
print('Finished Training')


torch.save(net.state_dict(), './BNN_lead'+str(lead)+'.pt')

print('BNN Model Saved')

############# Auto-regressive prediction #####################
STD_test_level1 = STD_test_level1.detach().cpu().numpy()
M_test_level1 = M_test_level1.detach().cpu().numpy()


STD_test_level2 = STD_test_level2.detach().cpu().numpy()
M_test_level2 = M_test_level2.detach().cpu().numpy()


STD_test_level3 = STD_test_level3.detach().cpu().numpy()
M_test_level3 = M_test_level3.detach().cpu().numpy()

STD_test_level4 = STD_test_level4.detach().cpu().numpy()
M_test_level4 = M_test_level4.detach().cpu().numpy()

psi_test_label_Tr = psi_test_label_Tr_torch.detach().cpu().numpy()

M=1000
autoreg_pred = np.zeros([M,4,dim_lon,96])

for k in range(0,M):

  if (k==0):

    out = (directstep(net,psi_test_input_Tr_torch[k].reshape([1,4,dim_lon,96]).cuda()))
    autoreg_pred[k,:,:,:] = out.detach().cpu().numpy()

  else:

    out = (directstep(net,torch.from_numpy(autoreg_pred[k-1,:,:,:].reshape([1,4,dim_lon,96])).float().cuda()))
    autoreg_pred[k,:,:,:] = out.detach().cpu().numpy()


autoreg_pred_level1_denorm = autoreg_pred[:,0,None,:,:]*STD_test_level1+M_test_level1
autoreg_pred_level2_denorm = autoreg_pred[:,1,None,:,:]*STD_test_level2+M_test_level2
autoreg_pred_level3_denorm = autoreg_pred[:,2,None,:,:]*STD_test_level3+M_test_level3
autoreg_pred_level4_denorm = autoreg_pred[:,3,None,:,:]*STD_test_level4+M_test_level4


autoreg_pred = np.concatenate((autoreg_pred_level1_denorm,autoreg_pred_level2_denorm,autoreg_pred_level3_denorm,autoreg_pred_level4_denorm),axis=1)





savenc(autoreg_pred, lon, new_lat, path_outputs+'predicted_directstep_lead'+str(lead)+'.nc')
#savenc(psi_test_label_Tr_unnormalized, lon, new_lat, path_outputs+'truth_RK4_FFT_loss_waveinit_'+str(wavenum_init)+'lead'+str(lead)+'.nc')

