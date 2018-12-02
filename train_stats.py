
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import numpy as np
from tqdm import tqdm

from helpers import *
from model import *
# from generate import *

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--n_epochs', type=int, default=100)
argparser.add_argument('--seq_len', type=int, default=1)
argparser.add_argument('--print_every', type=int, default=5)
argparser.add_argument('--hidden_size', type=int, default=2000)
argparser.add_argument('--n_layers', type=int, default=1)
argparser.add_argument('--learning_rate', type=float, default=0.001)
argparser.add_argument('--chunk_len', type=int, default=50)
argparser.add_argument('--batch_size', type=int, default=20)
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--cuda', action='store_true')
args = argparser.parse_args()

if args.cuda:
    print("Using CUDA")

data_train = np.load('array1.npy')
data_test = np.load('array2.npy')


file_len_train = data_train.shape[0]
file_len_test = data_test.shape[0]

data_pred=np.zeros((file_len_test,46))
data_train_pred=np.zeros((file_len_train,46))

data_fed=np.zeros((file_len_test,46))

feature_len = 46
data_train = torch.from_numpy(data_train)
data_test = torch.from_numpy(data_test)

seq_len=args.seq_len

def training_set(chunk_len, batch_size):
    batch_coverage = batch_size * chunk_len
    num_batches = (file_len_train) // batch_coverage
    inp = torch.zeros(num_batches, batch_size, chunk_len, feature_len)
    target = torch.zeros(num_batches, batch_size, chunk_len, feature_len)
    for batch in range(num_batches):
        start_index = batch * batch_coverage
        for bi in range(batch_size):
            end_index = start_index + chunk_len + seq_len
            chunk = data_train[start_index:end_index,:]
            if len(chunk)<chunk_len+seq_len:
                inp[batch,bi,:,:] = chunk
                target[batch,bi,:,:] = torch.cat((chunk[seq_len:,:],chunk[-seq_len:,:].view(-1,feature_len)),dim=0)
            else:

                inp[batch,bi,:,:] = chunk[:-seq_len,:]
                target[batch,bi,:,:] = chunk[seq_len:,:]
            start_index = end_index - seq_len
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def testing_set(chunk_len, batch_size):
    batch_coverage = batch_size * chunk_len
    num_batches = (file_len_test) // batch_coverage
    inp = torch.zeros(num_batches, batch_size, chunk_len, feature_len)
    target = torch.zeros(num_batches, batch_size, chunk_len, feature_len)
    for batch in range(num_batches):
        start_index = batch * batch_coverage
        for bi in range(batch_size):
            end_index = start_index + chunk_len + 1
            chunk = data_test[start_index:end_index,:]
            if len(chunk)<chunk_len+1:
                inp[batch,bi,:,:] = chunk
                target[batch,bi,:,:] = torch.cat((chunk[1:,:],chunk[-1,:].view(-1,feature_len)),dim=0)
            else:

                inp[batch,bi,:,:] = chunk[:-1,:]
                target[batch,bi,:,:] = chunk[1:,:]
            start_index = end_index - 1
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target


def train(inp, target, batch_num):
    batch_size=args.batch_size
    chunk_len=args.chunk_len
    batch_coverage = batch_size * chunk_len
    num_batches = (file_len_test) // batch_coverage
    hidden = decoder.init_hidden(args.batch_size)
    
    if args.cuda:
        c,h=hidden
        c = c.cuda()
        h = h.cuda()
        hidden=(c,h)

    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = decoder(inp[batch_num,:,c,:],hidden)
        loss += criterion(output.view(args.batch_size, -1),target[batch_num,:,c,:])
        out_put=output.data.numpy()
        in_put=inp[batch_num,:,c,:].data.numpy()
        for j in range(args.batch_size):
             data_train_pred[batch_coverage*batch_num+chunk_len*j+c,:]=out_put[j,:]

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / args.chunk_len


def test(inp, target, batch_num):
    batch_size=args.batch_size
    chunk_len=args.chunk_len
    batch_coverage = batch_size * chunk_len
    num_batches = (file_len_test) // batch_coverage
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        c,h=hidden
        c = c.cuda()
        h = h.cuda()
        hidden=(c,h)

    decoder.eval()
    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = decoder(inp[batch_num,:,c,:],hidden)
        loss += criterion(output.view(args.batch_size, -1),target[batch_num,:,c,:])
        # test_output.append(output[0,:])
        # test_target.append(target[batch_num,:,c,:])
        out_put=output.data.numpy()
        in_put=inp[batch_num,:,c,:].data.numpy()
        for j in range(args.batch_size):
             data_pred[batch_coverage*batch_num+chunk_len*j+c,:]=out_put[j,:]
             #data_fed[batch_coverage*batch_num+chunk_len*j+c,:]=in_put[j,:]



    return loss.item() / args.chunk_len

def save():
    save_filename = 'soccer.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

# Initialize models and start training

decoder = SoccerRNN(
    args.seq_len,
    feature_len,
    args.hidden_size
)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.MSELoss()

if args.cuda:
    decoder.cuda()

start = time.time()
all_losses = []
total_loss = 0
total_test_loss=0
try:
    print("Training for %d epochs..." % args.n_epochs)
    batch_coverage_train = args.batch_size * args.chunk_len
    batch_coverage_test = args.batch_size * args.chunk_len

    num_batches_train = file_len_train// batch_coverage_train
    num_batches_test = file_len_test// batch_coverage_test

    for epoch in tqdm(range(1, args.n_epochs + 1)):
        total_loss = 0
        total_test_loss=0
        for batch_num in range(num_batches_train):
            #print('Epoch is',epoch,'batch num is' ,batch_num)
            #inp,target=training_set(args.chunk_len, args.batch_size)
            #print(inp.shape)
            loss = train(*training_set(args.chunk_len, args.batch_size), batch_num)
            total_loss += loss
        


        for batch_num in range(num_batches_test):
            test_loss=test(*testing_set(args.chunk_len, args.batch_size),batch_num)
            total_test_loss+=test_loss



        print('Train loss ', total_loss / num_batches_train)
        print('Test loss ', total_test_loss / num_batches_test)

        np.save('data_pred_lr_%s_batch_size_%s_hidden_state_%s_layers_%s_%s.npy' % (args.learning_rate,args.batch_size,args.hidden_size,args.n_layers, str(epoch)),data_pred)        
        np.save('data_train_pred_lr_%s_batch_size_%s_hidden_state_%s_layers_%s_%s.npy' % (args.learning_rate,args.batch_size,args.hidden_size,args.n_layers, str(epoch)),data_train_pred)        

        #np.save('data_fed_hidden_state_500_layers_5_%s.npy' % str(epoch),data_fed)

        if epoch % args.print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, total_loss))
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, total_test_loss))
   
    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()
