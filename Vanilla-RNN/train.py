'''Training script for the vanilla RNN.
'''


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from load_dataset import VizWiz
from rnn_model import VanillaRNN
from torchsummary import summary
from torch.optim import Adam
from torch.nn.functional import one_hot


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    train_dataset = VizWiz(mode='train')
    # test_dataset = VizWiz(mode='test')

    train_generator = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    # test_generator = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)


    #30522 is the size of the vocab set of BERT.
    vanilla_rnn_model = VanillaRNN(hidden_input_size=49152, hidden_layer_size=2048, input_layer_size=3072, output_layer_size=30522)
    optimizer = Adam(vanilla_rnn_model.parameters(), lr=0.001)

    criterion = torch.nn.CrossEntropyLoss()
    vanilla_rnn_model = vanilla_rnn_model.to(device)
#    summary(vanilla_rnn_model, (1, 3072, 2048))

    for epoch_idx in range(1000):


        vanilla_rnn_model.train()
        epoch_loss = 0
        for idx, sample in tqdm(enumerate(train_generator)):

            batch_image_feature, batch_caption_embedding, batch_tokens = sample['image_fv'].to(device), sample['caption_embedding'], sample['tokens_target']


            batch_tokens_target = batch_tokens.reshape(-1)
            length = batch_tokens_target.size()[0]

            output_t, hidden_state_t = None, batch_image_feature

            sum_loss = None
            optimizer.zero_grad()
            for seq_idx in range(length):

                first_run = True if seq_idx == 0 else False

                batch_caption_embedding_t = batch_caption_embedding[seq_idx].to(device) #current caption embedding vector.
                batch_token_target_t = batch_tokens_target[seq_idx] #current token integer.
                batch_token_target_t_one_hot = one_hot(batch_token_target_t, num_classes=30522).unsqueeze(0).float().to(device)

                output_t, hidden_state_t = vanilla_rnn_model(batch_caption_embedding_t, hidden_state_t, first_run)

                loss_t = criterion(input=output_t, target=batch_token_target_t_one_hot)
                if sum_loss == None:
                    sum_loss = loss_t
                else:
                    sum_loss += loss_t

            sum_loss.backward()
            optimizer.step()
            epoch_loss += sum_loss.item()
        print(f"Loss on epoch {epoch_idx} is {epoch_loss}")




if __name__ == '__main__':

    main()





