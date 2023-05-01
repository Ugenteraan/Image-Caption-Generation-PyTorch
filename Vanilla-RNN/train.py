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

    train_generator = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
    # test_generator = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)


    #30522 is the size of the vocab set of BERT.
    vanilla_rnn_model = VanillaRNN(hidden_input_size=49152, hidden_layer_size=2048, input_layer_size=3072, output_layer_size=30522)
    optimizer = Adam(vanilla_rnn_model.parameters(), lr=0.001)

    criterion = torch.nn.CrossEntropyLoss()
    vanilla_rnn_model = vanilla_rnn_model.to(device)
#    summary(vanilla_rnn_model, (1, 3072, 2048))

    for epoch_idx in range(5):


        vanilla_rnn_model.train()
        epoch_loss = 0
        for idx, sample in tqdm(enumerate(train_generator)):

            if idx == 5000:
                break
            batch_image_feature, batch_caption_embedding, batch_tokens = sample['image_fv'].to(device), sample['caption_embedding'], sample['tokens_target']


            batch_tokens_target = batch_tokens.reshape(-1)
            length = batch_tokens_target.size()[0]

            output_t, hidden_state_t = None, batch_image_feature

            output_sequences = None
            optimizer.zero_grad()
            for seq_idx in range(length):

                batch_caption_embedding_t = batch_caption_embedding[seq_idx].to(device) #current caption embedding vector.
                batch_token_target_t = batch_tokens_target[seq_idx] #current token integer.
                # batch_token_target_t_one_hot = one_hot(batch_token_target_t, num_classes=30522).unsqueeze(0).float().to(device)

                output_t, hidden_state_t = vanilla_rnn_model(batch_caption_embedding_t, hidden_state_t)

                if output_sequences == None:
                    output_sequences = output_t.reshape(1, -1)
                else:
                    output_sequences = torch.concat([output_sequences, output_t.reshape(1, -1)], dim=0)


            batch_token_target_t_one_hot = one_hot(batch_tokens_target, num_classes=30522).float().to(device)
            # print(output_sequences, batch_token_target_t_one_hot)
            # print(output_sequences.size(), batch_token_target_t_one_hot.size())
            sample_loss = criterion(input=output_sequences, target=batch_token_target_t_one_hot)

            # print(sample_loss, sample_loss.size())
            sample_loss.backward()
            optimizer.step()
            epoch_loss += sample_loss.item()
        print(f"Loss on epoch {epoch_idx} is {epoch_loss}")


   vanilla_rnn_model.eval()

   dummy_input_caption_embedding = torch.randn(1, 3072)
   dummy_input_image_feature = torch.randn(1, 49152)

   dummy_output,hidden_state = vanilla_rnn_model(dummy_input_caption_embedding, dummy_input_image_feature)

   torch.onnx.export(vanilla_rnn_model, args=(dummy_input_caption_embedding, dummy_input_image_feature),
                        f='vanilla_rnn.onnx', input_names=['caption_input', 'image_feature_input'], output_names=['word', 'hidden_state'])


if __name__ == '__main__':

    main()






