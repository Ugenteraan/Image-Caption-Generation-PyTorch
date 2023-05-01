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
import cfg

def main():


    train_dataset = VizWiz(mode='train')
    # test_dataset = VizWiz(mode='test')

    train_generator = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=cfg.SHUFFLE, num_workers=cfg.NUM_WORKERS)
    # test_generator = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)


    vanilla_rnn_model = VanillaRNN(hidden_input_size=cfg.HIDDEN_FEATURE_SIZE, hidden_layer_size=cfg.HIDDEN_LAYER_SIZE, input_layer_size=cfg.INPUT_LAYER_SIZE, output_layer_size=cfg.OUTPUT_LAYER_SIZE)

    optimizer = Adam(vanilla_rnn_model.parameters(), lr=cfg.LEARNING_RATE)

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    vanilla_rnn_model = vanilla_rnn_model.to(cfg.DEVICE)

    for epoch_idx in range(10):


        vanilla_rnn_model.train()
        epoch_loss = 0
        for idx, sample in tqdm(enumerate(train_generator)):


            batch_image_feature, batch_caption_embedding, batch_tokens = sample['image_fv'].to(cfg.DEVICE), sample['caption_embedding'], sample['tokens_target']


            batch_tokens_target = batch_tokens.reshape(-1)
            length = batch_tokens_target.size()[0]

            output_t, hidden_state_t = None, batch_image_feature

            output_sequences = None
            optimizer.zero_grad()
            for seq_idx in range(length):

                batch_caption_embedding_t = batch_caption_embedding[seq_idx].to(cfg.DEVICE) #current caption embedding vector.

                output_t, hidden_state_t = vanilla_rnn_model(batch_caption_embedding_t, hidden_state_t)

                if output_sequences == None:
                    output_sequences = output_t.reshape(1, -1)
                else:
                    output_sequences = torch.concat([output_sequences, output_t.reshape(1, -1)], dim=0)


            batch_token_target_t_one_hot = one_hot(batch_tokens_target, num_classes=cfg.NUM_TOKENS).float().to(cfg.DEVICE)

            sample_loss = criterion(input=output_sequences, target=batch_token_target_t_one_hot)

            sample_loss.backward()

            optimizer.step()
            epoch_loss += sample_loss.item()

        print(f"Loss on epoch {epoch_idx} is {epoch_loss}")
        torch.save(vanilla_rnn_model, './vanilla-rnn-model.pth')


    vanilla_rnn_model.eval()

    dummy_input_caption_embedding = torch.randn(1, 3072).to(cfg.DEVICE)
    dummy_input_image_feature = torch.randn(1, 49152).to(cfg.DEVICE)

    dummy_output,hidden_state = vanilla_rnn_model(dummy_input_caption_embedding, dummy_input_image_feature)

    torch.onnx.export(vanilla_rnn_model,
                        args=(dummy_input_caption_embedding, dummy_input_image_feature),
                        f='vanilla_rnn.onnx', input_names=['caption_input', 'image_feature_input'],
                        output_names=['word', 'hidden_state'])


if __name__ == '__main__':

    main()






