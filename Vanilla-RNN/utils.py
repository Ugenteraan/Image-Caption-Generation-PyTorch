'''Helper functions.
'''
import torch
from torch.nn.utils.rnn import pad_sequence

def pad_batch_data(batch_data):
    '''The batch data from the dataloader contains captions with different lengths.
       In order to perform batch processing in RNN, the captions has to be padded with 0 to ensure the same length.
       This is a collate function for the dataloader.
    '''

    image_features = [sample['image_fv'] for sample in batch_data]
    captions = [sample['caption_embedding'] for sample in batch_data]
    tokens = [sample['tokens_target'] for sample in batch_data]

    padded_captions = pad_sequence(captions, batch_first=True)
    padded_tokens = pad_sequence(tokens, batch_first=True)

    return {
        'image_fv': torch.stack(image_features),
        'caption_embedding': padded_captions,
        'tokens_target': padded_tokens
    }
    # return torch.stack(image_features), padded_captions, padded_tokens