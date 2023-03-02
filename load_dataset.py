import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel


class VizWiz(Dataset):
    '''Loads the VizWiz image captioning dataset using PyTorch module.
       Structure:
            The JSON file contains three keys.
            1. info
            2. images
            3. annotations

            The images items hold the information about the image file such as file names and image id.
            The annotations items hold the information about the caption and image id.
       We will be using the image ID to match the files with their captions in our code.
       A single image may have multiple caption data for it. Therefore, we're going to need to index through the annotation data
       and get the corresponding image for the training. This way, we'll be feeding the model with multiple annotation per image one by one.
    '''

    def __init__(self, folder_location='./dataset/VizWiz/', mode='train'):
        '''Initialize parameters for the dataset.
        '''

        self.base_image_path = f"{folder_location}{mode}/"
        self.annotation_file = json.load(open(folder_location+'annotations/train.json', 'r'))
        self.images = self.annotation_file['images']
        self.annotations = self.annotation_file['annotations']
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel.eval()


        return None


    def __len__(self):
        '''Returns the total amount of data.
        '''
        return len(self.annotations)

    def __getitem__(self, idx):
        '''Returns a single data in a dict format containing the image array and the caption.
           The captions are...
        '''
        caption, image_id = self.annotations[idx]['caption'], self.annotations[idx]['image_id']
        corresponding_image_data = self.images[image_id] #since the data in images are arranged numerically, we can index directly.

        image_filepath = self.base_image_path+corresponding_image_data['file_name']
        image = cv2.imread(image_filepath) #images in BGR format.

        tokenized_caption = self.tokenizer.tokenize(f"[CLS] {caption} [SEP]") #the markings around the caption is necessary for BERT.

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_caption)


        segment_ids = [1] * len(tokenized_caption)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segment_ids])

        with torch.no_grad():

            outputs = self.bertmodel(tokens_tensor, segments_tensor)

            #the hidden states would be a tuple with 13 elements in it where each one has 3 dimensions [batch number, num of tokens, hidden unit size].
            hidden_states = outputs[2]

            token_embeddings = torch.stack(hidden_states, dim=0) #stack the elements in the tuple. Should be [13, 1, x, 768]
            token_embeddings = torch.squeeze(token_embeddings, dim=1) #we don't need the batch dimension.
            token_embeddings = token_embeddings.permute(1,0,2) #swap the layers dimension and the num of tokens dimension. [x, 13, 768]

            #There is no a single answer on which layer we should pick the word embedding from to get the best representation of the word. The BERT authors ran some trial and error and decided that the concatenation from the last 4 hidden layers produced the best F1 score. There has been other experiments that showed otherwise. E.g. https://github.com/hanxiao/bert-as-service concluded that the sweet spot is the second-to-last layer. In this experiment, we'll go with the concatenation of the last 4 layers.

            token_vecs_last_four_concat = [] #would be [x, 3072]

            for token in token_embeddings:
                #token is a [13 x 768] torch tensor. 13 denotes the layers. The first layer is the input embedding and it can be ignored. The other 12 layers are the output from BERT. Since we're only interested in the last 4 layers, we didn't really need that information in the first place.
                vecs = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
                token_vecs_last_four_concat.append(vecs)
            






            print("hidden states")
            print(len(hidden_states))
            print(len(token_vecs_last_four_concat), len(token_vecs_last_four_concat[0]))
            # print(token_embeddings)
            print("hidden states end")
















        return torch.Tensor(3)





if __name__ == '__main__':

    x = VizWiz()
    train_generator = DataLoader(x, batch_size=1, shuffle=True, num_workers=1)
    print('here')
    for i, sample in enumerate(train_generator):
        print("sample", sample)

