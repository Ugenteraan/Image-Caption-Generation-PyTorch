'''Dataset Loading.

   The first goal here is to tokenize each caption data and feed it into a language model. In this code, we're using a pretrained Word2Vec from SpaCy. The output from the model will be used to train the image caption generation model.
   The second goal here is to get the feature vector of each image using a pretrained CNN model.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import spacy
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, Swinv2Model



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


    def __init__(self, spacy_vocab, folder_location='./dataset/VizWiz/', mode='train', transforms=None):
        '''Initialize parameters for the dataset.
        '''

        self.base_image_path = f"{folder_location}{mode}/"
        self.annotation_file = json.load(open(folder_location+f'annotations/{mode}.json', 'r'))
        self.images = self.annotation_file['images']
        self.annotations = self.annotation_file['annotations']
        self.nlp = spacy.load(spacy_vocab)
        self.autoprocessor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.transforms = transforms
        self.vocab = list(self.nlp.strings) #vocabulary of words.

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
        corresponding_image_data = self.images[image_id] #Since the data in images are arranged numerically, we can index directly.

        image_filepath = self.base_image_path+corresponding_image_data['file_name']
        image = Image.open(image_filepath)

        processed_image = self.autoprocessor(image, return_tensors='pt')

        tokens = self.nlp(caption)

        token_vector_list = []
        token_target_list = []
        for token in tokens:
            token_vector_list.append(token.vector) #append the word embeddings into the list.
            token_target_list.append(self.vocab.index(token))

        token_vector_numpy, token_target_numpy = np.asarray(token_vector_list), np.asarray(token_target_list)
        token_vector, token_target = torch.from_numpy(token_vector_numpy), torch.from_numpy(token_target_numpy)

        return {
            'image_fv': img_feature_vector,
            'caption_embedding': token_vector,
            'tokens_target': token_target
        }



if __name__ == '__main__':

    x = VizWiz(spacy_vocab='en_core_web_lg')
    train_generator = DataLoader(x,  batch_size=1, shuffle=True, num_workers=1)
    print("Length: " , train_generator.__len__())
    for i, sample in enumerate(train_generator):
        print("sample", sample['image_fv'].size(), len(sample['caption_embedding']), sample['caption_embedding'][0].size(), sample['tokens_target'], sample['tokens_target'].size())
        pass

