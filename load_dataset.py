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

        print(caption, tokenized_caption)









        return torch.Tensor(3)





if __name__ == '__main__':

    x = VizWiz()
    train_generator = DataLoader(x, batch_size=1, shuffle=True, num_workers=1)
    print('here')
    for i, sample in enumerate(train_generator):
        print("sample", sample)

