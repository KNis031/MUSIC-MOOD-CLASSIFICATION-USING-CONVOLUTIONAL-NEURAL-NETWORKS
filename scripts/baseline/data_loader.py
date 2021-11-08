import os
import numpy as np
import pickle
from torch.utils import data


class AudioFolder(data.Dataset):
    def __init__(self, root, subset, tr_val='train', split=0):
        self.trval = tr_val
        self.root = root
        fn = '../../data/splits/split-%d/%s_%s_dict.pickle' % (split, subset, tr_val)
        self.get_dictionary(fn)

    def __getitem__(self, index):
        fn = os.path.join(self.root, 'npy', self.dictionary[index]['path'][:-3]+'npy')
        audio = np.array(np.load(fn))

        ### (Added by us) Adding code here to slice the mel spectrograms into 29.1 second segments which corresponds to 1366 windows ###
        slice_len=1366
        middle_window = int(audio.shape[1]/2)
        audio = audio[:,middle_window-(int(slice_len/2)):middle_window+(int(slice_len/2))]
        ### (Added by us) END ### 

        tags = self.dictionary[index]['tags']
        return audio.astype('float32'), tags.astype('float32'), self.dictionary[index]['path']

    def get_dictionary(self, fn):
        with open(fn, 'rb') as pf:
            dictionary = pickle.load(pf)
        self.dictionary = dictionary

    def __len__(self):
        return len(self.dictionary)


def get_audio_loader(root, subset, batch_size, tr_val='train', split=0, num_workers=0):
    dataset = AudioFolder(root, subset, tr_val, split)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader

