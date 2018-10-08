import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset
import numpy as np
import time
import os

from torch.nn import BCEWithLogitsLoss,MSELoss

from processing import roll_to_midi

USE_CUDA  = torch.cuda.is_available()
processor = torch.device("cuda" if USE_CUDA else "cpu")
warnings.filterwarnings('ignore')

class AutoTunes(nn.Module):
    def __init__(self, hidden_sz=32, seq_len=32, note_range = 49):
        super(AutoTunes,self).__init__()
        """
        ----------------------------------------------------------------------
        The model class behind our music generating engine.

        Parameters:
        ######################################################################
        --hidden_sz--
        An integer denoting the size of hidden state for our LSTM.

        --seq_len--
        An integer denoting how many timesteps we have per song.

        --note_range--
        A float denoting the range of the notes we can play.

        ######################################################################
        ----------------------------------------------------------------------
        """
        self.Encoder = nn.LSTM(note_range, hidden_sz)

        self.Decoder = nn.Sequential(nn.Linear(seq_len*hidden_sz, seq_len*note_range),
                                     nn.Sigmoid())
        self.hidden_sz = hidden_sz

    def note_sampler(self, seq_len, note_probs, threshold = 0.25):
        """
        ----------------------------------------------------------------------
        This function keeps only the notes with probability larger than 25%.

        Parameters:
        ######################################################################
        --note_probs--
        The probabilities returned by our model.
        Shape: (batch_sz, seq_len, note_range)

        --threshold--
        The threshold we use to keep notes.
        ######################################################################

        Returns:
        A valid piano roll for each element of the batch, of the same shape as
        note_probs.
        ----------------------------------------------------------------------
        """

        return (1+torch.sign(note_probs - threshold*torch.ones_like(note_probs)))/2.

    def forward(self,song_clip, thresh = 0.35):
        batch_sz = song_clip.shape[0]
        seq_len = song_clip.shape[1]
        hidden_sz = self.hidden_sz

        probs = self.Decoder(self.Encoder(song_clip)[0].view(batch_sz,hidden_sz*seq_len))
        return self.note_sampler(seq_len, probs.view(batch_sz,seq_len,-1), thresh).view(batch_sz,seq_len,-1)


    def train(self, notes, lr_rate = 1e-3, epochs = 100, batch_sz = 128):

        """
        ------------------------------------------------------------------------
        This function trains the model.

        Parameters:
        ########################################################################
        --notes--
        A dataset of the form Notes. See processing.py for more information.

        --gen_notes--
        A dataset of type Notes, used for generation purposes.

        --gen_notes--
        A dataset of type Notes, used for generation purposes.

        --lr_rate--
        A float denoting the learning rate used for the Adam optimizer.

        --epochs--
        An integer denoting the number of times we go through our dataset.

        --batch_sz--
        An integer denoting the size of our batches during training.
        ########################################################################

        Returns:
        None.
        ------------------------------------------------------------------------
        """

        #Initialize parameters.
        start = time.time()
        seq_len = notes.cur_notes.shape[1]
        hidden_sz = self.hidden_sz
        total_count = epochs*(notes.cur_notes.shape[0]//batch_sz)
        count = 0

        optimizer = optim.Adam(self.parameters(), lr = lr_rate)
        note_generator = data.DataLoader(notes,
                                         batch_size=batch_sz, shuffle=True, drop_last = True)
        loss_fn = nn.BCELoss()
        for epoch in range(epochs):
            for cur_notes, next_notes in note_generator:

                encd = self.Encoder(cur_notes)[0].view(batch_sz,hidden_sz*seq_len)
                encd = self.Decoder(encd).view(batch_sz,seq_len,-1)

                loss = loss_fn(encd,next_notes)

                #Clear the gradients and backprop.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #Keep track of training.
                count += 1
                percent = 100*count/(1.0*total_count)
                loss_val = loss.detach().item()
                print('Training... {:4.2f}% complete. Cross Entropy Loss: {:4.4f}    \r'.format(percent,loss_val), end='')
        print("")
        print("Total training time: {:4.2f} seconds.".format(time.time() - start))

    def create_music(self, seeds, midi_folder = "generated_samples",
    thresh=0.25, num_of_clips = 5, save_seed = False, random = True):
        """
        ----------------------------------------------------------------------
        This uses AutoTunes' note prediction to create music. Given a
        collection of pieces of music (seed_notes), we return the music
        generated by each one, limited by some max number (num_of_clips).

        Parameters:
        ######################################################################
        --seed_notes--
        A list of music where each entry is of the form: (seq_len, note_range)

        --midi_folder--
        A string denoting the folder path to where we save our generated music.

        --thresh--
        The threshold for deciding how likely a note has to be so its played.

        --num_of_clips--
        The number of generated samples.

        --save_seed--
        A bool which checks if we want to save the seed files (for comparison).

        --random--
        A bool which checks if you want to just generated random samples.
        ######################################################################
        ----------------------------------------------------------------------
        """

        for i in range(num_of_clips):
            idx = np.random.randint(0,seeds.shape[0])
            idx = idx if random else i
            seed = seeds[idx].unsqueeze(0)

            # Use model to predict the next sequence given primer
            roll = self.forward(seed,thresh).detach()[0]
            generated_mid = roll_to_midi(roll, subdiv=4, program=82,
                                                  pitch_offset=36)
            generated_mid.write(os.path.join("generated_music",  "final" + '{}.mid'.format(i+1)))

            #Save the seed for reference
            if save_seed:
                seed = roll_to_midi(seed[0], subdiv=4, program= 82,
                                                           pitch_offset=36)
                seed.write(os.path.join("generated_music",
                                                  "seed" + '_' + '{}.mid').format(i+1))
