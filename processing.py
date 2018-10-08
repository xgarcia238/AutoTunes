import os
import math
import torch
from torch.utils.data import Dataset

import pretty_midi
import numpy as np

USE_CUDA  = torch.cuda.is_available()
processor = torch.device("cuda" if USE_CUDA else "cpu")

class Notes(Dataset):
    def __init__(self, path, processor=processor, MIDI = False , seq_len = 64,
                 time_sigs = ['4/4'], keys = False, subdiv = 4,
                save_path = 'song_data', min_note = 36, max_note = 84):
        """
        ----------------------------------------------------------------------
        A custom dataset for handling our music. This prepares music from MIDI
        or, if the music has been processed, we can also divide it in different
        sequence lengths.

        Parameters:
        ######################################################################
        --path--
        A string containing the path to the data.

        --processor--
        Checks whether to use a gpu or cpu.

        --MIDI--
        A bool which checks whether the songs are in MIDI format.

        --seq_len--
        An integer that denotes the number of timesteps per song.

        --time_sigs--
        This will be a list of possible time signatures.

        --keys--
        A list of valid keys. If False, defaults to all keys.

        --subdiv--
        The number of steps per subdivision.

        --save_path--
        A string denoting the folder to save the songs as a tensor for later usage.

        --min_note--
        An integer denoting the smallest note you can use.

        --max_note--
        An integer denoting the largest note you can useself.
        ######################################################################
        ----------------------------------------------------------------------
        """
        if not MIDI:
            songs = torch.load(path, map_location=lambda storage, loc: storage)
            self.cur_notes, self.next_notes = cur_next_split(songs, seq_len)

            if self.cur_notes.shape[2] == 128: #If we stored entire songs, just restrict
                self.cur_notes = self.cur_notes[:,:,min_note:max_note+1].float()
                self.next_notes = self.next_notes[:,:,min_note:max_note+1].float()

            if self.cur_notes.shape[2] != max_note+1-min_note:
                raise ValueError("The loaded tensor does not have the right note range.")


        else:
            valid_keys = ['Db Major', 'D Major', 'Eb Major', 'E Major',
                          'F Major', 'Gb Major', 'G Major', 'Ab Major',
                          'A Major', 'Bb Major', 'B Major', 'C minor',
                          'C Major', 'C# minor', 'D minor', 'Eb minor',
                          'E minor', 'F minor', 'F# minor', 'G minor',
                          'G# minor', 'A minor', 'Bb minor', 'B minor']
            valid_keys = valid_keys if not keys else keys

            # Otherwise gather MIDIs and build dataset
            if os.path.exists(path):
                songs = midi_cleaner(path, time_sigs, valid_keys)
                midi_data = []
                errors = 0
                for song in songs:
                    try:
                        tensor = midi_to_roll(song, subdiv=subdiv, transpose_notes=True)
                        midi_data.append(tensor)
                    except:
                        errors += 1
            else:
                raise Exception('Invalid Path! Maybe a typo?')
            print("Number of errors: ", errors)
            midi_data = torch.cat(midi_data,0)
            torch.save(midi_data, save_path) # Save array containing dataset for future use
            self.cur_notes, self.next_notes = cur_next_split(midi_data, seq_len)
            self.cur_notes = self.cur_notes[:,:,min_note:max_note+1].float()
            self.next_notes = self.next_notes[:,:,min_note:max_note+1].float()

        #Finally, make a seed set by taking 10% of the data.
        seeds_sz = self.cur_notes.shape[0]//10
        perm = torch.randperm(self.cur_notes.shape[0])
        self.cur_notes, self.next_notes  = self.cur_notes[perm], self.next_notes[perm]
        self.seeds = self.cur_notes[:seeds_sz].to(processor)
        self.cur_notes = self.cur_notes[seeds_sz:].to(processor)
        self.next_notes = self.next_notes[seeds_sz:].to(processor)

    def __len__(self):
        return self.cur_notes.shape[0]

    def __getitem__(self, idx):
        return self.cur_notes[idx], self.next_notes[idx]


def cur_next_split(data, seq_len):
    '''
    --------------------------------------------------------------------------
    Split the dataset into a training set and corresponding set of labels.

    Parameters:
    ##########################################################################
    --data--
    A tensor of shape (total_time,note_range). See the NotesDataset definition
    for context.

    --seq_len
    An integer denoting the number of timesteps per song.
    ##########################################################################

    Returns:
    ##########################################################################
    --cur_notes--
    A tensor of shape (batch_sz,seq_len, note_range).

    --next_notes--
    A tensor of shape (batch_sz,seq_len, note_range) i.e. the notes we are
    trying to predict.
    ##########################################################################


    --------------------------------------------------------------------------
     '''

    cur_notes = []
    next_notes = []

    if data.shape[0] % seq_len != 0:
        zero_pad = torch.zeros((seq_len - data.shape[0] % seq_len, data.shape[1]))
        data = torch.cat((data.float(), zero_pad.float()), 0)


    for i in range(0, len(data) - seq_len, seq_len):

        cur_notes.append(data[i:i+seq_len].unsqueeze(0))
        next_notes.append(data[i + seq_len:i + 2*seq_len].unsqueeze(0))

    return torch.cat(cur_notes,0), torch.cat(next_notes,0)


# Exception class for MIDI-specific errors
class MIDIError(Exception):
    pass


def midi_cleaner(data_folder, allowed_time_sigs, allowed_keys,
 max_time_changes=1, max_key_changes=1, ignore_filters=False):
    '''
    ----------------------------------------------------------------------------
    A function to filter a group of MIDI files by selecting only the ones that
    meet the specified criteria supplied for key, time signature.

    The files are returned as a list of pretty_midi objects.

    Parameters:
    ############################################################################
    --data_folder--
    The path of the folder containing the files to be filtered

    --allowed_time_sigs--
    The time signatures to be allowed as an array of strings e.g. ['4/4']

    --allowed_keys--
    The key signatures to be allowed as an array of strings e.g. ['C Major', 'Bb Minor']

    --max_time_changes--
    The maximum number of time signature changes allowed. Default is 1.

    --max_key_changes--
    The maximum number of key signature changes allowed. Default is 1.

    --ignore_filters--
    If true, all MIDI files in the folder will be converted .

    ############################################################################

    Returns:
    A list of pretty_midi objects meeting the supplied filter settings
    ----------------------------------------------------------------------------
    '''

    midi_files = os.listdir(data_folder)

    errors = 0
    filtered_files = []
    size, count = len(midi_files), 0.0
    for num, midi_file in enumerate(midi_files):
        #print('Processing file {} of {}'.format(num+1,len(midi_files)))

        try:
            mid = pretty_midi.PrettyMIDI(os.path.join(data_folder, midi_file))

            if not ignore_filters:
                time_sig_is_good, key_is_good = False, False

                if mid.time_signature_changes and len(mid.time_signature_changes) <= max_time_changes:
                    time_sig_is_good = all('{}/{}'.format(ts.numerator,ts.denominator) in allowed_time_sigs for ts in mid.time_signature_changes)

                if mid.key_signature_changes and len(mid.key_signature_changes) <= max_key_changes:
                    key_is_good = all(pretty_midi.key_number_to_key_name(key.key_number) in allowed_keys for key in mid.key_signature_changes)

                if time_sig_is_good and key_is_good:
                    filtered_files.append(mid)

            else:
                filtered_files.append(mid)
        except:
            errors += 1
        count += 1
        print('Processing files... {:4.2f}% complete.   \r'.format(100*count/size), end='')
    print("")
    print('{} MIDI files found.'.format(len(filtered_files)))

    if errors:
        print('{} files could not be parsed.'.format(errors))

    return filtered_files


def midi_to_roll(mid, subdiv=4,
sensitivity=0.2, transpose_notes=False, drums=False):

    '''
    --------------------------------------------------------------------------
    Encodes a pretty_midi object as a piano-roll matrix with shape (t, 128)
    where the first axis is the number of timesteps and the second axis is
    MIDI pitch.

    Parameters:
    ##########################################################################
    --mid--
    The pretty_midi object to be encoded

    --subdiv--
    The resolution at which to sample notes in the song.

    --sensitivity--
    For notes that don't fall exactly on the grid to be included.

    --transpose_notes--
    If true, the notes will be transposed to C before encoding.

    --drums--
    A bool which checks whether we want to keep drum instruments.
    ##########################################################################

    Returns:
    A tensor of shape (t, 128) encoding the notes in the pretty_midi object,
    --------------------------------------------------------------------------
    '''

    if not (0 <= sensitivity < 0.5):
        raise Exception('Sensitivity must be in [0, 0.5).')

    if transpose_notes:
        try:
            transpose_to_c(mid)
        except:
            raise MIDIError('MIDI file could not be transposed.')

    if mid.resolution % subdiv == 0:
        step_size = mid.resolution // subdiv
    else:
        raise MIDIError('Invalid step size.')

    end_ticks = mid.time_to_tick(mid.get_end_time())
    num_measures = math.ceil(end_ticks / (mid.resolution * 4)) # Assumes 4/4 time

    piano_roll = torch.zeros((num_measures * subdiv * 4, 128))

    for inst in mid.instruments:
        if not drums and inst.is_drum:
            continue

        for note in inst.notes:
            pos = mid.time_to_tick(note.start) / step_size
            step = round(pos)

            # Ensure that notes don't jump between measures and prevent out of bounds errors
            if step % (subdiv * 4) == 0 and pos < step:
                step -= 1

            # If note is in the right range, add it to the piano roll
            if step - sensitivity <= pos <= step + sensitivity:
                note_start = int(step)
                piano_roll[note_start, note.pitch] = 1

    return piano_roll


def roll_to_midi(piano_roll, subdiv=4, program=82,
 tempo=120, resolution=480, pitch_offset=0):
    '''
    --------------------------------------------------------------------------
    This function takes an array and turns it back into MIDI.

    Parameters:
    ##########################################################################
    --piano_roll--
    The numpy array to be decoded

    --subdiv--
    The number of steps per quarter note.

    --program--
    The MIDI program number to use for playback.

    --tempo--
    The tempo of the pretty_midi object in BPM. Default is 120.

    --resolution--
    The resolution of the pretty_midi object.

    --pitch_offset--
    Adds an offset to pitch indices.
    ##########################################################################

    Returns:
    A pretty_midi object based on the contents of the numpy array
    --------------------------------------------------------------------------
    '''

    step_size = resolution // subdiv

    mid = pretty_midi.PrettyMIDI(initial_tempo=tempo, resolution=resolution)

    inst = pretty_midi.Instrument(program=program - 1, is_drum=False)
    mid.instruments.append(inst)

    for i, dur in np.ndenumerate(piano_roll):
        if dur:
            note_start = i[0] * step_size
            note_end = int(note_start + step_size*dur)
            note = pretty_midi.Note(velocity=110, pitch=pitch_offset + i[1],
                                    start=mid.tick_to_time(note_start),
                                    end=mid.tick_to_time(note_end))

            mid.instruments[0].notes.append(note)

    return mid



def transpose(mid, semitones):
    '''
    --------------------------------------------------------------------------
    Transposes all the notes in a pretty_midi object by the specified number
    of semitones. Any drum instruments in the object will not be modified.

    Parameter:
    ##########################################################################
    --mid--
    The pretty_midi object to be transposed

    --semitones--
    The number semitones to transpose the notes up (positive) or down (negative).
    ##########################################################################
    --------------------------------------------------------------------------
    '''

    for inst in mid.instruments:
        if not inst.is_drum: # Don't transpose drum tracks
            for note in inst.notes:
                note.pitch += semitones


def transpose_to_c(mid):
    '''
    ----------------------------------------------------------------------------
    A special case of transpose that moves all notes to the key of C
    Note that in order to know the how many semitones to transpose, the
    original key. Otherwise an exception will be thrown.

    Parameters:
    --mid-- The pretty_midi object to be transposed
    ----------------------------------------------------------------------------
    '''

    if mid.key_signature_changes:
        key = mid.key_signature_changes[0]
    else:
        raise MIDIError('MIDI key signature could not be determined.')

    pos_in_octave = key.key_number % 12

    if not pos_in_octave == 0:
        semitones = -pos_in_octave if pos_in_octave < 6 else 12 - pos_in_octave # Transpose up or down given dist from C

        transpose(mid, semitones)
