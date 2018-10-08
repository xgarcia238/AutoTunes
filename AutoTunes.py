import warnings
import time
import os
import click


from processing import Notes
from Muse import AutoTunes
import torch

USE_CUDA  = torch.cuda.is_available()
processor = torch.device("cuda" if USE_CUDA else "cpu")

warnings.filterwarnings('ignore')

############################## Model Parameters ################################
HIDDEN_SZ = 32
SEQ_LEN = 32
NOTE_RANGE = 49
THRESHOLD = 0.27
EPOCHS = 60
BATCH_SZ = 128
LR_RATE = 1e-3

################################################################################

############################## Command-line Commands ###########################
@click.group()
def cli():
    pass

@cli.command()
#@click.option('--train', is_flag=True, help='Whether we should train the model.')
@click.option('--weights_name', default = "muse_weights", help = "Name of model's weights save file.")
@click.option('--data_location', default = "gameboy_MIDI", help= "Where training data is stored.")
@click.option('--midi/--no-midi', default=True, help= "Checks if music is in MIDI form.")
@click.option('--path', default='training_notes', help = "Save training notes.")
def train(weights_name, data_location, midi, path):
    music = Notes(data_location, MIDI=midi, seq_len=SEQ_LEN, save_path=path)
    muse = AutoTunes(HIDDEN_SZ, SEQ_LEN, NOTE_RANGE).to(processor)
    muse.train(music,LR_RATE, EPOCHS, BATCH_SZ)
    torch.save(muse.state_dict(), weights_name)
    torch.save(music.seeds, 'seeds')
    return

@cli.command()
#@click.pass_context
#@click.option('--generate', is_flag=True, help = 'Whether we should generate music')
#@click.option('')
@click.option('--weights_name', default = "muse_weights", help = "Name of model's weights save file.")
@click.option('--thresh', default=0.26, help = 'Sampling threshold.')
@click.option('--base', default='seeds')
def generate(thresh, weights_name, base):
    muse = AutoTunes(HIDDEN_SZ, SEQ_LEN, NOTE_RANGE).to(processor)
    seeds_tensor = torch.load(base)
    muse.load_state_dict(torch.load(weights_name))
    muse.create_music(seeds_tensor,thresh)
    return

################################################################################
if __name__ == '__main__':
    cli()
