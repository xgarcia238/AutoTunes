# AutoTunes

An automatic music generation engine built with PyTorch! Currently, it only produces single-track music and has only really been tested with S-NES and GameBoy music.

For instructions on how to use, I recommend reading the attached Jupyter Notebook. The code itself has also been heavily documented, so you may feel free to read that. If you prefer command-line tools, you can also clone the repo and move into that directory. We'll assume you have your set of MIDI files, but if not, you can unzip the gameboyMIDI.zip into a folder titled "gameboyMIDI". Then, run the following commands.

```
python3 AutoTunes.py train --data_location='gameboyMIDI' --midi --path='GameBoy'
python3 AutoTunes.py generate 
```

You will find the model's output as MIDI files, which might be cumbersome to play. I recommend importing them to [Online Sequencer](https://onlinesequencer.net/import) and choosing an adequate instrument to use with the music. My personal favorites are "Synth Pluck","Concert Harp" and "Scifi" but you should definitely explore and see what you like. Some of the snippets of music you form may not be to your liking. Feel free to simply create more until you find one you like. The larger the dataset you use to train, the more stability you'll find in the model. 

For reference, using SNES songs for training and GameBoy songs as seeds for generating music, we produced the following clips:

[Clip 1](https://www.youtube.com/watch?v=jBg0v-mOAN0)

[Clip 2](https://www.youtube.com/watch?v=kKFUJWCmJzQ)

[Clip 3](https://www.youtube.com/watch?v=Ilj7s7OvUa8)

For comparison, I'm also attaching the seeds used for generation for each clip:

[Seed for Clip 1](https://www.youtube.com/watch?v=hL-20_ZUwSw)

[Seed for Clip 2](https://www.youtube.com/watch?v=upahaJC-zDg)

[Seed for Clip 3](https://www.youtube.com/watch?v=GiJC6MxM2CU)

## Author

**Xavier Garcia** 


## License

This project is licensed under the MIT License - see the [license](LICENSE.md) file for details.
