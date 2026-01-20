# XTTSv2 with Slovak support

XTTSv2 fine-tuned on the Mozilla Common Voice dataset. It's not perfect and still needs some work.

Repository containing training code: https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages

## Deploy

Model repository: https://huggingface.co/Felagund/XTTSv2-sk

1. Download model.pth, vocab.json and config.json from the model repository
2. Record audio of you speaking (6s minimal) and save it as recording.wav, in the same dir as example.py
3. Run example.py
