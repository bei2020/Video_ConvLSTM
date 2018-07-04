# Neural Scene

### training
    python experiment_mnist.py 
   
Target: 10 frames as a sequence of length 10, to predict 10 frames with these input 10 frames
A,B,C --> Network --> D,E,F

Network

Algorith: generate one frame for each input frame, share convLSTM layer between frames, link 1st layer hidden state between frames

