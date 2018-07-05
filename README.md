# Neural Scene

### training
    python experiment_mnist.py 
   
Target: 10 frames as a sequence of length 10, to predict 10 frames with these input 10 frames
A,B,C --> Network --> D,E,F

Network

Algorith: generate one frame for each input frame, share convLSTM layer between frames, link 1st layer hidden state between frames


    CL/enco_cl1/inputc:0 (1, 16, 16, 128)=32768, CL/enco_cl1/forgetc:0 (1, 16, 16, 128)=32768, CL/enco_cl1/outputc:0 (1, 16, 16, 128)=32768, CL/enco_cl1/inputx2state/kernel:0 (5, 5, 16, 128)=51200, CL/enco_cl1/inputh2state/kernel:0 (5, 5, 128, 128)=409600, CL/enco_cl1/inputh2state/bias:0 128, CL/enco_cl1/forgetx2state/kernel:0 (5, 5, 16, 128)=51200, CL/enco_cl1/forgeth2state/kernel:0 (5, 5, 128, 128)=409600, CL/enco_cl1/forgeth2state/bias:0 128, CL/enco_cl1/cellx2state/kernel:0 (5, 5, 16, 128)=51200, CL/enco_cl1/cellh2state/kernel:0 (5, 5, 128, 128)=409600, CL/enco_cl1/cellh2state/bias:0 128, CL/enco_cl1/outputx2state/kernel:0 (5, 5, 16, 128)=51200, CL/enco_cl1/outputh2state/kernel:0 (5, 5, 128, 128)=409600, CL/enco_cl1/outputh2state/bias:0 128, CL/enco_cl2/inputc:0 (1, 16, 16, 64)=16384, CL/enco_cl2/forgetc:0 (1, 16, 16, 64)=16384, CL/enco_cl2/outputc:0 (1, 16, 16, 64)=16384, CL/enco_cl2/inputx2state/kernel:0 (5, 5, 128, 64)=204800, CL/enco_cl2/inputh2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl2/inputh2state/bias:0 64, CL/enco_cl2/forgetx2state/kernel:0 (5, 5, 128, 64)=204800, CL/enco_cl2/forgeth2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl2/forgeth2state/bias:0 64, CL/enco_cl2/cellx2state/kernel:0 (5, 5, 128, 64)=204800, CL/enco_cl2/cellh2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl2/cellh2state/bias:0 64, CL/enco_cl2/outputx2state/kernel:0 (5, 5, 128, 64)=204800, CL/enco_cl2/outputh2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl2/outputh2state/bias:0 64, CL/enco_cl3/inputc:0 (1, 16, 16, 64)=16384, CL/enco_cl3/forgetc:0 (1, 16, 16, 64)=16384, CL/enco_cl3/outputc:0 (1, 16, 16, 64)=16384, CL/enco_cl3/inputx2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl3/inputh2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl3/inputh2state/bias:0 64, CL/enco_cl3/forgetx2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl3/forgeth2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl3/forgeth2state/bias:0 64, CL/enco_cl3/cellx2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl3/cellh2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl3/cellh2state/bias:0 64, CL/enco_cl3/outputx2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl3/outputh2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl3/outputh2state/bias:0 64, CL/fore_cl1/inputc:0 (1, 16, 16, 128)=32768, CL/fore_cl1/forgetc:0 (1, 16, 16, 128)=32768, CL/fore_cl1/outputc:0 (1, 16, 16, 128)=32768, CL/fore_cl1/inputx2state/kernel:0 (5, 5, 64, 128)=204800, CL/fore_cl1/inputh2state/kernel:0 (5, 5, 128, 128)=409600, CL/fore_cl1/inputh2state/bias:0 128, CL/fore_cl1/forgetx2state/kernel:0 (5, 5, 64, 128)=204800, CL/fore_cl1/forgeth2state/kernel:0 (5, 5, 128, 128)=409600, CL/fore_cl1/forgeth2state/bias:0 128, CL/fore_cl1/cellx2state/kernel:0 (5, 5, 64, 128)=204800, CL/fore_cl1/cellh2state/kernel:0 (5, 5, 128, 128)=409600, CL/fore_cl1/cellh2state/bias:0 128, CL/fore_cl1/outputx2state/kernel:0 (5, 5, 64, 128)=204800, CL/fore_cl1/outputh2state/kernel:0 (5, 5, 128, 128)=409600, CL/fore_cl1/outputh2state/bias:0 128, CL/fore_cl2/inputc:0 (1, 16, 16, 64)=16384, CL/fore_cl2/forgetc:0 (1, 16, 16, 64)=16384, CL/fore_cl2/outputc:0 (1, 16, 16, 64)=16384, CL/fore_cl2/inputx2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl2/inputh2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl2/inputh2state/bias:0 64, CL/fore_cl2/forgetx2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl2/forgeth2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl2/forgeth2state/bias:0 64, CL/fore_cl2/cellx2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl2/cellh2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl2/cellh2state/bias:0 64, CL/fore_cl2/outputx2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl2/outputh2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl2/outputh2state/bias:0 64, CL/fore_cl3/inputc:0 (1, 16, 16, 64)=16384, CL/fore_cl3/forgetc:0 (1, 16, 16, 64)=16384, CL/fore_cl3/outputc:0 (1, 16, 16, 64)=16384, CL/fore_cl3/inputx2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl3/inputh2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl3/inputh2state/bias:0 64, CL/fore_cl3/forgetx2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl3/forgeth2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl3/forgeth2state/bias:0 64, CL/fore_cl3/cellx2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl3/cellh2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl3/cellh2state/bias:0 64, CL/fore_cl3/outputx2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl3/outputh2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl3/outputh2state/bias:0 64, CL/finl_predict/kernel:0 (1, 1, 256, 16)=4096, CL/finl_predict/bias:0 16, 
    Total 92 variables, 8,386,576 params
    Sess created.
    2018/7/5 16:59:37 [NS_L3_BN]
    2018/7/5 17:1:41 Step:30 Loss:46551884.000000 (Training Loss:430186956.800)
    Epoch:30 LR:0.000729 (4.125sec/step) Estimated:0:6:31
    2018/7/5 17:1:41 Step:30 Loss:46551884.000000 (Training Loss:430186956.800)
    Epoch:30 LR:0.000729 (4.125sec/step) Estimated:0:6:31
    2018/7/5 17:3:44 Step:60 Loss:45802024.000000 (Training Loss:418819441.600)
    Epoch:60 LR:0.000531 (4.114sec/step) Estimated:0:4:27
    2018/7/5 17:3:44 Step:60 Loss:45802024.000000 (Training Loss:418819441.600)
    Epoch:60 LR:0.000531 (4.114sec/step) Estimated:0:4:27
    2018/7/5 17:5:48 Step:90 Loss:45058024.000000 (Training Loss:419390958.578)
    Epoch:90 LR:0.000349 (4.119sec/step) Estimated:0:2:24
    2018/7/5 17:5:48 Step:90 Loss:45058024.000000 (Training Loss:419390958.578)
    Epoch:90 LR:0.000349 (4.119sec/step) Estimated:0:2:24
    2018/7/5 17:7:51 Step:120 Loss:45136876.000000 (Training Loss:422800548.533)
    Epoch:120 LR:0.000254 (4.114sec/step) Estimated:0:0:20
    2018/7/5 17:7:51 Step:120 Loss:45136876.000000 (Training Loss:422800548.533)
    Epoch:120 LR:0.000254 (4.114sec/step) Estimated:0:0:20
