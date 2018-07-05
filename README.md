# Neural Scene

Target: 10 frames as a sequence of length 10, to predict 10 frames with these input 10 frames
A,B,C --> Network --> D,E,F

Algorith: generate one frame for each input frame with shared convLSTM layers

### training
    python experiment_mnist.py 
    

Ref:
1. [Convolutional LSTM network: A machine learning approach for precipitation nowcasting.](https://arxiv.org/abs/1506.04214)
2. [PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma in Spatiotemporal Predictive Learning](https://arxiv.org/abs/1804.06300)

   

    CL/enco_cl1/inputc:0 (1, 16, 16, 128)=32768, CL/enco_cl1/forgetc:0 (1, 16, 16, 128)=32768, CL/enco_cl1/outputc:0 (1, 16, 16, 128)=32768, CL/enco_cl1/inputx2state/kernel:0 (5, 5, 16, 128)=51200, CL/enco_cl1/inputh2state/kernel:0 (5, 5, 128, 128)=409600, CL/enco_cl1/inputh2state/bias:0 128, CL/enco_cl1/forgetx2state/kernel:0 (5, 5, 16, 128)=51200, CL/enco_cl1/forgeth2state/kernel:0 (5, 5, 128, 128)=409600, CL/enco_cl1/forgeth2state/bias:0 128, CL/enco_cl1/cellx2state/kernel:0 (5, 5, 16, 128)=51200, CL/enco_cl1/cellh2state/kernel:0 (5, 5, 128, 128)=409600, CL/enco_cl1/cellh2state/bias:0 128, CL/enco_cl1/outputx2state/kernel:0 (5, 5, 16, 128)=51200, CL/enco_cl1/outputh2state/kernel:0 (5, 5, 128, 128)=409600, CL/enco_cl1/outputh2state/bias:0 128, CL/enco_cl2/inputc:0 (1, 16, 16, 64)=16384, CL/enco_cl2/forgetc:0 (1, 16, 16, 64)=16384, CL/enco_cl2/outputc:0 (1, 16, 16, 64)=16384, CL/enco_cl2/inputx2state/kernel:0 (5, 5, 128, 64)=204800, CL/enco_cl2/inputh2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl2/inputh2state/bias:0 64, CL/enco_cl2/forgetx2state/kernel:0 (5, 5, 128, 64)=204800, CL/enco_cl2/forgeth2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl2/forgeth2state/bias:0 64, CL/enco_cl2/cellx2state/kernel:0 (5, 5, 128, 64)=204800, CL/enco_cl2/cellh2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl2/cellh2state/bias:0 64, CL/enco_cl2/outputx2state/kernel:0 (5, 5, 128, 64)=204800, CL/enco_cl2/outputh2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl2/outputh2state/bias:0 64, CL/enco_cl3/inputc:0 (1, 16, 16, 64)=16384, CL/enco_cl3/forgetc:0 (1, 16, 16, 64)=16384, CL/enco_cl3/outputc:0 (1, 16, 16, 64)=16384, CL/enco_cl3/inputx2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl3/inputh2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl3/inputh2state/bias:0 64, CL/enco_cl3/forgetx2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl3/forgeth2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl3/forgeth2state/bias:0 64, CL/enco_cl3/cellx2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl3/cellh2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl3/cellh2state/bias:0 64, CL/enco_cl3/outputx2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl3/outputh2state/kernel:0 (5, 5, 64, 64)=102400, CL/enco_cl3/outputh2state/bias:0 64, CL/fore_cl1/inputc:0 (1, 16, 16, 128)=32768, CL/fore_cl1/forgetc:0 (1, 16, 16, 128)=32768, CL/fore_cl1/outputc:0 (1, 16, 16, 128)=32768, CL/fore_cl1/inputx2state/kernel:0 (5, 5, 64, 128)=204800, CL/fore_cl1/inputh2state/kernel:0 (5, 5, 128, 128)=409600, CL/fore_cl1/inputh2state/bias:0 128, CL/fore_cl1/forgetx2state/kernel:0 (5, 5, 64, 128)=204800, CL/fore_cl1/forgeth2state/kernel:0 (5, 5, 128, 128)=409600, CL/fore_cl1/forgeth2state/bias:0 128, CL/fore_cl1/cellx2state/kernel:0 (5, 5, 64, 128)=204800, CL/fore_cl1/cellh2state/kernel:0 (5, 5, 128, 128)=409600, CL/fore_cl1/cellh2state/bias:0 128, CL/fore_cl1/outputx2state/kernel:0 (5, 5, 64, 128)=204800, CL/fore_cl1/outputh2state/kernel:0 (5, 5, 128, 128)=409600, CL/fore_cl1/outputh2state/bias:0 128, CL/fore_cl2/inputc:0 (1, 16, 16, 64)=16384, CL/fore_cl2/forgetc:0 (1, 16, 16, 64)=16384, CL/fore_cl2/outputc:0 (1, 16, 16, 64)=16384, CL/fore_cl2/inputx2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl2/inputh2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl2/inputh2state/bias:0 64, CL/fore_cl2/forgetx2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl2/forgeth2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl2/forgeth2state/bias:0 64, CL/fore_cl2/cellx2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl2/cellh2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl2/cellh2state/bias:0 64, CL/fore_cl2/outputx2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl2/outputh2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl2/outputh2state/bias:0 64, CL/fore_cl3/inputc:0 (1, 16, 16, 64)=16384, CL/fore_cl3/forgetc:0 (1, 16, 16, 64)=16384, CL/fore_cl3/outputc:0 (1, 16, 16, 64)=16384, CL/fore_cl3/inputx2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl3/inputh2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl3/inputh2state/bias:0 64, CL/fore_cl3/forgetx2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl3/forgeth2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl3/forgeth2state/bias:0 64, CL/fore_cl3/cellx2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl3/cellh2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl3/cellh2state/bias:0 64, CL/fore_cl3/outputx2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl3/outputh2state/kernel:0 (5, 5, 64, 64)=102400, CL/fore_cl3/outputh2state/bias:0 64, CL/finl_predict/kernel:0 (1, 1, 256, 16)=4096, CL/finl_predict/bias:0 16, 
    Total 92 variables, 8,386,576 params
    Sess created.
    2018/7/5 23:8:30 [NS_L3]
    Model restored [ model/NS_L3.ckpt ].
    2018/7/5 23:10:8 Step:23 Loss:44578896.000000 (Training Loss:55921526.087)
    2018/7/5 23:11:41 Step:46 Loss:41871624.000000 (Training Loss:51392044.609)

    === [frame20_seqs100.npz] Loss:42907704.000000 ===
    2018/7/5 23:13:15 Step:69 Loss:44277728.000000 (Training Loss:49965093.565)
    2018/7/5 23:14:48 Step:92 Loss:44377208.000000 (Training Loss:49102516.304)
    2018/7/5 23:16:22 Step:115 Loss:43108304.000000 (Training Loss:50123001.670)

    === [frame20_seqs100.npz] Loss:43161920.000000 ===
    2018/7/5 23:17:56 Step:138 Loss:44527944.000000 (Training Loss:49364110.058)
    2018/7/5 23:19:30 Step:161 Loss:44547020.000000 (Training Loss:49261986.360)

    === [frame20_seqs100.npz] Loss:41504084.000000 ===
    2018/7/5 23:21:5 Step:184 Loss:41624264.000000 (Training Loss:48901767.783)
    2018/7/5 23:22:41 Step:207 Loss:42997324.000000 (Training Loss:48333591.420)
    2018/7/5 23:24:19 Step:230 Loss:42479280.000000 (Training Loss:47697314.539)

    === [frame20_seqs100.npz] Loss:43351696.000000 ===
    2018/7/5 23:25:56 Step:253 Loss:41668776.000000 (Training Loss:47650659.700)
    2018/7/5 23:27:39 Step:276 Loss:44274172.000000 (Training Loss:47666247.014)
    2018/7/5 23:29:19 Step:299 Loss:41910532.000000 (Training Loss:47446479.960)

    === [frame20_seqs100.npz] Loss:42278360.000000 ===
    2018/7/5 23:30:54 Step:322 Loss:43754756.000000 (Training Loss:47387477.801)
    2018/7/5 23:32:28 Step:345 Loss:44537496.000000 (Training Loss:47278869.959)

    === [frame20_seqs100.npz] Loss:42185308.000000 ===
    2018/7/5 23:34:3 Step:368 Loss:40859800.000000 (Training Loss:47088006.141)
    2018/7/5 23:35:38 Step:391 Loss:41632872.000000 (Training Loss:47024555.120)
    2018/7/5 23:37:12 Step:414 Loss:38965572.000000 (Training Loss:46856665.372)

    === [frame20_seqs100.npz] Loss:41910004.000000 ===
    2018/7/5 23:38:49 Step:437 Loss:41328736.000000 (Training Loss:46702892.439)
