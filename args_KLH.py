class Train_Args():
    is_training                 =       True
    boxsize                     =       272
    dim_x                       =       2048
    dim_y                       =       2048
    name_length                 =       2
    name_prefix                 =       ""
    mic_path                    =       "data/KLHdata/mic/"
    model_save_path             =       "./data/KLHdata/model/model-4/"
    positive1_box_path          =       "data/KLHdata/positive/"
    negative1_box_path          =       "data/KLHdata/negative/"
    args_filename               =       "args_KLH.py"
    positive1_mic_start_num     =       1
    positive1_mic_end_num       =       50
    negative1_mic_start_num     =       1
    negative1_mic_end_num       =       40
    do_train_again              =       False
    num_positive1               =       1100 #800
    num_negative1               =       1100 #800
    num_positive2               =       800
    num_negative2               =       800

    positive2_box_path          =       "data/19Sdata/sel_positive/"
    negative2_box_path          =       "data/19Sdata/sel_negative/"
    positive2_mic_start_num     =       1
    positive2_mic_end_num       =       50
    negative2_mic_start_num     =       1
    negative2_mic_end_num       =       50

    rotation_angel              =       90
    rotation_n                  =       4   
    num_p_test                  =       117 
    num_n_test                  =       83  

    FL_feature_map              =       64
    FL_kernelsize               =       7

    SL_poolingsize              =       3

    TL_feature_map              =       128
    TL_kernelsize               =       7

    FOL_poolingsize             =       2

    FIL_feature_map             =       128
    FIL_kernelsize              =       5

    SIL_poolingsize             =       2

    regularization              =       False
    reg_rate                    =       0.005
    dropout                     =       True
    dropout_rate                =       0.5
    learning_rate               =       0.05

    batch_size                  =       100
    num_epochs                  =       50
    decay_rate                  =       0.96
    decay_step                  =       200
    sigma                       =       0.0001
    grad_step                   =       0
    keep_prob                   =       1.0

class Predict_Args():
    is_training                 =       False
    data_path                   =       "data/KLHdata/mic/"
    result_path                 =       "./data/KLHdata/result/"
    model_save_path             =       "./data/KLHdata/model/model-1-0.94/"
    boxsize                     =       272
    start_mic_num               =       70
    end_mic_num                 =       70
    dim_x                       =       2048
    dim_y                       =       2048
    scan_step                   =       20
    accuracy                    =       0.99999
    threhold                    =       0.7
    name_length                 =       2
    name_prefix                 =       ""
    rotation_angel              =       90
    rotation_n                  =       4

    FL_feature_map              =       32 
    FL_kernelsize               =       5 

    SL_poolingsize              =       3

    TL_feature_map              =       64 
    TL_kernelsize               =       3 

    FOL_poolingsize             =       3

    FIL_feature_map             =       128
    FIL_kernelsize              =       5

    SIL_poolingsize             =       2

    batch_size                  =       50