class Train_Args():
    is_training                 =       True
    boxsize                     =       272
    resize                      =       224
    dim_x                       =       2048
    dim_y                       =       2048
    name_length                 =       2
    name_prefix                 =       ""
    mic_path                    =       "data/KLHdata/mic/"
    model_save_path             =       "./data/KLHdata/model/model-9-272/"
    positive1_box_path          =       "data/KLHdata/positive/"
    negative1_box_path          =       "data/KLHdata/negative/"
    model_filename              =       "vgg19.py"
    positive1_mic_start_num     =       1
    positive1_mic_end_num       =       50
    negative1_mic_start_num     =       1
    negative1_mic_end_num       =       50
    do_train_again              =       False
    num_positive1               =       1000 #800
    num_negative1               =       1000 #800
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

    regularization              =       True
    reg_rate                    =       0.0000001
    dropout                     =       True
    dropout_rate                =       0.5

    learning_rate               =       0.001

    batch_size                  =       50

    num_epochs                  =       100

    decay_rate                  =       0.96
    decay_step                  =       200
    sigma                       =       0.0001
    grad_step                   =       0
    keep_prob                   =       1.0

class Predict_Args():
    is_training                 =       False
    data_path                   =       "data/KLHdata/mic/"
    result_path                 =       "./data/KLHdata/result/"
    model_save_path             =       "./data/KLHdata/model/model-9-272/"
    boxsize                     =       272
    resize                      =       224
    start_mic_num               =       75
    end_mic_num                 =       81
    dim_x                       =       2048
    dim_y                       =       2048
    scan_step                   =       20
    accuracy                    =       0.96
    threhold                    =       0.65
    name_length                 =       2
    name_prefix                 =       ""

    batch_size                  =       50