class Train_Args():
    is_training                 =       True
    output_cnn_name             =       "data/19S.cnn"
    boxsize                     =       160
    dim_x                       =       1855
    dim_y                       =       1919
    name_length                 =       5
    name_prefix                 =       "image_"
    mic_path                    =       "data/19Sdata/mrc_file/"
    model_save_path             =       "./data/19Sdata/model/32-64-128/"
    positive1_box_path          =       "data/19Sdata/positive/"
    negative1_box_path          =       "data/19Sdata/negative/"
    args_filename               =       "args_19S.py"

    positive1_mic_start_num     =       30001
    positive1_mic_end_num       =       30050
    negative1_mic_start_num     =       30001
    negative1_mic_end_num       =       30050
    num_positive1               =       800
    num_negative1               =       800

    do_train_again              =       True
    num_positive2               =       800
    num_negative2               =       800
    positive2_box_path          =       "data/19Sdata/sel_positive/"
    negative2_box_path          =       "data/19Sdata/sel_negative/"
    positive2_mic_start_num     =       30051
    positive2_mic_end_num       =       30090
    negative2_mic_start_num     =       30051
    negative2_mic_end_num       =       30100

    rotation_angel              =       90
    rotation_n                  =       1
    num_p_test                  =       150
    num_n_test                  =       150

    FL_feature_map              =       64
    FL_kernelsize               =       10

    SL_poolingsize              =       3

    TL_feature_map              =       128
    TL_kernelsize               =       10

    FOL_poolingsize             =       2

    FIL_feature_map             =       128
    FIL_kernelsize              =       5

    SIL_poolingsize             =       2

    regularization              =       False
    reg_rate                    =       0.005
    dropout                     =       True
    dropout_rate                =       0.5
    learning_rate               =       0.5
    batch_size                  =       50
    num_epochs                  =       150
    decay_rate                  =       0.96
    decay_step                  =       200
    sigma                       =       0.0001
    grad_step                   =       0
    keep_prob                   =       1.0


class Predict_Args():
    is_training                 =       False
    data_path                   =       "data/19Sdata/mrc_file/"
    result_path                 =       "./data/19Sdata/result/"
    model_save_path             =       "./data/19Sdata/model/0.88-32-64-64/"
    boxsize                     =       160     
    start_mic_num               =       30188 
    end_mic_num                 =       30190
    dim_x                       =       1855
    dim_y                       =       1919
    scan_step                   =       10
    accuracy                    =       0.998
    threhold                    =       0.7
    name_length                 =       5
    name_prefix                 =       "image_"

    FL_feature_map              =       64
    FL_kernelsize               =       10

    SL_poolingsize              =       3

    TL_feature_map              =       128
    TL_kernelsize               =       10

    FOL_poolingsize             =       2

    FIL_feature_map             =       128
    FIL_kernelsize              =       4

    SIL_poolingsize             =       2

    batch_size                  =       100