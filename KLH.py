class Train_Args():
    is_training                 =       True
    boxsize                     =       272
    dim_x                       =       2048
    dim_y                       =       2048
    name_length                 =       0
    name_prefix                 =       ""
    mic_path                    =       "data/KLHdata/mic/"
# model-4 mean square std and epochs = 30 batchsize = 100
    model_save_path             =       "./data/KLHdata/model-0.2-30"
    # model-3 changed the loss function
#    model_save_path             =       "./data/19Sdata/model-3"
#    model_save_path             =       "./data/19Sdata/model-2"
#    model_save_path             =       "./model"
    positive1_box_path          =       "data/KLHdata/positive/"
    negative1_box_path          =       "data/KLHdata/negative/"
    positive1_mic_start_num     =       1
    positive1_mic_end_num       =       100
    negative1_mic_start_num     =       1
    negative1_mic_end_num       =       100
    do_train_again              =       0
    num_positive1               =       800
    num_negative1               =       800
    num_positive2               =       800
    num_negative2               =       800

    positive2_box_path          =       "data/19Sdata/sel_positive/"
    negative2_box_path          =       "data/19Sdata/sel_negative/"
    positive2_mic_start_num     =       30051
    positive2_mic_end_num       =       30090
    negative2_mic_start_num     =       30051
    negative2_mic_end_num       =       30100

    rotation_angel              =       90
    rotation_n                  =       4
    num_p_test                  =       300
    num_n_test                  =       300

    FL_feature_map              =       6
    FL_kernelsize               =       51

    SL_poolingsize              =       3

    TL_feature_map              =       12
    TL_kernelsize               =       21

    FOL_poolingsize             =       2

    FIL_feature_map             =       12
    FIL_kernelsize              =       10

    SIL_poolingsize             =       2

    alpha                       =       0.2
    batch_size                  =       500
    num_epochs                  =       30
    decay_rate                  =       0.96
    decay_step                  =       200
    sigma                       =       0.0001
    grad_step                   =       0
    keep_prob                   =       1.0


class Predict_Args():
    is_training                 =       False
    data_path                   =       "data/KLHdata/mic/"
    result_path                 =       "./data/KLHdata/result/"
    #model_save_path             =       "./data/19Sdata/model"
    model_save_path             =       "./data/KLHdata/model"
    #model_save_path             =       "data/19Sdata/model/"
    boxsize                     =       272
    start_mic_num               =       78
    end_mic_num                 =       78
    dim_x                       =       2048
    dim_y                       =       2048
    scan_step                   =       20
    range1                      =       70
    range2                      =       40
    min_std                     =       22
    max_std                     =       34
    name_length                 =       0
    name_prefix                 =       ""
    rotation_angel              =       90
    rotation_n                  =       4

    FL_feature_map              =       6
    FL_kernelsize               =       51

    SL_poolingsize              =       3

    TL_feature_map              =       12
    TL_kernelsize               =       21

    FOL_poolingsize             =       2

    FIL_feature_map             =       12
    FIL_kernelsize              =       10

    SIL_poolingsize             =       2

    alpha                       =       0.01
    batch_size                  =       50
    num_epochs                  =       20
    decay_rate                  =       0.96
    decay_step                  =       200
    sigma                       =       0.0001
    grad_step                   =       0
    keep_prob                   =       1.0

