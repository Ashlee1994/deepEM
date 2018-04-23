class Train_Args():
    is_training                 =       True
    output_cnn_name             =       "data/19S.cnn"
    boxsize                     =       160
    dim_x                       =       1855
    dim_y                       =       1919
    name_length                 =       5
    name_prefix                 =       "image_"
    mic_path                    =       "data/19Sdata/mic/"
# model-4 mean square std and epochs = 30 batchsize = 100
    model_save_path             =       "./data/19Sdata/model-4"
    # model-3 changed the loss function
#    model_save_path             =       "./data/19Sdata/model-3"
#    model_save_path             =       "./data/19Sdata/model-2"
#    model_save_path             =       "./model"
    positive1_box_path          =       "data/19Sdata/positive/"
    negative1_box_path          =       "data/19Sdata/negative/"
    positive1_mic_start_num     =       30001
    positive1_mic_end_num       =       30050
    negative1_mic_start_num     =       30001
    negative1_mic_end_num       =       30050
    do_train_again              =       1
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
    num_p_test                  =       150
    num_n_test                  =       150

    FL_feature_map              =       6
    FL_kernelsize               =       20

    SL_poolingsize              =       3

    TL_feature_map              =       12
    TL_kernelsize               =       10

    FOL_poolingsize             =       2

    FIL_feature_map             =       12
    FIL_kernelsize              =       4

    SIL_poolingsize             =       2

    alpha                       =       0.01
    batch_size                  =       100
    num_epochs                  =       30
    decay_rate                  =       0.96
    decay_step                  =       200
    sigma                       =       0.0001
    grad_step                   =       0
    keep_prob                   =       1.0


class Predict_Args():
    is_training                 =       False
    cnn_file                    =       "data/19S.cnn"
    data_path                   =       "data/19Sdata/mic/"
    result_path                 =       "./data/19Sdata/result/"
    #model_save_path             =       "./data/19Sdata/model"
    model_save_path             =       "./model"
    #model_save_path             =       "data/19Sdata/model/"
    boxsize                     =       160
    start_mic_num               =       30245
    end_mic_num                 =       30245
    dim_x                       =       1855
    dim_y                       =       1919
    scan_step                   =       20
    range1                      =       70
    range2                      =       40
    min_std                     =       3.45
    max_std                     =       3.6
    name_length                 =       5
    name_prefix                 =       "image_"
    rotation_angel              =       90
    rotation_n                  =       4

    FL_feature_map              =       6
    FL_kernelsize               =       20

    SL_poolingsize              =       3

    TL_feature_map              =       12
    TL_kernelsize               =       10

    FOL_poolingsize             =       2

    FIL_feature_map             =       12
    FIL_kernelsize              =       4

    SIL_poolingsize             =       2

    alpha                       =       0.01
    batch_size                  =       50
    num_epochs                  =       20
    decay_rate                  =       0.96
    decay_step                  =       200
    sigma                       =       0.0001
    grad_step                   =       0
    keep_prob                   =       1.0

