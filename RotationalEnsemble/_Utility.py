

class model_params:
    def __init__(self):
        self.model_type = None
        self.target_activation = None
        self.target_neurons = None
        self.performance_metrics = None
        self.loss_function = None
        self.target_time = None
        self.monitor_parameter = None
        self.monitor_condition = None
        #classification specfic
        self.class_split_val = None
        self.num_classes = None
        self.class_weights = None
        
''' NOTE
    ANY CHANGE OF METRICS ARRAY WILL CAUSE ERROR IN 
    performance_printout.py. Either implement array methodology
    in printout of history in performance_printout file, or 
    manually change function in performance_prinout to match
    metrics used.
    NOTE
    Did not implement array methodology as I don't immediately
    foresee changing the metrics used..
'''
#this function initiates a parameters class for implementation
#of fast changable variables for all variables that differ in
#           REGRESSION, BINARY, AND MULTICLASS CLASSIFICATION
def get_model_params(m_type, target_time, c_split_val, c_class_cnt):
    #initialize class and pull in model type
    params = model_params()
    params.model_type = m_type
    params.target_time = target_time
    match(params.model_type):
        case 'Regression':#################################
            params.target_activation = 'linear'
            params.target_neurons = 1
            params.performance_metrics = \
                ['R2Score','root_mean_squared_error']
            params.loss_function = 'loss'
            params.monitor_parameter = 'val_loss'
            params.monitor_condition = 'min'
        case 'Classification':#############################
            params.num_classes = c_class_cnt
            params.class_split_val = c_split_val
            params.target_activation = 'sigmoid' if \
                (c_class_cnt == 2) else 'softmax'
            params.loss_function = \
                'binary_crossentropy' if (c_class_cnt == 2)\
                else 'categorical_crossentropy'
            params.target_neurons = 1 if (c_class_cnt == 2)\
                else c_class_cnt
            params.performance_metrics = \
                ['precision','recall','accuracy']
            params.monitor_parameter = 'val_accuracy'
            params.monitor_condition = 'max'
        case _:
            raise ValueError(f"Invalid m_type (model type) \
                             {params.model_type}.")
    return params