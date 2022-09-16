# dataset = ["ihc_pure"]
dataset = ["ihc_pure_imp"]
# dataset = ["sbic"]
# dataset = ["sbic_imp"]
# dataset = ["dynahate"]

tuning_param  = ["lambda_loss", "main_learning_rate", "train_batch_size", "eval_batch_size", "nepoch", "temperature", "SEED", "dataset", "load_dir", "decay"] ## list of possible paramters to be tuned
lambda_loss = [0.75]
temperature = [0.3]
train_batch_size = [8]
eval_batch_size = [8]
decay = [0.0] # default value of AdamW
main_learning_rate = [2e-5]

hidden_size = 768
nepoch = [6]
run_name = "run0"
loss_type = "w_aug_no_sup" # only for saving file name
model_type = "bert-base-uncased"

SEED = [0]
w_aug = True
w_double = False
w_separate = False
w_sup = False

debug = False

####################################################(CROSS DATASET EVALUATION)#########################################################
# dataset = ["ihc_pure", "dynahate", "sbic"]
# when load = True, save should be False since it is not implemented to do both at the same time. vice versa.
cross_eval = False
save = False
load = False

# load_dir should be None when not loading
load_dir = [None]

param = {"temperature":temperature,"run_name":run_name,"dataset":dataset,"main_learning_rate":main_learning_rate,"train_batch_size":train_batch_size,"eval_batch_size":eval_batch_size,"hidden_size":hidden_size,"nepoch":nepoch,"dataset":dataset,"lambda_loss":lambda_loss,"loss_type":loss_type,"decay":decay,"SEED":SEED,"model_type":model_type,"w_aug":w_aug, "w_sup":w_sup, "save":save, "load":load, "load_dir":load_dir, "cross_eval":cross_eval, "debug":debug, "w_double":w_double, "w_separate":w_separate}


