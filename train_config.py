# dataset = ["ihc_pure"]
dataset = ["ihc_pure_imp"]
# dataset = ["sbic"]
# dataset = ["sbic_imp"]
# dataset = ["dynahate"]

tuning_param  = ["lambda_loss", "main_learning_rate","train_batch_size","eval_batch_size","nepoch","temperature","SEED","dataset", "decay"] ## list of possible paramters to be tuned
lambda_loss = [0.25]
temperature = [0.3]
train_batch_size = [8]
eval_batch_size = [8]
decay = [0.0] # default value of AdamW
main_learning_rate = [2e-5]

hidden_size = 768
nepoch = [6]
run_name = "best"
loss_type = "impcon" # only for saving file name
model_type = "bert-base-uncased"

SEED = [0]
w_aug = True
w_double = False
w_separate = False
w_sup = False

save = True
param = {"temperature":temperature,"run_name":run_name,"dataset":dataset,"main_learning_rate":main_learning_rate,"train_batch_size":train_batch_size,"eval_batch_size":eval_batch_size,"hidden_size":hidden_size,"nepoch":nepoch,"dataset":dataset,"lambda_loss":lambda_loss,"loss_type":loss_type,"decay":decay,"SEED":SEED,"model_type":model_type,"w_aug":w_aug, "w_sup":w_sup, "save":save,"w_double":w_double, "w_separate":w_separate}