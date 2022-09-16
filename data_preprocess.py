import pandas as pd
import pickle
import argparse
import numpy as np
import random
import nlpaug.augmenter.word as naw

from transformers import AutoTokenizer

import numpy as np
import random
import os

# Credits https://github.com/varsha33/LCL_loss
np.random.seed(0)
random.seed(0)

def get_one_hot(emo, class_size):

	targets = np.zeros(class_size)
	emo_list = [int(e) for e in emo.split(",")]
	for e in emo_list:
		targets[e-1] = 1
	return list(targets)

def preprocess_data(dataset,tokenizer_type,w_aug,aug_type):
	os.makedirs("preprocessed_data", exist_ok=True)
	if dataset == "ihc_pure":
		class2int = {'not_hate': 0 ,'implicit_hate': 1}

		data_dict = {}
		data_home = "dataset/ihc_pure/"

		for datatype in ["train","valid","test"]:

			datafile = data_home + datatype + ".tsv"
			data = pd.read_csv(datafile, sep='\t')

			label,post = [],[]
			aug_sent1_of_post = []

			for i,one_class in enumerate(data["class"]):
				label.append(class2int[one_class])
				post.append(data["post"][i])
				
			if datatype == "train" and w_aug:
				for i, one_aug_sent in enumerate(data["aug_sent1_of_post"]):
					aug_sent1_of_post.append(one_aug_sent)

				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post =tokenizer.batch_encode_plus(post).input_ids
				tokenized_post_augmented =tokenizer.batch_encode_plus(aug_sent1_of_post).input_ids

				tokenized_combined_prompt = [list(i) for i in zip(tokenized_post,tokenized_post_augmented)]
				combined_prompt = [list(i) for i in zip(post,aug_sent1_of_post)]
				combined_label = [list(i) for i in zip(label,label)]

				processed_data = {}

				processed_data["tokenized_post"] = tokenized_combined_prompt
				processed_data["label"] = combined_label
				processed_data["post"] = combined_prompt

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

			else:
				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post =tokenizer.batch_encode_plus(post).input_ids

				processed_data = {}

				processed_data["tokenized_post"] = tokenized_post
				processed_data["label"] = label
				processed_data["post"] = post

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

		if w_aug:
			with open("./preprocessed_data/ihc_pure_waug_"+aug_type+"_preprocessed_bert.pkl", 'wb') as f:
				pickle.dump(data_dict, f)
			f.close()
		else:
			with open("./preprocessed_data/ihc_pure_preprocessed_bert.pkl", 'wb') as f:
				pickle.dump(data_dict, f)
				f.close()

	# implicit_hate : use implication as a positive sample, not_hate : use sym aug as a positive sample
	elif dataset == "ihc_pure_imp":
		class2int = {'not_hate':0 ,'implicit_hate': 1}

		data_dict = {}
		data_home = "dataset/ihc_pure/"

		for datatype in ["train","valid","test"]:
			datafile = data_home + datatype + ".tsv"
			data = pd.read_csv(datafile, sep='\t') 

			label,post = [],[]
			aug_sent1_of_post = []

			for i,one_class in enumerate(data["class"]):
				label.append(class2int[one_class])
				post.append(data["post"][i])

			if datatype == "train" and w_aug:
				augmented_post = []
				for i,one_class in enumerate(data["class"]):
					if one_class == 'implicit_hate':
						augmented_post.append(data["implied_statement"][i])
					elif one_class == 'not_hate':
						augmented_post.append(data["aug_sent1_of_post"][i])
					else:
						raise NotImplementedError


				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post =tokenizer.batch_encode_plus(post).input_ids
				tokenized_post_augmented =tokenizer.batch_encode_plus(augmented_post).input_ids

				tokenized_combined_prompt = [list(i) for i in zip(tokenized_post,tokenized_post_augmented)]
				combined_prompt = [list(i) for i in zip(post,augmented_post)]
				combined_label = [list(i) for i in zip(label,label)]

				processed_data = {}

				processed_data["tokenized_post"] = tokenized_combined_prompt
				processed_data["label"] = combined_label
				processed_data["post"] = combined_prompt

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

			else:
				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post =tokenizer.batch_encode_plus(post).input_ids

				processed_data = {}
				
				processed_data["tokenized_post"] = tokenized_post
				processed_data["label"] = label
				processed_data["post"] = post

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

		if w_aug:
			with open("./preprocessed_data/ihc_pure_imp_waug_"+aug_type+"_preprocessed_bert.pkl", 'wb') as f:
				pickle.dump(data_dict, f)
		else:
			with open("./preprocessed_data/ihc_pure_imp_preprocessed_bert.pkl", 'wb') as f:
				pickle.dump(data_dict, f)
	
	elif dataset == "dynahate":
		class2int = {'nothate':0 ,'hate': 1}

		data_dict = {}
		data_home = "dataset/DynaHate/"

		for datatype in ["train","dev","test"]:
			datafile = data_home + datatype + ".csv"
			data = pd.read_csv(datafile, sep=',')

			label,post = [],[]

			for i,one_class in enumerate(data["label"]):
				label.append(class2int[one_class])
				post.append(data["text"][i])
			

			print("Tokenizing data")
			tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
			tokenized_post =tokenizer.batch_encode_plus(post).input_ids

			processed_data = {}
			processed_data["tokenized_post"] = tokenized_post
			processed_data["label"] = label
			processed_data["post"] = post

			processed_data = pd.DataFrame.from_dict(processed_data)
			data_dict[datatype] = processed_data

		with open("./preprocessed_data/dynahate_preprocessed_bert.pkl", 'wb') as f:
			pickle.dump(data_dict, f)


	elif dataset == "sbic":
		class2int = {'not_offensive':0 ,'offensive': 1}

		data_dict = {}
		data_home = "dataset/SBIC.v2/"

		for datatype in ["train","dev","test"]:
			datafile = data_home + datatype + ".csv"
			data = pd.read_csv(datafile, sep=',')
			label,post = [],[]

			for i,one_class in enumerate(data["offensiveLABEL"]):
				label.append(class2int[one_class])
				post.append(data["post"][i])


			if datatype == "train" and w_aug:
				augmented_post = []
				for i, one_aug_sent in enumerate(data['aug_sent1_of_post']):
					augmented_post.append(one_aug_sent)

				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post =tokenizer.batch_encode_plus(post).input_ids
				tokenized_post_augmented =tokenizer.batch_encode_plus(augmented_post).input_ids

				tokenized_combined_prompt = [list(i) for i in zip(tokenized_post,tokenized_post_augmented)]
				combined_prompt = [list(i) for i in zip(post,augmented_post)]
				combined_label = [list(i) for i in zip(label,label)]

				processed_data = {}
				processed_data["tokenized_post"] = tokenized_combined_prompt
				processed_data["label"] = combined_label
				processed_data["post"] = combined_prompt

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

			else:

				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post =tokenizer.batch_encode_plus(post).input_ids

				processed_data = {}
				processed_data["tokenized_post"] = tokenized_post
				processed_data["label"] = label
				processed_data["post"] = post

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

		if w_aug:
			with open("./preprocessed_data/sbic_waug_"+aug_type+"_preprocessed_bert.pkl", 'wb') as f:
				pickle.dump(data_dict, f)
			f.close()
		else:
			with open("./preprocessed_data/sbic_preprocessed_bert.pkl", 'wb') as f:
				pickle.dump(data_dict, f)
				f.close()


	elif dataset == "sbic_imp":
		class2int = {'not_offensive': 0 ,'offensive': 1}

		data_dict = {}
		data_home = "dataset/SBIC.v2/"

		for datatype in ["train","dev","test"]:
			datafile = data_home + datatype + ".csv"
			data = pd.read_csv(datafile, sep=',')
			data = data.fillna('')

			label,post = [],[]


			for i,one_class in enumerate(data["offensiveLABEL"]):
				label.append(class2int[one_class])
				post.append(data["post"][i])


			if datatype == "train" and w_aug:
				augmented_post = []
				for i,one_sstype in enumerate(data["selectedStereotype"]):
					if one_sstype != '':
						augmented_post.append(data["selectedStereotype"][i])
					else:
						augmented_post.append(data["aug_sent1_of_post"][i])

				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post =tokenizer.batch_encode_plus(post).input_ids
				tokenized_post_augmented =tokenizer.batch_encode_plus(augmented_post).input_ids

				tokenized_combined_prompt = [list(i) for i in zip(tokenized_post,tokenized_post_augmented)]
				combined_prompt = [list(i) for i in zip(post,augmented_post)]
				combined_label = [list(i) for i in zip(label,label)]

				processed_data = {}
				processed_data["tokenized_post"] = tokenized_combined_prompt
				processed_data["label"] = combined_label
				processed_data["post"] = combined_prompt

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

			else:
				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post =tokenizer.batch_encode_plus(post).input_ids

				processed_data = {}	
				processed_data["tokenized_post"] = tokenized_post
				processed_data["label"] = label
				processed_data["post"] = post

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

		if w_aug:
			with open("./preprocessed_data/sbic_imp_waug_"+aug_type+"_preprocessed_bert.pkl", 'wb') as f:
				pickle.dump(data_dict, f)
		else:
			with open("./preprocessed_data/sbic_imp_preprocessed_bert.pkl", 'wb') as f:
				pickle.dump(data_dict, f)

####################################AugCon+ImpCon#####################################################
	elif dataset == "ihc_pure_imp_double":
		assert w_aug == True, "w_aug should be set to True for double"
		class2int = {'not_hate':0 ,'implicit_hate': 1}

		data_dict = {}
		data_home = "dataset/ihc_pure/"

		for datatype in ["train","valid","test"]:


			datafile = data_home + datatype + ".tsv"
			data = pd.read_csv(datafile, sep='\t')

			label,post = [],[]
			aug_sent1_of_post = []
			aug_sent2_of_post = []

			for i,one_class in enumerate(data["class"]):
				label.append(class2int[one_class])
				post.append(data["post"][i])

			if datatype == "train" and w_aug:
				for i,one_class in enumerate(data["class"]):
					if one_class == 'implicit_hate':
						aug_sent1_of_post.append(data["implied_statement"][i])
					elif one_class == 'not_hate':
						aug_sent1_of_post.append(data["aug_sent1_of_post"][i])
					else:
						raise NotImplementedError
					aug_sent2_of_post.append(data["aug_sent2_of_post"][i])


				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post =tokenizer.batch_encode_plus(post).input_ids
				tokenized_post_augmented_1 =tokenizer.batch_encode_plus(aug_sent1_of_post).input_ids
				tokenized_post_augmented_2 =tokenizer.batch_encode_plus(aug_sent2_of_post).input_ids

				tokenized_combined_prompt = [list(i) for i in zip(tokenized_post,tokenized_post_augmented_1,tokenized_post_augmented_2)]
				combined_prompt = [list(i) for i in zip(post,aug_sent1_of_post,aug_sent2_of_post)]
				combined_label = [list(i) for i in zip(label,label,label)]

				processed_data = {}

				processed_data["tokenized_post"] = tokenized_combined_prompt
				processed_data["label"] = combined_label
				processed_data["post"] = combined_prompt

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

			else:
				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post =tokenizer.batch_encode_plus(post).input_ids

				processed_data = {}
				processed_data["tokenized_post"] = tokenized_post
				processed_data["label"] = label
				processed_data["post"] = post

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

		if w_aug:
			with open("./preprocessed_data/ihc_pure_imp_double_"+aug_type+"_preprocessed_bert.pkl", 'wb') as f:
				pickle.dump(data_dict, f)
		else:
			raise NotImplementedError

	elif dataset == "sbic_imp_double":
		assert w_aug == True, "w_aug should be set to True for double"
		class2int = {'not_offensive':0 ,'offensive': 1}

		data_dict = {}
		data_home = "dataset/SBIC.v2/"

		for datatype in ["train","dev","test"]:
			datafile = data_home + datatype + ".csv"
			data = pd.read_csv(datafile, sep=',')
			data = data.fillna('')

			label,post = [],[]
			aug_sent1_of_post = []
			aug_sent2_of_post = []

			for i,one_class in enumerate(data["offensiveLABEL"]):
				label.append(class2int[one_class])
				post.append(data["post"][i])

			if datatype == "train" and w_aug:
				for i,one_sstype in enumerate(data["selectedStereotype"]):
					if one_sstype != '':
						aug_sent1_of_post.append(data["selectedStereotype"][i])
					else:
						aug_sent1_of_post.append(data["aug_sent1_of_post"][i])
					aug_sent2_of_post.append(data["aug_sent2_of_post"][i])

				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post =tokenizer.batch_encode_plus(post).input_ids
				tokenized_post_augmented_1 =tokenizer.batch_encode_plus(aug_sent1_of_post).input_ids
				tokenized_post_augmented_2 =tokenizer.batch_encode_plus(aug_sent2_of_post).input_ids

				tokenized_combined_prompt = [list(i) for i in zip(tokenized_post,tokenized_post_augmented_1,tokenized_post_augmented_2)]
				combined_prompt = [list(i) for i in zip(post,aug_sent1_of_post,aug_sent2_of_post)]
				combined_label = [list(i) for i in zip(label,label,label)]
				#############################

				processed_data = {}
				processed_data["tokenized_post"] = tokenized_combined_prompt
				processed_data["label"] = combined_label
				processed_data["post"] = combined_prompt

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

			else:
				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post =tokenizer.batch_encode_plus(post).input_ids

				processed_data = {}
				processed_data["tokenized_post"] = tokenized_post
				processed_data["label"] = label
				processed_data["post"] = post

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

		if w_aug:
			with open("./preprocessed_data/sbic_imp_double_"+aug_type+"_preprocessed_bert.pkl", 'wb') as f:
				pickle.dump(data_dict, f)
		else:
			raise NotImplementedError


#########################################################with aug for baseline###################################################################
	elif dataset == "ihc_pure_with_aug":
		class2int = {'not_hate':0 ,'implicit_hate': 1}

		data_dict = {}
		data_home = "dataset/ihc_pure/"

		for datatype in ["train","valid","test"]:

			datafile = data_home + datatype + ".tsv"
			data = pd.read_csv(datafile, sep='\t') # ,names=["ID","class","implied_statement", "post"]

			label,post = [],[]
			aug_sent1_of_post = []

			for i,one_class in enumerate(data["class"]):
				label.append(class2int[one_class])
				post.append(data["post"][i])
				

			if datatype == "train":
				for i, one_aug_sent in enumerate(data["aug_sent1_of_post"]):
					aug_sent1_of_post.append(one_aug_sent)

				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				
				post_with_aug = post + aug_sent1_of_post
				label_with_aug = label + label

				tokenized_post_with_aug =tokenizer.batch_encode_plus(post_with_aug).input_ids

				processed_data = {}
				processed_data["tokenized_post"] = tokenized_post_with_aug
				processed_data["label"] = label_with_aug
				processed_data["post"] = post_with_aug

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

			else:
				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post =tokenizer.batch_encode_plus(post).input_ids

				processed_data = {}
				processed_data["tokenized_post"] = tokenized_post
				processed_data["label"] = label
				processed_data["post"] = post

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

		with open("./preprocessed_data/ihc_pure_with_aug_preprocessed_bert.pkl", 'wb') as f:
			pickle.dump(data_dict, f)
			f.close()

	elif dataset == "sbic_with_aug":
		class2int = {'not_offensive':0 ,'offensive': 1}

		data_dict = {}
		data_home = "dataset/SBIC.v2/" 

		for datatype in ["train","dev","test"]:
			datafile = data_home + datatype + ".csv"
			data = pd.read_csv(datafile, sep=',')
			label,post = [],[]

			for i,one_class in enumerate(data["offensiveLABEL"]):
				label.append(class2int[one_class])
				post.append(data["post"][i])


			if datatype == "train":
				augmented_post = []
				for i, one_aug_sent in enumerate(data['aug_sent1_of_post']):
					augmented_post.append(one_aug_sent)

				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				post_with_aug = post + augmented_post
				label_with_aug = label + label

				tokenized_post_with_aug =tokenizer.batch_encode_plus(post_with_aug).input_ids

				processed_data = {}
				processed_data["tokenized_post"] = tokenized_post_with_aug
				processed_data["label"] = label_with_aug
				processed_data["post"] = post_with_aug

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

			else:
				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post =tokenizer.batch_encode_plus(post).input_ids

				processed_data = {}
				processed_data["tokenized_post"] = tokenized_post
				processed_data["label"] = label
				processed_data["post"] = post

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

		with open("./preprocessed_data/sbic_with_aug_preprocessed_bert.pkl", 'wb') as f:
			pickle.dump(data_dict, f)
			f.close()



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Enter tokenizer type')

	parser.add_argument('-d', default="ihc_pure_imp",type=str,
				   help='Enter dataset')
	parser.add_argument('-t', default="bert-base-uncased",type=str,
				   help='Enter tokenizer type')
	parser.add_argument('--aug_type', default="syn",type=str,
				   help='Enter augmentation type')
	parser.add_argument('--aug', action='store_true')
	args = parser.parse_args()

	preprocess_data(args.d,args.t,w_aug=args.aug,aug_type=args.aug_type)


