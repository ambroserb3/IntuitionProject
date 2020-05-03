from simpletransformers.conv_ai import ConvAIModel


train_args={
    'reprocess_input_data': True,
    'overwrite_output_dir': False,
    'evaluate_during_training': True,
    'logging_steps': 5,
    "num_candidates": 1,
    "personality_permutations": 1,
    "max_history": 2,
    "lm_coef": 2.0,
    "mc_coef": 1.0,
    "no_sample": False,
    "max_length": 20000,
    "min_length": 1,
    "temperature": 0.7,
    "top_k": 0,
    "top_p": 0.9,
}


def TrainEmotionPersona():
	model = ConvAIModel("gpt2", "gpt2", use_cuda=False, args=train_args)
	model.train_model("Data/EmotionPersona.json")
	model.eval_model()
	# model.interact()
 
def TrainExperentialPersona():
    model = ConvAIModel("gpt2", "gpt2", use_cuda=False, args=train_args)
    model.train_model("Data/ExperentialPersona.json")
    model.eval_model()
	# model.interact()

TrainEmotionPersona()
TrainExperentialPersona()
