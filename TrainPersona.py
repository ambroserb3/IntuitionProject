from simpletransformers.conv_ai import ConvAIModel


train_args = {
    "overwrite_output_dir": True,
    "reprocess_input_data": True
}


def TrainEmotionPersona():
	model = ConvAIModel("gpt", "gpt_personachat_cache", use_cuda=False, args=train_args)
	model.train_model("Data/EmotionPersona.json")
	model.eval_model()
	# model.interact()

def TrainExperentialPersona():
    model = ConvAIModel("gpt", "gpt_personachat_cache", use_cuda=False, args=train_args)
	model.train_model("Data/EmotionPersona.json")
	model.eval_model()
	# model.interact()

TrainEmotionPersona()
TrainExperentialPersona()







def bulk_train_and_vectorize(model_type, train_sizes):
	model = getattr(basic_models, model_type)()
	print("Opening pickle files")
	full_train = pickle.load(open(TRAIN_DATA, 'rb'))
	num_train = (full_train.shape)[0]
	test = pickle.load(open(TEST_DATA, 'rb'))
	np.random.seed(726)

	for train_size in train_sizes:
		train_indices = np.random.choice(num_train, train_size, replace=False)
		train = full_train.iloc[train_indices]
		print("Handling text for training size " + str(train_size))
		x_train, y_train, x_test, y_test = handle_text(train, test)

		print("Saving test data as a .npy file")
		test_file = data_file(train_size)
		np.save(test_file, x_test)
		np.save('fb_test_y', y_test)

		print("Training model now")
		model_filename = model_file(model_type, train_size)
		train_model(model, x_train, y_train, model_filename)
