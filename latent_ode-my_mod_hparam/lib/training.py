
import lib.utils
from lib.utils import compute_loss_all_batches
from ray.tune import track
import pdb

def construct_and_train_model(config):
	# Create ODE-GRU model

	pdb.set_trace()

	#get arguments
	args 


	n_ode_gru_dims = args.latents
	method = args.ode_method
	#print(args.ode_method)
	
	if args.poisson:
		print("Poisson process likelihood not implemented for ODE-RNN: ignoring --poisson")

	if args.extrap:
		raise Exception("Extrapolation for ODE-RNN not implemented")

	ode_func_net = utils.create_net(n_ode_gru_dims, n_ode_gru_dims, 
		n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.Tanh)

	rec_ode_func = ODEFunc(
		input_dim = input_dim, 
		latent_dim = n_ode_gru_dims,
		ode_func_net = ode_func_net,
		device = device).to(device)

	z0_diffeq_solver = DiffeqSolver(input_dim, rec_ode_func, method, args.latents, 
		odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

	model = ODE_RNN(input_dim, n_ode_gru_dims, device = device, 
		z0_diffeq_solver = z0_diffeq_solver, n_gru_units = args.gru_units,
		concat_mask = True, obsrv_std = obsrv_std,
		use_binary_classif = args.classif,
		classif_per_tp = classif_per_tp,
		n_labels = n_labels,
		train_classif_w_reconstr = (args.dataset == "physionet")
		).to(device)



	train_it(model,
		data_obj,
		args,
		file_name,
		optimizer,
		experimentID,
		trainwriter,
		validationwriter)

def train_it(
		model,
		data_obj,
		args,
		file_name,
		optimizer,
		experimentID,
		trainwriter,
		validationwriter):

	"""
	parameters:
		model,
		data_obj,
		args,
		file_name,
		optimizer,
		experimentID,
		trainwriter,
		validationwriter,
	"""
	
	log_path = "logs/" + file_name + "_" + str(experimentID) + ".log"
	if not os.path.exists("logs/"):
		utils.makedirs("logs/")
	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
	logger.info(input_command)

	optimizer = optim.Adamax(model.parameters(), lr=args.lr)

	num_batches = data_obj["n_train_batches"]

	
	for itr in tqdm(range(1, num_batches * (args.niters) + 1)):
		optimizer.zero_grad()
		utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)

		wait_until_kl_inc = 10
		if itr // num_batches < wait_until_kl_inc:
			kl_coef = 0.
		else:
			kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))
		
		batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
		train_res = model.compute_all_losses(batch_dict, n_traj_samples = 3, kl_coef = kl_coef)
		train_res["loss"].backward()
		optimizer.step()

		n_iters_to_viz = 0.2
		if (itr % round(n_iters_to_viz * num_batches + 0.499999)== 0) and (itr!=0):
			
			with torch.no_grad():

				test_res = compute_loss_all_batches(model, 
					data_obj["test_dataloader"], args,
					n_batches = data_obj["n_test_batches"],
					experimentID = experimentID,
					device = device,
					n_traj_samples = 3, kl_coef = kl_coef)

				message = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
					itr//num_batches, 
					test_res["loss"].detach(), test_res["likelihood"].detach(), 
					test_res["kl_first_p"], test_res["std_first_p"])
		 	
				logger.info("Experiment " + str(experimentID))
				logger.info(message)
				logger.info("KL coef: {}".format(kl_coef))
				logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
				logger.info("Train CE loss (one batch): {}".format(train_res["ce_loss"].detach()))
				
				# write training numbers
				if "accuracy" in train_res:
					logger.info("Classification accuracy (TRAIN): {:.4f}".format(train_res["accuracy"]))
					trainwriter.add_scalar('Classification_accuracy', train_res["accuracy"], itr*args.batch_size)
				
				if "loss" in train_res:
					trainwriter.add_scalar('loss', train_res["loss"].detach(), itr*args.batch_size)
				
				if "ce_loss" in train_res:
					trainwriter.add_scalar('CE_loss', train_res["ce_loss"].detach(), itr*args.batch_size)
				
				if "mse" in train_res:
					trainwriter.add_scalar('MSE', train_res["mse"], itr*args.batch_size)
				
				if "pois_likelihood" in train_res:
					trainwriter.add_scalar('Poisson_likelihood', train_res["pois_likelihood"], itr*args.batch_size)
				
				#write test numbers
				if "auc" in test_res:
					logger.info("Classification AUC (TEST): {:.4f}".format(test_res["auc"]))
					validationwriter.add_scalar('Classification_AUC', test_res["auc"], itr*args.batch_size)
					
				if "mse" in test_res:
					logger.info("Test MSE: {:.4f}".format(test_res["mse"]))
					validationwriter.add_scalar('MSE', test_res["mse"], itr*args.batch_size)
					
				if "accuracy" in test_res:
					logger.info("Classification accuracy (TEST): {:.4f}".format(test_res["accuracy"]))
					validationwriter.add_scalar('Classification_accuracy', test_res["accuracy"], itr*args.batch_size)

				if "pois_likelihood" in test_res:
					logger.info("Poisson likelihood: {}".format(test_res["pois_likelihood"]))
					validationwriter.add_scalar('Poisson_likelihood', test_res["pois_likelihood"], itr*args.batch_size)
				
				if "loss" in train_res:
					validationwriter.add_scalar('loss', test_res["loss"].detach(), itr*args.batch_size)
				
				if "ce_loss" in test_res:
					logger.info("CE loss: {}".format(test_res["ce_loss"]))
					validationwriter.add_scalar('CE_loss', test_res["ce_loss"], itr*args.batch_size)
	
				logger.info("-----------------------------------------------------------------------------------")

				# ray tune elements:
				"""
				reporter(
					timesteps_total=itr*args.batch_size,
					mean_accuracy=test_res["accuracy"]  )
				"""
				track.log(mean_accuracy=test_res["accuracy"])


				torch.save({
					'args': args,
					'state_dict': model.state_dict(),
				}, ckpt_path)

				if test_res["accuracy"] > best_test_acc:
					best_test_acc = test_res["accuracy"]
					torch.save({
						'args': args,
						'state_dict': model.state_dict(),
					}, top_ckpt_path)
			
			# Plotting
			if args.viz:
				with torch.no_grad():
					test_dict = utils.get_next_batch(data_obj["test_dataloader"])

					print("plotting....")
					if isinstance(model, LatentODE) and (args.dataset == "periodic"): #and not args.classic_rnn and not args.ode_rnn:
						plot_id = itr // num_batches // n_iters_to_viz
						viz.draw_all_plots_one_dim(test_dict, model, 
							plot_name = file_name + "_" + str(experimentID) + "_{:03d}".format(plot_id) + ".png",
						 	experimentID = experimentID, save=True)
						plt.pause(0.01)
						