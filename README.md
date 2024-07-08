# Scar_enhancement
Myocardial Scar Enhancement in LGE Cardiac MRI using Localized Diffusion


The example train, test, and validation data are in csv files in the `splits` folder. 

## Environment

The environment can be generated using either the requirements.txt or environment.yml file. 

## Target Data Generation

The current version of the data generation is curated for my dataset, i.e., the gamma correction values are designed such that they work for the distribution of contrast in my data. All that would have to be edited if the data splits are changed. To generate scar enhanced data for training go to `data_preparation` folder and use:

`python generate.py -o OUTPUT_FOLDER`

where the OUTPUT_FOLDER is the path of where you want the enhanced training data to be saved.

## ROI Segmentation

Another step of our framework is generation of ROI masks. The ROI masks are binary masks, where 1 is the region of interest and 0 is the background. With that, the regions of 1's are being noised during the diffusion process and those of 0's are not. To train the roi unet go to `roi_segmentation` directory and run:

`python train_unet.py -t TRAIN_CSV -v VAL_CSV -r RESUME -c RESUME_PATH -p RESULTS_PATH`

where TRAIN_CSV is the path to the csv file with training data, VAL_CSV is the path to the csv file with validation data, RESUME is a boolean value of True/False that states whether you want to resume the checkpoint, RESUME_PATH is the path to the checkpoint you want to resume from, and RESULT_PATH is the path where you want to save your trained model. Only RESULTS_PATH is required, other parameters are optional and have default values setup such that they work with my csv files and train from scratch. 

When the model is trained, you can generate the ROI masks using:

`python generate_roi_masks.py -m MODEL_PATH -o OUTPUT_PATH`,

where MODEL_PATH is the path to the model you want to use, OUTPUT_PATH is the path where you want to save the masks. 


## ROI Diffusion

When the ROI masks and the target data are both ready, the enhancement training can be started. To train the diffusion model go to the `diffusion` directory. There you can run:

`python diffusion_training.py 42`,

where 42 is the number or the arguments file located in `test_args` folder, specifying the hyperparameters for the training. 

Inference can be run using:

`python inference.py -m MODEL_PATH -o OUTPUT_PATH -r REPEAT -t DISTANCE`

where MODEL_PATH is the path to the model you want to use for inference, OUTPUT_PATH is where you want to save the images, REPEAT is the number of times you want to repeat inference for generation of the mean image (default=5), and DISTANCE is how many noising steps you want to go to for inference ($\lambda$ from the paper).

To generate the mean image, use:

`python make_mean.py -p RESULTS_PATH -s SPLITS -o OUTPUT`

where RESULTS_PATH is the location of the repeated images, SPLITS is the amount of repetitions of inference, and OUTPUT is where you want to save the images. 

