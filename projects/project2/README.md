Usage guide for main.py

To have all the dependencies, you can run this venv.
$ source .mlenv/bin/activate

$ python main.py
Simple command, you can add following flags :

--load_model, Default: None Used to load a pretrained model from a string path
--learning_rate -lr, Default: 1e-3 learning rate. A scheduler decreasing the learning rate on plateau is programmed.
--batch_size -bs, Default: 64 batch size
--epochs -e, Default: 100 number of epochs. An early stop on plateau is programmed.
--depth -d, Default: 4 the depth of the UNet model
--first_layer_size -fls, Default: 64 Size of the first UNet layer. Size of internal layers are computed automatically.
--record -r, Default: False Used to record logs in .txt file and plots (best, worst, iou curve, loss curve) in ./plots folder. Also saves final mean results in results.csv
--sample -s, Default: 500 Number of training samples used. Maximum is 80 percent of total samples (to save some for validation set)
--fold -f, Default: 3 Number of folds a similar model is trained. If --record, results are averaged over every fold
--annotation -a, Default: 1 Quality of annotation, currently choseable between 1,0.9,0.77,0.67,0.57,0.84

To run the main file, you can also do
$ ./run_main.sh
which runs a bash file that iterates over different parameters, and saves them

