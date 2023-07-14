# latent-control-idea
### clone dependencies
```
git clone https://github.com/pac48/latent-control-idea.git
git clone https://github.com/pac48/matlab_mex.git
git clone https://github.com/pac48/instant-ngp.git
```
## setup
### build
1. Follow instructions in matlab_mex to setup matlab 
2. Build instant-ngp from source

### create NERF models
You need to create NERF models using the GUI for instant-ngp. Once a model is trained, save it in the data folder, for example `data/nerf/objects/`. Then create a .sh file to run the NERF for each object you train, for example run_book.sh looks like this:

```
../scripts/run.py --zmq_port 5565 --zmq_topic nerf_book --load_snapshot ../data/nerf/objects/book.msgpack --mode nerf
``` 
 
It is important that the zmq_port 5565 matches the values set in MATLAB. Then run all of the .sh files in their own terminal from the scripts folder.

### LofTR
Run the following from the latent-control-idea/LoFTR/demo directory 

`demo_loftr.py --weight ../weights/indoor_ds_new.ckpt --input 0`

You will need to downlaod the model weights.   

### train the network
You need to run main.m with MATLAB to train the network

