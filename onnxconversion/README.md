Please make sure to follow the following steps:
1. convert rllib to ".h5"(keras) file:
- run `scripts/model_conversion/rllib_model_to_keras.py` by passing the experiment directory and checkpoint number. In this case, you also need to modify `control = np.copy(env.env_objs['deputy'].current_control)` to `control = np.copy(env.env_objs['wingman'].current_control)` 
2. Move the ".h5" file from experiment directory to this folder and run `converth5onnx.py` to get `f16dubins.onnx`
3. move your config file (e.g.: `rejoin_f16.yaml`)and then run `plot_onnx_runrollout.py` to get the plot