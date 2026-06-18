# Images recorder - Parameters #

Some parameters are needed while recording images. All parameters are defined in [recorder_parameters.hpp](../include/aps/recorder_parameters.hpp).

The parameters are stored under [parameters/](../parameters/) in yaml files.

## CUDA device ##
* cuda_device_uuid [-]: Universially unique identifier (UUID) of CUDA device which will be used for images recording. Run `nvidia-smi -L` to list all available GPUs and their UUID.
