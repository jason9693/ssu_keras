# SSU Keras

### This repository is custom module inherited from tf.keras.layers bundle for tensorflow v2.x

## Install

> pip install ssu_tk

## Simple Usage

> import ssu_tf_keras as stf_keras
>
>
> ed = stf_keras.layers.ExpandDims(axis=-1)(inputs)
>
> ed_model = tf.keras.Model(inputs, ed)
>
> ed_out = ed_model(x)

## Supported Modules List
* Reverse ( like tf.reverse )
* ExpandDims ( like tf.expand_dims )
* ReflectionPadding2D ( like tf.pad(inputs, pattern, mode='REFLECT') )

## License
Project is published under the MIT licence. Feel free to clone and modify repo as you want, but don'y forget to add reference to authors :)
