# ElecFF

Thank you for your interest in this project, and I hope that the following presentations will give you a deeper understanding of this research!

![image](ElecFF.png)

## dataset

In the master branch, we have uploaded the JSON files of the four ionic liquids used in this research, which contain information about energy, force, position, atom_number and cell_size for the reader's reference.

## SB_descriptor

bessel_descriptors.py computes the final representation of the atomic environment descriptor, describing the environment as a power spectral density function.

There are three files in the pre folder, they are coefficients.py, functions.py and roots.py, they are used in generating spherical Bessel descriptors.

## model.py

model.py Defines all models used in this research.

## train.py

train.py includes the training parameter phase and the predicted force field phase for all models.
