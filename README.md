# KeCo #
An implementation to a kernel based co-agreement algorithm, designed for partially-labelled, large-scale datasets
with multiple views that contain non-linear relations.

## About ##

The implementation contained in this repository is a result of the joint work between [Laurens van de Wiel](https://nl.linkedin.com/in/laurensvdwiel), [Tom Heskes](http://www.cs.ru.nl/~tomh/) and [Evgeni Levin](http://www.learning-machines.com/). The work is published in [the 18th International Conference on Discovery Science (DS 2015)](https://ds2015.cs.dal.ca/).

## How to run ##

### Dependencies ###

The code implemented here is tested in a Python 2.7.9 environment, containing packages:

* numpy (v1.9.2),
* scikit-learn (v0.16.1),
* scipy (v0.15.1).

### Example run ###

First ensure you are in the folder;

	cd .../KeCo/src/

Then you may run the code with the included dataset in the following way:

	python main.py "../datasets/ionosphere_scale.txt"
