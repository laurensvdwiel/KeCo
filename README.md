# KeCo #
An implementation to a kernel based co-agreement algorithm, designed for partially-labelled, large-scale datasets
with multiple views that contain non-linear relations.

## About ##

The implementation contained in this repository is a result of the joint work between [Laurens van de Wiel](https://nl.linkedin.com/in/laurensvdwiel), [Tom Heskes](http://www.cs.ru.nl/~tomh/) and [Evgeni Levin](http://www.learning-machines.com/). The work is published in [the 18th International Conference on Discovery Science (DS 2015)](https://ds2015.cs.dal.ca/) and can be found via [doi:10.1007/978-3-319-24282-8_26](http://dx.doi.org/10.1007/978-3-319-24282-8_26).

## Citation ##

If you make use of this software for academic purposes, please cite the article [doi:10.1007/978-3-319-24282-8_26](http://dx.doi.org/10.1007/978-3-319-24282-8_26).

The bibtex for citation is the following

	@incollection{
	year={2015},
	isbn={978-3-319-24281-1},
	booktitle={Discovery Science},
	volume={9356},
	series={Lecture Notes in Computer Science},
	editor={Japkowicz, Nathalie and Matwin, Stan},
	doi={10.1007/978-3-319-24282-8_26},
	title={KeCo: Kernel-Based Online Co-agreement Algorithm},
	url={http://dx.doi.org/10.1007/978-3-319-24282-8_26},
	publisher={Springer International Publishing},
	keywords={Kernel; Non-linear; Online; Large-scale; Semi-supervised; Co-agreement; Multi-view; Classification},
	author={Wiel, Laurens and Heskes, Tom and Levin, Evgeni},
	pages={308-315},
	language={English}
	}

## How to run ##

### Dependencies ###

The code implemented here is tested in a Python 2.7.9 environment, containing packages:

* numpy (v1.9.2),
* scikit-learn (v0.16.1),
* scipy (v0.15.1).

### Example run ###

First ensure you are in the src folder;

	cd .../KeCo/src/

Then you may run the code with the included dataset in the following way:

	python main.py "../datasets/ionosphere_scale.txt"
