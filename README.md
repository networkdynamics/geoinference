# The Network Dynamics Geoinference Library

Geoinference predicts the location from which a piece of text was written.  The
Network Dynamics Geoinference Library is a collection of state-of-the-art
geoinference methods for predicting the locations of posts in
[Twitter](https://twitter.com).  This repository hosts the source code for the
reference implementations evaluated in [Jurgens et
al. (2015)](http://cs.mcgill.ca/~jurgens/docs/jurgens-et-al_icwsm-2015.pdf), all
documentation for the project, and the issue tracker for bugs and feature
requests.

# Why use this library?

  * Reference implmentations for many highly-cited geoinference techniques
  * A flexible API that makes it easy to build new geoinference methods
  * Support for both Java and Python implementations

# Documentation

See our [project
page](https://github.com/networkdynamics/geoinference/wiki/Home) for full
details of the project.  The
[Installation](https://github.com/networkdynamics/geoinference/wiki/Installation)
page has additional for detailed instructions on how to use and extend the
software library.  Also, see our [Frequently Asked
Questions](https://github.com/networkdynamics/geoinference/wiki/Frequently-Asked-Questions)
for additional details documentation.

# Related Projects

This repository connects to a multi-part effort for geoinference in social
media.  The Network Dynamics FREESR project (described
[here](http://cs.mcgill.ca/~jurgens/docs/jurgens-et-al_spsm-2015.pdf)) aims to
allow social media researchers anywhere to test and evaluate their methods on
the same datasets.  See the [FREESR] website (_forthcoming_) for full details.

# Credits

The Geoinference library was made possible through the development efforts of many people.

  * [David Jurgens](http://cs.mcgill.ca/~jurgens), McGill University
  * Tyler Finethy, McGill University
  * James McCorriston, McGill University
  * Yi Tian Xu, McGill University
  * [Derek Ruths](http://www.pilevar.com/taher/), McGill University

We especially thank the original authors of the papers which are implemented in
the library for their work and occasional feedback on how the algorithms were
originally implemented.

We kindly ask that if you use this library in a piece of academic work, that you
cite the paper associated with it.

    @inproceedings{jurgens2015geolocation,
        title={Geolocation Prediction in Twitter Using Social Networks: A Critical Analysis and Review of Current Practice},
        author={David Jurgens and Tyler Finethy and James McCorriston and Xu, Yi Tian and Derek Ruths},
        booktitle={Proceedings of the 9th International AAAI Conference on Weblogs and Social Media (ICWSM)},
        year={2015}
    }

# Contact

If you have discovered a bug in the build software or want to report an error in
the library, please create a new
[Issue](https://github.com/networkdynamics/geoinference/issues) on our github page.
# geoinference
