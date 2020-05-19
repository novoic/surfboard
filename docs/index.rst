:github_url: https://github.com/novoic/surfboard

Surfboard: audio-feature extraction for modern machine learning
===============================================================

You can find our paper `on arXiv
<https://arxiv.org/abs/2005.08848>`_.

.. toctree::
    :maxdepth: 2

    intro_installation

Core Surfboard classes: Waveform and Barrel
-------------------------------------------

At the heart of Surfboard lie two classes: the :code:`Waveform` class and the :code:`Barrel` class.

.. toctree::
    :maxdepth: 2

    waveform_barrel


Feature extraction
------------------

An alternative to extracting features with the :code:`Waveform` class is to use functions specifically written for that purpose, either with the vanilla approach, or with the multiprocessing approach.

.. toctree::
    :maxdepth: 2

    feature_extraction


Under the hood
--------------

Under the hood lies a variety of files containing functions which are imported by the :code:`Waveform` class. We split the code as such for the sake of readability.

.. toctree::
    :maxdepth: 2

    under_the_hood


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
