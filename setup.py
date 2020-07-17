from setuptools import setup

setup(
    name="surfboard",
    version="0.2.0",
    description="Novoic's audio feature extraction library https://novoic.com",
    url="http://github.com/novoic/surfboard",
    author="Raphael Lenain",
    author_email="raphael@novoic.com",
    license="GPL-3.0",
    packages=["surfboard"],
    keywords=[
        "feature-extraction",
        "audio",
        "machine-learning",
        "audio-processing",
        "python",
        "speech-processing",
        "healthcare",
        "signal-processing",
        "alzheimers-disease",
        "parkinsons-disease",
    ],
    download_url="https://github.com/novoic/surfboard/archive/v0.2.0.tar.gz",
    install_requires=[
        "librosa>=0.7.2",
        "numba==0.48.0", # Needed until Librosa deploys fix to mute warnings.
        "pysptk>=0.1.18",
        "PeakUtils>=1.3.3",
        "pyloudnorm==0.1.0",
        "pandas>=1.0.1",
        "tqdm>=4.42.1",
        "pyyaml>=5.3",
        "Cython>=0.29.15",
        "pytest>=5.4.1",
        "SoundFile>=0.10.3.post1",
    ],
    scripts=['bin/surfboard'],
    zip_safe=False,
    classifiers=[
	'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
	'Intended Audience :: Developers',      # Define that your audience are developers
	'Topic :: Software Development :: Build Tools',
	'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',   # Again, pick a license
	'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
	'Programming Language :: Python :: 3.4',
	'Programming Language :: Python :: 3.5',
	'Programming Language :: Python :: 3.6',
	'Programming Language :: Python :: 3.7',
  ],
)
