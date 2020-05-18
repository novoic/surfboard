from setuptools import setup

setup(
    name="surfboard",
    version="0.1",
    description="The Novoic acoustic feature extraction package.",
    url="http://github.com/novoic/surfboard",
    author="Raphael Lenain",
    author_email="raphael@novoic.com",
    license="GPL-3.0",
    packages=["surfboard"],
    install_requires=[
        "librosa==0.7.2",
        "pysptk==0.1.18",
        "PeakUtils==1.3.3",
        "pyloudnorm==0.1.0",
        "pandas==1.0.1",
        "tqdm==4.42.1",
        "pyyaml==5.3",
        "Cython==0.29.15",
        "pytest==5.4.1",
        "SoundFile==0.10.3.post1",
    ],
    scripts=['bin/surfboard'],
    zip_safe=False,
)
