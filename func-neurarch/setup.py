from setuptools import setup

setup(
    name='fna',
    version='0.1',
    packages=['fna', 'fna.tasks', 'fna.tasks.symbolic', 'fna.tools', 'fna.tools.utils', 'fna.tools.signals',
              'fna.tools.analysis', 'fna.tools.visualization', 'fna.tools.network_architect', 'fna.decoders',
              'fna.encoders', 'fna.networks', 'fna.networks.ann', 'fna.networks.rnn', 'fna.networks.snn'],
    url='https://github.com/rcfduarte/func-neurarch',
    license='GPLv2',
    author='Renato Duarte, Barna Zajzon',
    author_email='r.duarte@fz-juelich.de',
    description='Functional Neural Architectures (F:N-A) is a python package that provides a set of tools to build, simulate and analyse neuronal microcircuit models with any degree of complexity, as well as to probe the circuits with arbitrarily complex input stimuli / signals and to analyse the relevant functional aspects of single neuron, population and network dynamics.',
    install_reqs=[
        "librosa",
        "psutil",
        "pandas",
        "tqdm",
        "seaborn",
        "more_itertools",
        "matplotlib>=2.1.0",
        "numpy>=1.14.2",
        "scikit-learn>=0.20.2",
        "scipy>=1.0.1",
        "hickle>=3.3.0",
    ]
)
