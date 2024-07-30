from setuptools import setup, find_packages

setup(
    name='patient_outcome_predictor',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'numpy',
    ],
    author='Wachira Crispine Mwangi',
    author_email='chrismwangicw@gmail.com',
    description='A package to predict patient outcomes using machine learning',
    url='https://github.com/chrispin-m/patient_outcome_predictor',
)
