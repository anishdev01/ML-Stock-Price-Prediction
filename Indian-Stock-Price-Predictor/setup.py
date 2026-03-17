from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function reads a requirements file and returns a list of packages.
    """
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='Indian_Stock_Price_Prediction',
    version='0.1',
    author='Piyush Agarwal',
    author_email='piyushagarwal2003k@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)