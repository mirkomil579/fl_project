from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT='-e .'
def  get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements=file_obj.readlines()
        [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='fl_project',
    version='0.0.1',
    author='Mirkomil',
    author_email='egamberdiyev.mirkomil@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),

)