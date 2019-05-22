from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup


setup_args = generate_distutils_setup(
    name='face_recognition_ros',
    description='Facial recognition node for ROS. Using Tensorflow and Facenet',
    author='Samuel Felipe Baltanas Molero',
    author_email='samuelbaltanas@gmail.com',
    version='0.0.0',
    scripts=['scripts/compare_faces_demo', 'scripts/save_embedings'],
    packages=['face_recognition_ros', 'utils', 'facenet'],
    package_dir={
        '': 'src'#,
        #'facenet': 'facenet/src/facenet'
    }
)

setup(**setup_args)
