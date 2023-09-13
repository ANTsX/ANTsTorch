from setuptools import setup

long_description = open("README.md").read()

setup(name='deepsimlr',
      version='0.0.0',
      description='Interpretable, similarity-driven multi-view embeddings from high-dimensional biomedical data.',
      long_description=long_description,
      long_description_content_type="text/markdown; charset=UTF-8; variant=GFM",
      url='https://github.com/ntustison/DeepSiMLR',
      author='Brian B. Avants and Nicholas J. Tustison',
      author_email='ntustison@gmail.com',
      packages=['deepsimlr','deepsimlr/architectures','deepsimlr/utilities'],
      install_requires=['antspyx','torch','scikit-learn','numpy','requests','statsmodels','matplotlib','jax','optax','jaxopt'],
      zip_safe=False)
