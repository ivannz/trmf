from distutils.core import setup, Extension

setup(
    name="trmf",
    version="0.9",
    description="""A pythonic library of matrix decomposition methods """
                """for multivariate time-series.""",
    license="MIT License",
    author="Ivan Nazarov",
    author_email="ivan.nazarov@skolkovotech.ru",
    packages=["trmf"],
    requires=["numpy", "numba", "scipy"]
)
