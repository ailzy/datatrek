from distutils.core import setup

setup(
    name = "datatrek",
    version = "0.0.1",
    author = "Qiang Luo",
    author_email = "luoq08@gmail.com",
    description = "data analysis toolkit",
    packages = ["datatrek",
                "datatrek.sklearn_addon",
                "datatrek.sklearn_addon.transformation",
		],
)    
