"""
===========================================================================
FNA Tools
===========================================================================
Main collection of utilities, classes and functions to setup the numerical experiments
"""

__all__ = ['get_import_warning', 'check_dependency', 'parameters', 'analysis', 'network_architect',
           'signals', 'utils', 'visualization']


def get_import_warning(name):
    return """** %s ** package is not installed. To have functions using %s please install the package.""" % (name,
                                                                                                              name)


def check_dependency(name):
    """
    verify if package is installed
    :param name: string with the name of the package to import
    :return:
    """
    try:
        exec("import %s" % name)
        return True
    except ImportError:
        print(get_import_warning(name))
