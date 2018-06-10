import abc
import ast


class Parameter:

    def __init__(self, parser=None, **kwargs):
        """
        Parameter options to pass on to the add_argument method of the ArgumentParser object from the argparse module
        :param action:      The basic type of action to be taken when this argument is encountered at the command line.
        :param nargs:       The number of command-line arguments that should be consumed.
        :param default:     The value produced if the argument is absent from the command line.
        :param type:        The type to which the command-line argument should be converted.
        :param choices:     A container of the allowable values for the argument.
        :param required:    Whether or not the command-line option may be omitted
        :param help:        A brief description of what the argument does
        """
        assert all(k in ['action', 'nargs', 'default', 'type', 'choices', 'required', 'help'] for k in kwargs.keys())
        self._options = kwargs
        self.parser = parser

    @property
    def options(self):
        return self._options

    def has_parser(self):
        return self.parser is not None

    def parse(self, arg):
        return self.parser.parse(arg)

    def default(self):
        return self._options['default']


class Parser(abc.ABC):

    @abc.abstractmethod
    def parse(self, x):
        """ Parses input """


class LiteralParser(Parser):

    def parse(self, x):
        return ast.literal_eval(x)