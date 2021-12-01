# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

class Block:
    """
    Base class for online data processing blocks.
    """
    def __init__(self):
        self.params = {}
        self.ignoreUnknownParams = False

    def command(self, command):  # noqa
        """
        This function is called to handle commands, i.e., events that change the state of the block in a certain way.

        Commands are always lists and the first entry is the command name. One common example for a command is
        ``['reset']``. The default implementation ignores all commands.

        :param command: Command as list. The first element is a string with the command name. Optional further elements
            depend on the command.
        :return: True if the command is handled by the block, False if the command was ignored
        """
        return False

    def setParams(self, params):
        """
        This function is called to update parameters, i.e., variables that influence how the input data is processed.

        The default implementation updates the ``params`` property if a corresponding key already exists and raises a
        RuntimeError on other parameters (unless ignoreUnknownParams is set to True). If this is not enough, this
        function can be overridden to implement custom logic if parameters change.

        :param params: dictionary with the parameters to update
        :return: None
        """
        for k in params:
            if k in self.params:
                self.params[k] = params[k]
            elif not self.ignoreUnknownParams:
                raise RuntimeError(f'invalid parameter name: "{k}" (valid names: {list(self.params.keys())})')

    def step(self, *args, **kwargs):  # noqa
        """
        Performs one data processing step.

        The default implementation just returns None.

        :param args: application-specific inputs
        :param kwargs: application-specific inputs
        :return: application-specific outputs
        """
        return None
