"""Docstring for the debug.py module

This module implements a custom dabble Node class for debugging purposes.

Usage
-----
This module should be part of a package that follows the file structure as specified by the
[PeekingDuck documentation](https://peekingduck.readthedocs.io/en/stable/tutorials/03_custom_nodes.html).

Navigate to the root directory of the package and run the following line on the terminal:

```
peekingduck run
```

For more information on debugging nodes, visit [the official PeekingDuck tutorial](https://peekingduck.readthedocs.io/en/stable/tutorials/03_custom_nodes.html#recipe-3-debugging).
"""  # pylint: disable=line-too-long

from typing import Any, Mapping, Optional

from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):
    """Custom Node class for debugging

    Methods
    -------
    run : dict
        Provides debugging log
    """

    def __init__(
            self,
            config: Optional[Mapping[str, Any]] = None,
            **kwargs
    ) -> None:
        """Initialises the custom Node class

        Parameters
        ----------
        config : dict, optional
            Node custom configuration

        Other Parameters
        ----------------
        **kwargs
            Keyword arguments for instantiating the AbstractNode parent class
        """

        super().__init__(config, node_path=__name__, **kwargs)  # type: ignore

    def run(
            self,
            inputs: Mapping[str, Any]
    ) -> Mapping:
        """Provides debugging log

        Parameters
        ----------
        inputs : dict
            Dictionary with the following keys:

            - 'all' - all keys in the current pipeline

        Returns
        -------
        dict
            An empty dictionary
        """

        debug_msg = '\n--     debug    --'
        for key, value in inputs.items():

            # Skip over img and filename
            if key == 'img' or key == 'filename':
                continue

            debug_msg += f'\n\t{key}: {value}'
        debug_msg += '\n-- end of debug --\n'

        # Return debugging log
        self.logger.info(debug_msg)
        return {}


if __name__ == '__main__':
    pass
