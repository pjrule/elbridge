""" Custom errors for elbridge. """

class ContiguityError(Exception):
    """
    Raised when an invalid contiguity criterion is passed
    to the Plan constructor.
    """