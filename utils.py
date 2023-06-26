

class Singleton(type):
    """Metaclass implementing the Singleton pattern.

    This metaclass ensures that only one instance of a class is created and shared among all instances.

    Attributes:
        _instances (dict): Dictionary holding the unique instances of each class.

    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        """Overrides the call behavior when creating an instance.

        This method checks if an instance of the class already exists. If not, it creates a new instance and
        stores it in the _instances dictionary.

        Args:
            cls (type): Class type.

        Returns:
            object: The instance of the class.

        """
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]