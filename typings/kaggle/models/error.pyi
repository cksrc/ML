"""
This type stub file was generated by pyright.
"""

"""
    Kaggle API

    API for kaggle.com  # noqa: E501

    OpenAPI spec version: 1
    
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""
class Error:
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    swagger_types = ...
    attribute_map = ...
    def __init__(self, code=..., message=...) -> None:
        """Error - a model defined in Swagger"""
        ...
    
    @property
    def code(self): # -> None:
        """Gets the code of this Error.  # noqa: E501

        The server error code returned  # noqa: E501

        :return: The code of this Error.  # noqa: E501
        :rtype: int
        """
        ...
    
    @code.setter
    def code(self, code): # -> None:
        """Sets the code of this Error.

        The server error code returned  # noqa: E501

        :param code: The code of this Error.  # noqa: E501
        :type: int
        """
        ...
    
    @property
    def message(self): # -> None:
        """Gets the message of this Error.  # noqa: E501

        The error message generated by the server  # noqa: E501

        :return: The message of this Error.  # noqa: E501
        :rtype: str
        """
        ...
    
    @message.setter
    def message(self, message): # -> None:
        """Sets the message of this Error.

        The error message generated by the server  # noqa: E501

        :param message: The message of this Error.  # noqa: E501
        :type: str
        """
        ...
    
    def to_dict(self): # -> dict[Any, Any]:
        """Returns the model properties as a dict"""
        ...
    
    def to_str(self): # -> str:
        """Returns the string representation of the model"""
        ...
    
    def __repr__(self): # -> str:
        """For `print` and `pprint`"""
        ...
    
    def __eq__(self, other) -> bool:
        """Returns true if both objects are equal"""
        ...
    
    def __ne__(self, other) -> bool:
        """Returns true if both objects are not equal"""
        ...
    

