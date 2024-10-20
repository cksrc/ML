"""
This type stub file was generated by pyright.
"""

"""
    Kaggle API

    API for kaggle.com  # noqa: E501

    OpenAPI spec version: 1
    
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""
class CreateInboxFileRequest:
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    swagger_types = ...
    attribute_map = ...
    def __init__(self, virtual_directory=..., blob_file_token=...) -> None:
        """CreateInboxFileRequest - a model defined in Swagger"""
        ...
    
    @property
    def virtual_directory(self): # -> None:
        """Gets the virtual_directory of this CreateInboxFileRequest.  # noqa: E501

        Directory name used for tagging the uploaded file  # noqa: E501

        :return: The virtual_directory of this CreateInboxFileRequest.  # noqa: E501
        :rtype: str
        """
        ...
    
    @virtual_directory.setter
    def virtual_directory(self, virtual_directory): # -> None:
        """Sets the virtual_directory of this CreateInboxFileRequest.

        Directory name used for tagging the uploaded file  # noqa: E501

        :param virtual_directory: The virtual_directory of this CreateInboxFileRequest.  # noqa: E501
        :type: str
        """
        ...
    
    @property
    def blob_file_token(self): # -> None:
        """Gets the blob_file_token of this CreateInboxFileRequest.  # noqa: E501

        Token representing the uploaded file  # noqa: E501

        :return: The blob_file_token of this CreateInboxFileRequest.  # noqa: E501
        :rtype: str
        """
        ...
    
    @blob_file_token.setter
    def blob_file_token(self, blob_file_token): # -> None:
        """Sets the blob_file_token of this CreateInboxFileRequest.

        Token representing the uploaded file  # noqa: E501

        :param blob_file_token: The blob_file_token of this CreateInboxFileRequest.  # noqa: E501
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
    

