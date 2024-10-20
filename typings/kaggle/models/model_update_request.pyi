"""
This type stub file was generated by pyright.
"""

"""
    Kaggle API

    API for kaggle.com  # noqa: E501

    OpenAPI spec version: 1
    
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""
class ModelUpdateRequest:
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    swagger_types = ...
    attribute_map = ...
    def __init__(self, title=..., subtitle=..., is_private=..., description=..., publish_time=..., provenance_sources=..., update_mask=...) -> None:
        """ModelUpdateRequest - a model defined in Swagger"""
        ...
    
    @property
    def title(self): # -> None:
        """Gets the title of this ModelUpdateRequest.  # noqa: E501

        The title of the new model  # noqa: E501

        :return: The title of this ModelUpdateRequest.  # noqa: E501
        :rtype: str
        """
        ...
    
    @title.setter
    def title(self, title): # -> None:
        """Sets the title of this ModelUpdateRequest.

        The title of the new model  # noqa: E501

        :param title: The title of this ModelUpdateRequest.  # noqa: E501
        :type: str
        """
        ...
    
    @property
    def subtitle(self): # -> None:
        """Gets the subtitle of this ModelUpdateRequest.  # noqa: E501

        The subtitle of the new model  # noqa: E501

        :return: The subtitle of this ModelUpdateRequest.  # noqa: E501
        :rtype: str
        """
        ...
    
    @subtitle.setter
    def subtitle(self, subtitle): # -> None:
        """Sets the subtitle of this ModelUpdateRequest.

        The subtitle of the new model  # noqa: E501

        :param subtitle: The subtitle of this ModelUpdateRequest.  # noqa: E501
        :type: str
        """
        ...
    
    @property
    def is_private(self): # -> None:
        """Gets the is_private of this ModelUpdateRequest.  # noqa: E501

        Whether or not the model should be private  # noqa: E501

        :return: The is_private of this ModelUpdateRequest.  # noqa: E501
        :rtype: bool
        """
        ...
    
    @is_private.setter
    def is_private(self, is_private): # -> None:
        """Sets the is_private of this ModelUpdateRequest.

        Whether or not the model should be private  # noqa: E501

        :param is_private: The is_private of this ModelUpdateRequest.  # noqa: E501
        :type: bool
        """
        ...
    
    @property
    def description(self): # -> None:
        """Gets the description of this ModelUpdateRequest.  # noqa: E501

        The description to be set on the model  # noqa: E501

        :return: The description of this ModelUpdateRequest.  # noqa: E501
        :rtype: str
        """
        ...
    
    @description.setter
    def description(self, description): # -> None:
        """Sets the description of this ModelUpdateRequest.

        The description to be set on the model  # noqa: E501

        :param description: The description of this ModelUpdateRequest.  # noqa: E501
        :type: str
        """
        ...
    
    @property
    def publish_time(self): # -> None:
        """Gets the publish_time of this ModelUpdateRequest.  # noqa: E501

        When the model was initially published  # noqa: E501

        :return: The publish_time of this ModelUpdateRequest.  # noqa: E501
        :rtype: date
        """
        ...
    
    @publish_time.setter
    def publish_time(self, publish_time): # -> None:
        """Sets the publish_time of this ModelUpdateRequest.

        When the model was initially published  # noqa: E501

        :param publish_time: The publish_time of this ModelUpdateRequest.  # noqa: E501
        :type: date
        """
        ...
    
    @property
    def provenance_sources(self): # -> None:
        """Gets the provenance_sources of this ModelUpdateRequest.  # noqa: E501

        The provenance sources to be set on the model  # noqa: E501

        :return: The provenance_sources of this ModelUpdateRequest.  # noqa: E501
        :rtype: str
        """
        ...
    
    @provenance_sources.setter
    def provenance_sources(self, provenance_sources): # -> None:
        """Sets the provenance_sources of this ModelUpdateRequest.

        The provenance sources to be set on the model  # noqa: E501

        :param provenance_sources: The provenance_sources of this ModelUpdateRequest.  # noqa: E501
        :type: str
        """
        ...
    
    @property
    def update_mask(self): # -> None:
        """Gets the update_mask of this ModelUpdateRequest.  # noqa: E501

        Describes which fields to update  # noqa: E501

        :return: The update_mask of this ModelUpdateRequest.  # noqa: E501
        :rtype: str
        """
        ...
    
    @update_mask.setter
    def update_mask(self, update_mask): # -> None:
        """Sets the update_mask of this ModelUpdateRequest.

        Describes which fields to update  # noqa: E501

        :param update_mask: The update_mask of this ModelUpdateRequest.  # noqa: E501
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
    


