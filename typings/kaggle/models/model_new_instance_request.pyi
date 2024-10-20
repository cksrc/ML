"""
This type stub file was generated by pyright.
"""

"""
    Kaggle API

    API for kaggle.com  # noqa: E501

    OpenAPI spec version: 1
    
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""
class ModelNewInstanceRequest:
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    swagger_types = ...
    attribute_map = ...
    def __init__(self, instance_slug=..., framework=..., overview=..., usage=..., license_name=..., fine_tunable=..., training_data=..., model_instance_type=..., base_model_instance=..., external_base_model_url=..., files=...) -> None:
        """ModelNewInstanceRequest - a model defined in Swagger"""
        ...
    
    @property
    def instance_slug(self): # -> None:
        """Gets the instance_slug of this ModelNewInstanceRequest.  # noqa: E501

        The slug that the model instance should be created with  # noqa: E501

        :return: The instance_slug of this ModelNewInstanceRequest.  # noqa: E501
        :rtype: str
        """
        ...
    
    @instance_slug.setter
    def instance_slug(self, instance_slug): # -> None:
        """Sets the instance_slug of this ModelNewInstanceRequest.

        The slug that the model instance should be created with  # noqa: E501

        :param instance_slug: The instance_slug of this ModelNewInstanceRequest.  # noqa: E501
        :type: str
        """
        ...
    
    @property
    def framework(self): # -> None:
        """Gets the framework of this ModelNewInstanceRequest.  # noqa: E501

        The framework of the model instance  # noqa: E501

        :return: The framework of this ModelNewInstanceRequest.  # noqa: E501
        :rtype: str
        """
        ...
    
    @framework.setter
    def framework(self, framework): # -> None:
        """Sets the framework of this ModelNewInstanceRequest.

        The framework of the model instance  # noqa: E501

        :param framework: The framework of this ModelNewInstanceRequest.  # noqa: E501
        :type: str
        """
        ...
    
    @property
    def overview(self): # -> None:
        """Gets the overview of this ModelNewInstanceRequest.  # noqa: E501

        The overview of the model instance (markdown)  # noqa: E501

        :return: The overview of this ModelNewInstanceRequest.  # noqa: E501
        :rtype: str
        """
        ...
    
    @overview.setter
    def overview(self, overview): # -> None:
        """Sets the overview of this ModelNewInstanceRequest.

        The overview of the model instance (markdown)  # noqa: E501

        :param overview: The overview of this ModelNewInstanceRequest.  # noqa: E501
        :type: str
        """
        ...
    
    @property
    def usage(self): # -> None:
        """Gets the usage of this ModelNewInstanceRequest.  # noqa: E501

        The description of how to use the model instance (markdown)  # noqa: E501

        :return: The usage of this ModelNewInstanceRequest.  # noqa: E501
        :rtype: str
        """
        ...
    
    @usage.setter
    def usage(self, usage): # -> None:
        """Sets the usage of this ModelNewInstanceRequest.

        The description of how to use the model instance (markdown)  # noqa: E501

        :param usage: The usage of this ModelNewInstanceRequest.  # noqa: E501
        :type: str
        """
        ...
    
    @property
    def license_name(self): # -> None:
        """Gets the license_name of this ModelNewInstanceRequest.  # noqa: E501

        The license that should be associated with the model instance  # noqa: E501

        :return: The license_name of this ModelNewInstanceRequest.  # noqa: E501
        :rtype: str
        """
        ...
    
    @license_name.setter
    def license_name(self, license_name): # -> None:
        """Sets the license_name of this ModelNewInstanceRequest.

        The license that should be associated with the model instance  # noqa: E501

        :param license_name: The license_name of this ModelNewInstanceRequest.  # noqa: E501
        :type: str
        """
        ...
    
    @property
    def fine_tunable(self): # -> None:
        """Gets the fine_tunable of this ModelNewInstanceRequest.  # noqa: E501

        Whether the model instance is fine tunable  # noqa: E501

        :return: The fine_tunable of this ModelNewInstanceRequest.  # noqa: E501
        :rtype: bool
        """
        ...
    
    @fine_tunable.setter
    def fine_tunable(self, fine_tunable): # -> None:
        """Sets the fine_tunable of this ModelNewInstanceRequest.

        Whether the model instance is fine tunable  # noqa: E501

        :param fine_tunable: The fine_tunable of this ModelNewInstanceRequest.  # noqa: E501
        :type: bool
        """
        ...
    
    @property
    def training_data(self): # -> None:
        """Gets the training_data of this ModelNewInstanceRequest.  # noqa: E501

        A list of training data (urls or names)  # noqa: E501

        :return: The training_data of this ModelNewInstanceRequest.  # noqa: E501
        :rtype: list[str]
        """
        ...
    
    @training_data.setter
    def training_data(self, training_data): # -> None:
        """Sets the training_data of this ModelNewInstanceRequest.

        A list of training data (urls or names)  # noqa: E501

        :param training_data: The training_data of this ModelNewInstanceRequest.  # noqa: E501
        :type: list[str]
        """
        ...
    
    @property
    def model_instance_type(self): # -> None:
        """Gets the model_instance_type of this ModelNewInstanceRequest.  # noqa: E501

        Whether the model instance is a base model, external variant, internal variant, or unspecified  # noqa: E501

        :return: The model_instance_type of this ModelNewInstanceRequest.  # noqa: E501
        :rtype: str
        """
        ...
    
    @model_instance_type.setter
    def model_instance_type(self, model_instance_type): # -> None:
        """Sets the model_instance_type of this ModelNewInstanceRequest.

        Whether the model instance is a base model, external variant, internal variant, or unspecified  # noqa: E501

        :param model_instance_type: The model_instance_type of this ModelNewInstanceRequest.  # noqa: E501
        :type: str
        """
        ...
    
    @property
    def base_model_instance(self): # -> None:
        """Gets the base_model_instance of this ModelNewInstanceRequest.  # noqa: E501

        If this is an internal variant, the `{owner-slug}/{model-slug}/{framework}/{instance-slug}` of the base model instance  # noqa: E501

        :return: The base_model_instance of this ModelNewInstanceRequest.  # noqa: E501
        :rtype: str
        """
        ...
    
    @base_model_instance.setter
    def base_model_instance(self, base_model_instance): # -> None:
        """Sets the base_model_instance of this ModelNewInstanceRequest.

        If this is an internal variant, the `{owner-slug}/{model-slug}/{framework}/{instance-slug}` of the base model instance  # noqa: E501

        :param base_model_instance: The base_model_instance of this ModelNewInstanceRequest.  # noqa: E501
        :type: str
        """
        ...
    
    @property
    def external_base_model_url(self): # -> None:
        """Gets the external_base_model_url of this ModelNewInstanceRequest.  # noqa: E501

        If this is an external variant, a URL to the base model  # noqa: E501

        :return: The external_base_model_url of this ModelNewInstanceRequest.  # noqa: E501
        :rtype: int
        """
        ...
    
    @external_base_model_url.setter
    def external_base_model_url(self, external_base_model_url): # -> None:
        """Sets the external_base_model_url of this ModelNewInstanceRequest.

        If this is an external variant, a URL to the base model  # noqa: E501

        :param external_base_model_url: The external_base_model_url of this ModelNewInstanceRequest.  # noqa: E501
        :type: int
        """
        ...
    
    @property
    def files(self): # -> None:
        """Gets the files of this ModelNewInstanceRequest.  # noqa: E501

        A list of files that should be associated with the model instance version  # noqa: E501

        :return: The files of this ModelNewInstanceRequest.  # noqa: E501
        :rtype: list[UploadFile]
        """
        ...
    
    @files.setter
    def files(self, files): # -> None:
        """Sets the files of this ModelNewInstanceRequest.

        A list of files that should be associated with the model instance version  # noqa: E501

        :param files: The files of this ModelNewInstanceRequest.  # noqa: E501
        :type: list[UploadFile]
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
    

