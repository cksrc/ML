"""
This type stub file was generated by pyright.
"""

import io

"""
    Kaggle API

    API for kaggle.com  # noqa: E501

    OpenAPI spec version: 1
    
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""
logger = ...
class RESTResponse(io.IOBase):
    def __init__(self, resp) -> None:
        ...
    
    def getheaders(self):
        """Returns a dictionary of the response headers."""
        ...
    
    def getheader(self, name, default=...):
        """Returns a given response header."""
        ...
    


class RESTClientObject:
    def __init__(self, configuration, pools_size=..., maxsize=...) -> None:
        ...
    
    def request(self, method, url, query_params=..., headers=..., body=..., post_params=..., _preload_content=..., _request_timeout=...):
        """Perform requests.

        :param method: http request method
        :param url: http request url
        :param query_params: query parameters in the url
        :param headers: http request headers
        :param body: request json body, for `application/json`
        :param post_params: request post parameters,
                            `application/x-www-form-urlencoded`
                            and `multipart/form-data`
        :param _preload_content: if False, the urllib3.HTTPResponse object will
                                 be returned without reading/decoding response
                                 data. Default is True.
        :param _request_timeout: timeout setting for this request. If one
                                 number provided, it will be total request
                                 timeout. It can also be a pair (tuple) of
                                 (connection, read) timeouts.
        """
        ...
    
    def GET(self, url, headers=..., query_params=..., _preload_content=..., _request_timeout=...):
        ...
    
    def HEAD(self, url, headers=..., query_params=..., _preload_content=..., _request_timeout=...):
        ...
    
    def OPTIONS(self, url, headers=..., query_params=..., post_params=..., body=..., _preload_content=..., _request_timeout=...):
        ...
    
    def DELETE(self, url, headers=..., query_params=..., body=..., _preload_content=..., _request_timeout=...):
        ...
    
    def POST(self, url, headers=..., query_params=..., post_params=..., body=..., _preload_content=..., _request_timeout=...):
        ...
    
    def PUT(self, url, headers=..., query_params=..., post_params=..., body=..., _preload_content=..., _request_timeout=...):
        ...
    
    def PATCH(self, url, headers=..., query_params=..., post_params=..., body=..., _preload_content=..., _request_timeout=...):
        ...
    


class ApiException(Exception):
    def __init__(self, status=..., reason=..., http_resp=...) -> None:
        ...
    
    def __str__(self) -> str:
        """Custom error messages for exception"""
        ...
    

