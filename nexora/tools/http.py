from .basetool import BaseTool
import requests

class HTTP(BaseTool):
    name = "http"

    def request(self, method, url, data=None):
        """openai.function: Allows making arbitrary HTTP requests.

        method,url,data

        :param str method: The HTTP method to use (e.g., GET, POST, PUT).
        :param str url: The URL to make the request to.
        :param str data: Optional data to be sent with the request, if applicable.
        """
        try:
            response = requests.request(method.upper(), url, data=data)

            result = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.text
            }
            
            return result
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
