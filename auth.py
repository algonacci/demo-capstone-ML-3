from flask_httpauth import HTTPTokenAuth

auth = HTTPTokenAuth(scheme="Bearer")

SECRET_KEY = "secret"


@auth.error_handler
def unauthorized():
    return {
        "status": {
            "code": 401,
            "message": "Unauthorized"
        },
        "data": None,
    }, 401


@auth.verify_token
def verify_token(token):
    return SECRET_KEY == token
