from flask import Flask

# app = Flask(__name__, template_folder='./templates')
app = Flask(__name__, static_folder="../static", static_url_path="/static", template_folder="./templates")

from app import routes  # Import routes to register endpoints

def create_app():
    return app
