from flask import Flask, redirect, url_for, render_template, request
import os
from index import d_dtcn
from face_data import func


secret_key = str(os.urandom(24))

app = Flask(__name__)
app.config['TESTING'] = True
app.config['DEBUG'] = True
app.config['FLASK_ENV'] = 'development'
app.config['SECRET_KEY'] = secret_key
app.config['DEBUG'] = True

# Defining the home page of our site


@app.route("/", methods=['GET', 'POST'])
def home():
    print(request.method,"home")
    if request.method == 'POST':
        if request.form.get('Continue') == 'Continue':
            return render_template("test1.html")
    else:
        # pass # unknown
        return render_template("index.html")


@app.route("/start", methods=['GET', 'POST'])
def index():
    print(request.method,"start")
    if request.method == 'POST':
        if request.form.get('Start') == 'Start':
            # pass
            d_dtcn()

            return render_template("test1.html")
    else:
        # pass # unknown
        return render_template("test1.html")


@app.route('/contact', methods=['GET', 'POST'])
def cool_form():
    print(request.method,"contact")
    if request.method == 'POST':
        # do stuff when the form is submitted

        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('index'))

    # show the form, it wasn't submitted
    return render_template('contact.html')

@app.route('/result', methods=['POST', 'GET'])
def result():
    output = request.form.to_dict()
    print(output)
    name = output["name"]
    DL = output["Drivinglicsence"]
    func(name)

    return render_template("driver_registration_form.html", name=name, Drivinglicsence=DL)

@app.route('/driver_registration_form', methods=['POST', 'GET'])
def showfunc():
    print(request.method)
    return render_template("driver_registration_form.html")


# @app.route('/Thankyou_for_registration')
# def thankyou():
#     return render_template("Thankyou_for_registration.html")

if __name__ == "__main__":
    app.run()
    
