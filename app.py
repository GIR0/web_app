from flask import Flask, render_template, url_for, request, redirect
from helper import get_result

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        material_id = request.form['content']
        print(material_id)
        if material_id == '':
            return redirect('/')
        else:
            return redirect(f'/predict/{material_id}')
    else:
        return render_template('index.html')


@app.route('/predict/<id>', methods=['GET', 'POST'])
def predict(id):
    if request.method == 'POST':
        return redirect('/')
    else:
        try:
            result, pred = get_result(id)
            return render_template('predict.html', result=result, pred=pred)
        except:
            return redirect('/invalid_id')

@app.route('/invalid_id', methods=['GET', 'POST'])
def invalid_id():
    if request.method == 'POST':
        return redirect('/')
    else:
        return render_template('invalid.html')

@app.route('/credits', methods=['GET', 'POST'])
def credits():
    if request.method == 'POST':
        return redirect('/')
    else:
        return render_template('credits.html')

if __name__ == "__main__":
    app.run(debug=True)
