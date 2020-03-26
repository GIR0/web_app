from flask import Flask, render_template, url_for, request, redirect
from processing import model_prediction

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        material_id = request.form['content']

        try:
            return redirect(f'/predict/{material_id}')
        except:
            return 'Invalid material_id'

    else:
        return render_template('index.html')


@app.route('/predict/<id>', methods=['GET', 'POST'])
def predict(id):
    if request.method == 'POST':
        return redirect('/')
    else:
        result = model_prediction(id)
        print(result)
        return render_template('predict.html', result=result)

@app.route('/credits', methods=['GET', 'POST'])
def credits():
    if request.method == 'POST':
        return redirect('/')
    else:
        return render_template('credits.html')

if __name__ == "__main__":
    app.run(debug=True)
