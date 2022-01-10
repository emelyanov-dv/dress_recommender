from flask import render_template
import connexion

app = connexion.App(__name__, specification_dir="./")
app.add_api('swagger.yaml')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9091, debug=True, threaded=True)
    

