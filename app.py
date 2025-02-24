import flask
from flask import render_template , jsonify , request ,send_file
from src.logger import logging as lg
import os, sys
from src.exception import CustomException

from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify("home")


@app.rout("/")
def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        
        return "Training Completed"
    
    except Exception as e:
        raise CustomException(e , sys) from e
    
@app.route('/predict' , methods =['POST' , 'GET'])
def predict():
    
    try:
        if request.method == "Post":
            prediction_pipeline = PredictionPipeline(request)
            prediction_file_detail = prediction_pipeline.run_pipeline()
            
            
            lg.info("Prediction Completed. Downloading Prediction File")
            return send_file(prediction_file_detail.prediction_file_path, download_name=prediction_file_detail.prediction_file_name,as_attachment=True)
        else:
            return render_template('prediction.html')
    
    except Exception as e:
        raise CustomException(e , sys) from e 
    
    
if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 500 , debug = True)      
    

