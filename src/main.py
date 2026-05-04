import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

model=joblib.load('../models/Trained_Dec_tree.pkl')
app = Flask(__name__)





