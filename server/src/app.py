from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/process_form', methods=['POST', 'GET'])
def process_form():
    muscle_group = request.form.get('muscleGroup')
    skill_level = request.form.get('skillLevel')
    user_profile = "{recommendedperweek:2-3,musclegroup:" + f"{muscle_group}" + ",difficulty:" f"{skill_level}"+ "}"
    # Process the form data here
    # read from the txt file
    with open('ex.txt', 'r') as file:
        contents = file.read().replace('\n', '')
        contents = contents.replace("'", "")
        contents = contents.replace("```", "")
        contents = contents.replace("json", "")


    workouts = contents.split(", ")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(workouts)

    cosine_similarities = linear_kernel(tfidf_vectorizer.transform([user_profile]), tfidf_matrix)

    recommendations = sorted(enumerate(cosine_similarities[0]), key=lambda x: x[1], reverse=True)[:6]

    workout_details = []
    for rec in recommendations:
        full_recommendation = workouts[rec[0]]
        workout_name = full_recommendation.split(":")[0][1:].title().replace("{", " ")
        per_week = full_recommendation.split(":")[2].split(',')[0]
        muscle_group = full_recommendation.split(":")[3].split(',')[:-1]
        difficulty = full_recommendation.split(":")[4].replace("difficulty:", "").replace("}", "")
        workout_details.append((workout_name, per_week, muscle_group, difficulty))

    return render_template('index.html', recommended=workout_details)