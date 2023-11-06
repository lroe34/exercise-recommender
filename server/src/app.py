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

    recommendations = sorted(enumerate(cosine_similarities[0]), key=lambda x: x[1], reverse=True)

    print(recommendations[0][0])
    return render_template('index.html', recommended=workouts[recommendations[0][0]])
